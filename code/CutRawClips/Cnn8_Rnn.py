import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from torchaudio import transforms


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value 
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    if(framewise_output.ndim == 3):
      pad = framewise_output[:, -1 :, :].repeat(1, frames_num - framewise_output.shape[1], 1)
      """tensor for padding"""

      output = torch.cat((framewise_output, pad), dim=1)
      """(batch_size, frames_num, classes_num)"""

    elif(framewise_output.ndim == 2):
      pad = framewise_output[:, -1 :].repeat(1, frames_num - framewise_output.shape[1])
      """tensor for padding"""

      output = torch.cat((framewise_output, pad), dim=1)
      """(batch_size, frames_num, classes_num)"""

    return output


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x


class LinearSoftmax(nn.Module):
    def __init__(self, pooldim=1):
        super().__init__()
        self.pooldim = pooldim

    def forward(self, time_decision):
        return (time_decision**2).sum(self.pooldim) / time_decision.sum(
            self.pooldim)


class Cnn8_Rnn(nn.Module):
    def __init__(self, classes_num):
        
        super(Cnn8_Rnn, self).__init__()

        self.time_resolution = 0.01
        self.interpolate_ratio = 4     # Downsampled ratio

        # Logmel spectrogram extractor
        self.melspec_extractor = transforms.MelSpectrogram(
            sample_rate=32000,
            n_fft=1024,
            win_length=1024,
            hop_length=320,
            f_min=50,
            f_max=14000,
            n_mels=64,
            norm="slaney",
            mel_scale="slaney"
        )
        self.db_transform = transforms.AmplitudeToDB()
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)


        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.rnn = nn.GRU(512, 256, bidirectional=True, batch_first=True)
        self.fc_audioset = nn.Linear(512, classes_num, bias=True)
        self.temporal_pooling = LinearSoftmax()
        
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
 
    def forward(self, input, mixup_lambda=None,
                specaug=False, segment_length=None):
        """
        Input: (batch_size, n_samples)"""

        x = self.melspec_extractor(input)
        x = self.db_transform(x)    # (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)   # (batch_size, time_steps, mel_bins)
        x = x.unsqueeze(1)  # (batch_size, channel, time_steps, mel_bins)

        frames_num = x.shape[2]
        
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        # SpecAugment
        if self.training and specaug:
            x = self.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg+max')
        x = F.dropout(x, p=0.2, training=self.training) # (batch_size, 256, time_steps / 4, mel_bins / 16)
        x = torch.mean(x, dim=3)
        
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        x, _  = self.rnn(x)
        segmentwise_output = torch.sigmoid(self.fc_audioset(x)).clamp(1e-7, 1.)
        clipwise_output = self.temporal_pooling(segmentwise_output)

        if segment_length is None:
            # Get framewise output
            framewise_output = interpolate(segmentwise_output,
                                           self.interpolate_ratio)
            framewise_output = pad_framewise_output(framewise_output, frames_num)

            output_dict = {
                'framewise_output': framewise_output,
                'clipwise_output': clipwise_output
            }
        else:
            segment_seq_length = segment_length / (self.time_resolution * \
                self.interpolate_ratio)
            segmentwise_outputs = []
            for i in range(
                math.floor(
                    segmentwise_output.size(1) / segment_seq_length
                )
            ):
                start = round(i * segment_seq_length)
                end = round((i + 1) * segment_seq_length)
                segment_output = self.temporal_pooling(
                    segmentwise_output[:, start: end, :])
                segmentwise_outputs.append(segment_output)
            segmentwise_outputs = torch.stack(segmentwise_outputs).transpose(0, 1)
            output_dict = {
                "clipwise_output": clipwise_output,
                "segmentwise_output": segmentwise_outputs
            }

        return output_dict