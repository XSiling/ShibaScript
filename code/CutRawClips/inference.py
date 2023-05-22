import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from pathlib import Path, PosixPath
from tqdm import tqdm
import argparse
from Cnn8_Rnn import Cnn8_Rnn
import pdb
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ----------utils----------


def create_folder(fd):
    """ 检查并创建文件夹"""
    if not os.path.exists(fd):
        os.makedirs(fd)


# ----------sed_utils----------


def find_contiguous_regions(activity_array):
    """Find contiguous regions from bool valued numpy.array.
    Copy of https://dcase-repo.github.io/dcase_util/_modules/dcase_util/data/decisions.html#DecisionEncoder

    Reason is:
    1. This does not belong to a class necessarily
    2. Import DecisionEncoder requires sndfile over some other imports..which causes some problems on clusters

    """

    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:],
                                    activity_array[:-1]).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def double_threshold(x, high_thres, low_thres, n_connect=1):
    """double_threshold
    Helper function to calculate double threshold for n-dim arrays

    :param x: input array
    :param high_thres: high threshold value
    :param low_thres: Low threshold value
    :param n_connect: Distance of <= n clusters will be merged
    """
    assert x.ndim <= 3, "Whoops something went wrong with the input ({}), check if its <= 3 dims".format(
        x.shape)
    if x.ndim == 3:
        apply_dim = 1
    elif x.ndim < 3:
        apply_dim = 0
    # x is assumed to be 3d: (batch, time, dim)
    # Assumed to be 2d : (time, dim)
    # Assumed to be 1d : (time)
    # time axis is therefore at 1 for 3d and 0 for 2d (
    return np.apply_along_axis(lambda x: _double_threshold(
        x, high_thres, low_thres, n_connect=n_connect),
                               axis=apply_dim,
                               arr=x)


def _double_threshold(x, high_thres, low_thres, n_connect=1, return_arr=True):
    """_double_threshold
    Computes a double threshold over the input array

    :param x: input array, needs to be 1d
    :param high_thres: High threshold over the array
    :param low_thres: Low threshold over the array
    :param n_connect: Postprocessing, maximal distance between clusters to connect
    :param return_arr: By default this function returns the filtered indiced, but if return_arr = True it returns an array of tsame size as x filled with ones and zeros.
    """
    assert x.ndim == 1, "Input needs to be 1d"
    high_locations = np.where(x > high_thres)[0]
    locations = x > low_thres
    encoded_pairs = find_contiguous_regions(locations)

    filtered_list = list(
        filter(
            lambda pair:
            ((pair[0] <= high_locations) & (high_locations <= pair[1])).any(),
            encoded_pairs))

    filtered_list = connect_(filtered_list, n_connect)
    if return_arr:
        zero_one_arr = np.zeros_like(x, dtype=int)
        for sl in filtered_list:
            zero_one_arr[sl[0]:sl[1]] = 1
        return zero_one_arr
    return filtered_list


def connect_(pairs, n=1):
    """connect_
    Connects two adjacent clusters if their distance is <= n

    :param pairs: Clusters of iterateables e.g., [(1,5),(7,10)]
    :param n: distance between two clusters 
    """
    if len(pairs) == 0:
        return []
    start_, end_ = pairs[0]
    new_pairs = []
    for i, (next_item, cur_item) in enumerate(zip(pairs[1:], pairs[0:])):
        end_ = next_item[1]
        if next_item[0] - cur_item[1] <= n:
            pass
        else:
            new_pairs.append((start_, cur_item[1]))
            start_ = next_item[0]
    new_pairs.append((start_, end_))
    return new_pairs


# ----------SED----------


def finetune_net():
    """ 获取用于finetuning的模型"""
    # 预训练模型
    audioset_classes_num = 447
    pretrained_model = Cnn8_Rnn(classes_num=audioset_classes_num)

    # 微调模型，更换最后一层全连接层
    pretrained_model.fc_audioset = nn.Linear(in_features=512, out_features=1, bias=True)

    return pretrained_model


def interpolate(x: np.ndarray, ratio:int):
    """ 对一维向量进行复制插值"""
    upsampled = np.repeat(x, ratio, axis=0)
    return upsampled


def show_audio_with_pred(waveform: np.ndarray, pred: np.ndarray, prob:np.ndarray, outpath) -> None:
    # 读取音频
    pred = pred[:-1]
    pad_ratio = waveform.shape[0] // pred.shape[0]
    pred = interpolate(pred, pad_ratio)

    # 绘图
    plt.figure(figsize=(14, 5))

    # 波形图
    plt.plot(waveform)

    # 预测概率图
    prob = prob[:-1]
    prob = interpolate(prob, pad_ratio)
    plt.plot(prob, color='r')

    # 0.5参考线
    plt.axhline(y=0.5, color='r', linestyle='--')

    # 事件图
    continue_regions = find_contiguous_regions(pred)
    for i in range(len(continue_regions)):
        start = continue_regions[i][0]
        end = continue_regions[i][1]
        # 标定范围
        rect = plt.Rectangle(
            (start, -1), 
            end-start, 
            2, 
            color='b', 
            alpha=0.5,
        )
        plt.gca().add_patch(rect)

    duration = waveform.shape[0] / 32000
    xlabels = [f"{x:.2f}" for x in np.arange(0, duration+1e-5, duration / 5)]
    plt.xticks(ticks=np.arange(0, waveform.shape[0]+1e-5, waveform.shape[0] / 5),
                labels=xlabels,
                fontsize=15)
    plt.title('Barking SED')
    plt.xlabel("Time / second", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def load_audio(audio_path):
    if isinstance(audio_path, PosixPath):
        audio_path = audio_path.__str__()
    waveform, _ = librosa.core.load(audio_path, sr=32000)
    return waveform


def sound_event_detection(audio_path, checkpoint_path, result_out_dir):
    # 参数
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 事件判定阈值
    threshold = [0.75, 0.25]
    single_threshold = 0.5

    # 模型
    model = finetune_net()
    checkpoint = torch.load(checkpoint_path, device)

    # print(checkpoint)
    # print([*checkpoint])

    # model.load_state_dict(checkpoint["model"])
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    for dir in tqdm(os.listdir(audio_path)):
        waveform = load_audio(audio_path + dir)
        waveform = torch.tensor(waveform).unsqueeze(0)

        waveform = waveform.to(device)
        output_dict = model(waveform)
        framewise_output = output_dict['framewise_output']
        prob = framewise_output[0, :, 0].detach().cpu().numpy()

        postprocessing_method = double_threshold
        thresholded_predictions = postprocessing_method(prob, *threshold)

        with open(result_out_dir,'a') as fp:
            fp.write(dir)
            fp.write('\n')
            fp.write(' '.join(str(i) for i in thresholded_predictions))
            fp.write('\n')

        torch.cuda.empty_cache()
    # pdb.set_trace()
    
    # 可视化
    # audio_name = audio_path.split('/')[-1].split('.wav')[0]
    # create_folder(result_out_dir)
    # outpath = os.path.join(result_out_dir, audio_name+'.png')
    # show_audio_with_pred(np.array(torch.squeeze(waveform, dim=0).cpu()), np.array(thresholded_predictions), prob, outpath)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--out_dir', type=str, required=True)
    argparser.add_argument('--audio_path', type=str, required=True)
    opt = argparser.parse_args()
    out_dir = opt.out_dir
    audio_path = opt.audio_path

    checkpoint_path = 'model.pth'
    sound_event_detection(audio_path, checkpoint_path, out_dir)
