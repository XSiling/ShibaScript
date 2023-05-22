import os
import sys
import wave
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
from moviepy.editor import *
from tqdm import tqdm
import pdb
from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def sound_event_detection(args):
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size

    # Paths
    fig_path = os.path.join('results', '{}.png'.format(get_filename(audio_path)))
    create_folder(os.path.dirname(fig_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_sed = subparsers.add_parser('sound_event_detection')
    parser_sed.add_argument('--sample_rate', type=int, default=32000)
    parser_sed.add_argument('--window_size', type=int, default=1024)
    parser_sed.add_argument('--hop_size', type=int, default=320)
    parser_sed.add_argument('--mel_bins', type=int, default=64)
    parser_sed.add_argument('--fmin', type=int, default=50)
    parser_sed.add_argument('--fmax', type=int, default=14000) 
    parser_sed.add_argument('--model_type', type=str, required=True)
    parser_sed.add_argument('--checkpoint_path', type=str, required=True)
    parser_sed.add_argument('--file_path',type=str, default='/home/jieyi/Ani/code/singleDog_IPA/data/1/process1/auto_wav_single/')
    parser_sed.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()

    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size

    # Paths
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)

    count = 0
    move_count = 0
    # process audio!
    for dir in tqdm(os.listdir(args.file_path)):
        count += 1
        try:
            (waveform, _) = librosa.core.load(args.file_path + dir, sr=sample_rate, mono=True)
            waveform = waveform[None, :]
            waveform = move_data_to_device(waveform, device)

            with torch.no_grad():
                model.eval()
                batch_output_dict = model(waveform, device)
            framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
            sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]

            top_k = 10  # Show top results

            if 0 in sorted_indexes[0:5] or 137 in sorted_indexes[0:5]:
                move_count += 1
                print("move", dir, move_count, "/", count)
                os.remove(args.file_path + dir)
                

        
            torch.cuda.empty_cache()
            
        except:
            print("sth wrong")




            