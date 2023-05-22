from scipy.io import wavfile
from tqdm import tqdm
import json
import pdb
import os 
from moviepy.editor import *
from pydub import AudioSegment
import argparse


def process_audio(s_data, save_path, audio_path, process_audio = True):
    ref = []
    start_second = []
    end_second = []
    tmp = s_data[:-1].split(' ')
    audio_name = tmp[-1]
    current_second = int(float(tmp[0]))
    start_second.append(current_second)
    last_second = start_second
    for tmpp in tmp[:-1]:
        if int(float(tmpp))>(current_second + 1):
            end_second.append(last_second + 1)
            current_second = int(float(tmpp))
            start_second.append(int(float(tmpp)))
        else:
            current_second = int(float(tmpp))
        last_second = int(float(tmpp))
    
    if len(end_second)<len(start_second):
        end_second.append(int(float(tmp[-2]))+3)

    assert len(end_second) == len(start_second)


    #pdb.set_trace()
    audio = AudioFileClip(audio_path + audio_name)
    audio.write_audiofile(save_path + audio_name[:-4] + '.wav')

    #

    like = wavfile.read(save_path + audio_name[:-4] + '.wav')
    for i in range(len(start_second)):
        wavfile.write(save_path + audio_name[:-4] + '_' + str(i) + '.wav', like[0], like[1][start_second[i]*like[0]:end_second[i]*like[0]])
        tmpref = {}
        tmpref['name'] = audio_name + "_" + str(i)
        tmpref['start_time'] = start_second[i]
        tmpref['end_time'] = end_second[i]
        ref.append(tmpref)

    os.remove(save_path + audio_name[:-4] + '.wav')
    return ref


def main(log_path, save_path, audio_path):
    with open(log_path,'r') as fp:
        data = fp.readlines()
    
    reference = []

    for jj in tqdm(range(len(data))):
        reference = reference + process_audio(data[jj], save_path, audio_path)

# python process_pannsresult.py --log_path /home/jieyi/Ani/data/ShibaLang/ringoro/log.txt --save_path /home/jieyi/Ani/data/ShibaLang/ringoro/sentences/
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--log_path', required=True, help='the path of the log file')
    argparser.add_argument('--save_path', required=True, help='the path of the saved wavs')
    argparser.add_argument('--audio_path', required=True, help='the source path of audios')
    
    opt = argparser.parse_args()
    log_path = opt.log_path
    save_path = opt.save_path
    audio_path = opt.audio_path

    main(log_path, save_path, audio_path)
