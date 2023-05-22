import pandas as pd
from tqdm import tqdm
import sys
import pdb
import os
import argparse

sys.path.append('/home/jieyi/shennong/shennong-master') 

from shennong.audio import Audio
from shennong.processor.filterbank import FilterbankProcessor

data_path = '../shiba_data/audio/shiba_data/'
save_path = '../shiba_data/data/'
file_path = '../shiba_data/shiba_4.csv'

def generate_filter_bank():
    data = pd.read_csv(file_path)
    for i in tqdm(range(data.shape[0])):
        sample_id = str(data.iloc[i,0])
        video_id = data.iloc[i,6]
        
        if os.path.exists(save_path + sample_id + '/filterbank'):
            pass
        else:
            os.mkdir(save_path + sample_id + '/filterbank')

        if os.path.exists(save_path + sample_id + '/audio/result.wav'):
            #pdb.set_trace()
            audio = Audio.load(save_path + sample_id + '/audio/result.wav')
            processor = FilterbankProcessor(sample_rate=audio.sample_rate)
            processor.use_energy = True
            fbank = processor.process(Audio.channel(audio,0))
            data_m = pd.DataFrame(fbank.data)
            data_m.to_csv(save_path + sample_id + '/filterbank/filterbank.csv')    

def generate_filter_bank1():
    data_path = '/home/jieyi/Ani/data/EJShibaVoice/audio_clip_words/'
    save_path = '/home/jieyi/Ani/data/EJShibaVoice/audio_clip_words_features/filterbank/'

    for dir in tqdm(os.listdir(data_path)):
        clip_name = dir[:-4]
        if os.path.exists(save_path + clip_name + '.csv'):
            continue
        audio = Audio.load(data_path + dir)
        processor = FilterbankProcessor(sample_rate=audio.sample_rate)

        processor.use_energy = True

        fbank = processor.process(Audio.channel(audio,0))

        data_m = pd.DataFrame(fbank.data)
        data_m.to_csv(save_path + clip_name + '.csv')

def generate_filter_bank3(data_path, ref_path, save_path):
    for dir in tqdm(os.listdir(data_path)):
        if dir.endswith('png'):
            continue
        try:
            if len(os.listdir(data_path + dir)) == 1:
                audio = Audio.load(ref_path + dir + '.wav')
                clip_name = dir
                processor = FilterbankProcessor(sample_rate=audio.sample_rate)
                processor.use_energy = True

                fbank = processor.process(Audio.channel(audio,0))
                data_m = pd.DataFrame(fbank.data)
                data_m.to_csv(save_path + clip_name + '.csv')

            else:
                for subdir in os.listdir(data_path + dir):
                    clip_name = dir + '_' + subdir
                    if os.path.exists(save_path + clip_name + '.csv'):
                        continue
                    audio = Audio.load(data_path + dir + '/' + subdir)

                    processor = FilterbankProcessor(sample_rate=audio.sample_rate)
                    processor.use_energy = True

                    fbank = processor.process(Audio.channel(audio,0))

                    data_m = pd.DataFrame(fbank.data)
                    data_m.to_csv(save_path + clip_name + '.csv')
        except:
            print(dir) 


if __name__ == '__main__':
    generate_filter_bank3()
