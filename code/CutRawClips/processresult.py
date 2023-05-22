import os
import librosa
from pydub import AudioSegment
import argparse
import pdb
from tqdm import tqdm 

# audio_path = '/home/jieyi/Ani/code/singleDog_IPA/single_dog_clips/1/'
# save_path = '/home/jieyi/Ani/code/singleDog_IPA/data/1/process1/auto_wav_single/'
def main(audio_path, save_path, result_file):
    mark_data = {}
    with open(result_file,'r') as fp:
        ref = fp.readlines()
        for i in range(len(ref)):
            if i%2 == 0:
                mark_data[ref[i][:-1]] = ref[i+1][:-1]
            else:
                pass

    for dir in tqdm(os.listdir(audio_path)):
        sound = AudioSegment.from_wav(audio_path + dir)

        if '0' not in mark_data[dir].split(' '):
            sound.export(save_path + dir[:-4] + '_raw.wav', format='wav')
            continue

        if len(sound) == (len(mark_data[dir].split(' '))-1)*10:
            # process!
            marks = mark_data[dir].split(' ')
            find = False
            idx = 0
            for i in range(len(marks)):
                mark = marks[i]
                if mark == '1' and find == False:
                    find = True
                    start = i
                if mark == '0' and find == True:
                    # pdb.set_trace()
                    find = False
                    end = i
                    small_sound = sound[start * 10:end * 10]
                    small_sound.export(save_path + dir[:-4] + "_" + str(idx) + '.wav',format='wav')
                    idx += 1
                    

        else:
            print(dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--result_file',type=str, required=True)
 
    args = parser.parse_args()
    # print(args)
    audio_path = args.audio_path
    save_path = args.save_path
    result_file = args.result_file

    main(audio_path, save_path, result_file)
    # print('show {}  {}'.format(epochs, batch))
