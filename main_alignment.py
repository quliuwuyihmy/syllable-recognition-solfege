import matplotlib
matplotlib.use('Tkagg')

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from parameters import *
from utils import figure_plot
from data.feature_extraction import mfcc_extraction
from model.concatenative_hmm import concatenative_hmm_alignment


# path and transcription
dir_path = os.path.dirname(os.path.realpath(__file__))
path_pretrained_model = os.path.join(dir_path, 'pretrained_models/mfcc_delta')
path_training_data = os.path.join(dir_path, 'training_data/pomegranate_mfcc_delta')
test_wav_file = './test/reference_exercise_01_norm.wav'

test_wav_trans = ['sil', 'do', 'si', 'do', 'mi', 're', 'mi', 'do',
                  're', 'la', 're', 'do', 'si', 'la', 'si', 're', 'do', 'sil']

# test_wav_trans = ['sil', 'sol', 'sol', 'do', 'sol']

scaler = pickle.load(open(os.path.join(path_training_data, 'scaler.pkl'), 'rb'))

# create alignment HMM
hmm_conc = concatenative_hmm_alignment(trans=test_wav_trans)

mfcc_test, sr_test = mfcc_extraction(test_wav_file, framesize_t=framesize_t, hopsize_t=hopsize_t)

# obtain the best alignment path
path = hmm_conc.predict(scaler.transform(mfcc_test.T), algorithm='viterbi')

# print and visualize the path
dict_state = dict()
for id, state in enumerate(hmm_conc.states):
    dict_state[id] = state.name

path = [dict_state[id] for id in path if 'hmm_conc' not in dict_state[id]]
onset_label = [[0, path[0]]]
for ii in range(1, len(path)):
    if path[ii].split('-')[0] != path[ii-1].split('-')[0] and path[ii][0] != path[ii-1][0]:
        onset_label.append((ii, path[ii]))

print(len(mfcc_test.T))
print(len(path))
print(onset_label)

figure_plot(plt=plt, mfcc=mfcc_test.T, onset_label=onset_label)
plt.show()