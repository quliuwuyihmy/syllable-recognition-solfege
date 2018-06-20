import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from parameters import *
from utils import figure_plot
from data.feature_extraction import mfcc_extraction

# path
dir_path = os.path.dirname(os.path.realpath(__file__))
path_pretrained_model = os.path.join(dir_path, 'pretrained_models/mfcc_delta')
path_training_data = os.path.join(dir_path, 'training_data/pomegranate_mfcc_delta')
test_wav_file = './test/reference_exercise_01_norm.wav'
scaler = pickle.load(open(os.path.join(path_training_data, 'scaler.pkl'), 'rb'))

# load the recognition HMM
hmm_conc = pickle.load(open(os.path.join(path_pretrained_model, 'hmm_conc.pkl'), 'rb'))

mfcc_test, sr_test = mfcc_extraction(test_wav_file, framesize_t=framesize_t, hopsize_t=hopsize_t)

# obtain the decoding path
path = hmm_conc.predict(scaler.transform(mfcc_test.T), algorithm='viterbi')

dict_state = dict()
for id, state in enumerate(hmm_conc.states):
    dict_state[id] = state.name

path = [dict_state[id] for id in path if dict_state[id] != 'hmm_conc-start']
onset_label = [[0, path[0]]]
for ii in range(1, len(path)):
    if path[ii].split('-')[0] != path[ii-1].split('-')[0] and path[ii][0] != path[ii-1][0]:
        onset_label.append((ii, path[ii]))

print(len(mfcc_test.T))
print(len(path))
print(onset_label)

figure_plot(plt=plt, mfcc=mfcc_test.T, onset_label=onset_label)
plt.show()