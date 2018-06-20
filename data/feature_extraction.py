import librosa
import numpy as np
from sklearn import preprocessing
from data.parser_nus import onset_offset_label_yield


def mfcc_extraction(filename, framesize_t, hopsize_t):
    """extract 13 dimension mfcc feature"""
    y, sr = librosa.core.load(filename, sr=None)
    framesize = int(np.power(2, np.ceil(np.log(framesize_t * sr)/np.log(2))))
    hopsize = int(hopsize_t * sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=framesize, hop_length=hopsize)
    mfcc = mfcc[1:, :]  # remove energy channel
    mfcc_delta = librosa.feature.delta(mfcc, width=5, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, width=5, order=2)
    mfcc = np.concatenate((mfcc, mfcc_delta, mfcc_delta2))
    return mfcc, sr


if __name__ == '__main__':
    import os
    import pickle
    from parameters import *
    from lexicon import list_phn, list_sil
    from file_path import filepath_nus

    dict_phn = dict()
    list_phn_all = []  # all phn feature
    for read_sing in ['read', 'sing']:
        for fn, sp, onset, offset, label in onset_offset_label_yield(filepath_nus, read_sing):
            print('collecting feature for', read_sing, sp, fn)
            mfcc, sr = mfcc_extraction(os.path.join(filepath_nus, sp, read_sing, fn.replace('txt', 'wav')),
                                       framesize_t=framesize_t,
                                       hopsize_t=hopsize_t)
            for ii_l, l in enumerate(label):
                if l in list_phn + list_sil:
                    phn_start = int(round(onset[ii_l] / hopsize_t))
                    phn_end = int(round(offset[ii_l] / hopsize_t))
                    mfcc_phn = mfcc[:, phn_start:phn_end]

                    if mfcc_phn.shape[1]:
                        # add mfcc phn to dictionary
                        list_phn_l = dict_phn.get(l, [])
                        list_phn_l.append(mfcc_phn.T)
                        dict_phn[l] = list_phn_l

                        list_phn_all.append(mfcc_phn.T)

    path_training_data = '../training_data/pomegranate_mfcc_delta'

    # dump feature and lengths
    for l in dict_phn.keys():
        list_phn_l = dict_phn[l]
        # feature = np.concatenate(list_phn_l)
        length = np.array([len(phn) for phn in list_phn_l], dtype=np.int32)
        # print(feature.shape, length.shape)
        pickle.dump([list_phn_l, length], open(os.path.join(path_training_data, l+'.pkl'), 'wb'), protocol=2)

    # dump scaler
    scaler = preprocessing.StandardScaler()
    list_phn_all = np.concatenate(list_phn_all)
    scaler.fit(list_phn_all)
    pickle.dump(scaler, open(os.path.join(path_training_data, 'scaler.pkl'), 'wb'), protocol=2)
