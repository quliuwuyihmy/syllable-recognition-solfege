import os
import numpy as np
from parameters import *
from hmmlearn import hmm

import warnings


def fit_non_sil_phn(training_data, n_states, n_mix, verbose):
    model = hmm.GMMHMM(n_components=n_states,
                       n_mix=n_mix,
                       covariance_type=covariance_type,
                       n_iter=10,
                       params='tmc',
                       init_params='cm',
                       verbose=verbose)
    model.startprob_ = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    # add a end state, however, this state emit something
    model.transmat_ = np.array([[0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.5, 0.5, 0.0, 0.0],
                                [0.0, 0.0, 0.5, 0.5, 0.0],
                                [0.0, 0.0, 0.0, 0.5, 0.5],
                                [0.0, 0.0, 0.0, 0.0, 1.0]])
    model.fit(training_data[0], training_data[1])

    print(model.startprob_)
    print(model.transmat_)
    print(model.gmms_[-1].means_)

    return model, model.monitor_.converged


def fit_sil(training_data, n_states, n_mix, verbose):
    model = hmm.GMMHMM(n_components=n_states,
                       n_mix=n_mix,
                       covariance_type=covariance_type,
                       n_iter=100,
                       params='tmc',
                       init_params='cm',
                       verbose=verbose)
    model.startprob_ = np.array([1.0, 0.0, 0.0])
    model.transmat_ = np.array([[0.333333, 0.333333, 0.333333],
                                [0.0, 0.5, 0.5],
                                [0.5, 0.0, 0.5]])
    model.fit(training_data[0], training_data[1])

    return model, model.monitor_.converged


if __name__ == '__main__':
    import pickle
    from lexicon import list_phn

    # non-sil phone
    for l in list_phn:
        training_data = pickle.load(open(os.path.join('../training_data_hmmlearn', l+'.pkl'), 'rb'))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            model, converged = fit_non_sil_phn(training_data, n_states=n_states_phn+2, n_mix=n_mix, verbose=True)

            pickle.dump(model, open(os.path.join('../pretrained_monophone_models', l+'.pkl'), 'wb'), protocol=2)

    # sil phone
    training_data = pickle.load(open(os.path.join('../training_data_hmmlearn',  'sil.pkl'), 'rb'))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model, converged = fit_sil(training_data, n_states=n_states_phn, n_mix=n_mix)

        pickle.dump(model, open(os.path.join('../pretrained_monophone_models', 'sil.pkl'), 'wb'), protocol=2)