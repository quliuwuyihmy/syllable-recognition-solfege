"""train non-sil and sil monophone HMM"""

import os
import numpy as np
from parameters import *
import pomegranate


def multivariate_gaussian_distribution_diag(data_init, dim_feature):
    """multivariate gaussian diag covar"""
    guassianDiag = []
    for ii in range(dim_feature):
        # print(data_init[ii, 0], data_init[ii, 1])
        guassianDiag.append(pomegranate.distributions.NormalDistribution(data_init[ii, 0], data_init[ii, 1]))
    return guassianDiag


def create_mixture_diag_covar(data_init, n_mix, dim_feature):
    """diag covar mixture"""
    mixtures = [pomegranate.distributions.IndependentComponentsDistribution(multivariate_gaussian_distribution_diag(data_init[i], dim_feature)) for i in range(n_mix)]
    return mixtures


def create_mixture_full_covar(n_mix, dim_feature):
    """full covar mixture"""
    mixtures = [pomegranate.distributions.MultivariateGaussianDistribution(np.random.normal(0.0, 1.0, dim_feature),
                                                                           np.identity(dim_feature)) for i in
                range(n_mix)]
    # mixtures = [pomegranate.distributions.MultivariateGaussianDistribution(np.zeros((dim_feature,)),
    #                                                                        np.identity(dim_feature)/n_mix) for i in
    #             range(n_mix)]
    return mixtures


def create_state(data_init, n_mix, dim_feature, name_phn, name_state, covar_type='full'):
    """full covar GMM"""
    if covar_type == 'full':
        mixtures = create_mixture_full_covar(n_mix=n_mix, dim_feature=dim_feature)
    elif covar_type == 'diag':
        mixtures = create_mixture_diag_covar(data_init=data_init, n_mix=n_mix, dim_feature=dim_feature)
    else:
        raise ValueError("{} is not a valid covar type.".format(covar_type))

    state = pomegranate.State(pomegranate.GeneralMixtureModel(mixtures), name=name_phn + name_state)
    return state


def fit_non_sil_phn(data_init, n_mix, dim_feature, name_phn, covar_type='full'):
    # Create model with 3 states
    # Left-to-right: each state is connected to itself and its direct successor

    state_0 = create_state(data_init=data_init[0],
                           n_mix=n_mix,
                           dim_feature=dim_feature,
                           name_phn=name_phn,
                           name_state='-first',
                           covar_type=covar_type)
    state_1 = create_state(data_init=data_init[1],
                           n_mix=n_mix,
                           dim_feature=dim_feature,
                           name_phn=name_phn,
                           name_state='-mid',
                           covar_type=covar_type)
    state_2 = create_state(data_init=data_init[2],
                           n_mix=n_mix,
                           dim_feature=dim_feature,
                           name_phn=name_phn,
                           name_state='-last',
                           covar_type=covar_type)

    model = pomegranate.HiddenMarkovModel(name_phn)

    model.add_state(state_0)
    model.add_state(state_1)
    model.add_state(state_2)

    model.add_transition(model.start, state_0, 1.0)
    model.add_transition(state_0, state_0, 0.5)
    model.add_transition(state_0, state_1, 0.5)
    model.add_transition(state_0, model.end, 0.000001)
    model.add_transition(state_1, state_1, 0.5)
    model.add_transition(state_1, model.end, 0.000001)
    model.add_transition(state_1, state_2, 0.5)
    model.add_transition(state_2, state_2, 0.5)
    model.add_transition(state_2, model.end, 0.5)

    model.bake()
    return model


def fit_sil(data_init, n_mix, dim_feature, covar_type='full'):
    # Create model with 3 states
    # Left-to-right: each state is connected to itself and its direct successor

    state_0 = create_state(data_init=data_init[0], n_mix=n_mix, dim_feature=dim_feature, name_phn='sil', name_state='-first', covar_type=covar_type)
    state_1 = create_state(data_init=data_init[1], n_mix=n_mix, dim_feature=dim_feature, name_phn='sil', name_state='-mid', covar_type=covar_type)
    state_2 = create_state(data_init=data_init[2], n_mix=n_mix, dim_feature=dim_feature, name_phn='sil', name_state='-last', covar_type=covar_type)

    model = pomegranate.HiddenMarkovModel('sil')

    model.add_state(state_0)
    model.add_state(state_1)
    model.add_state(state_2)

    model.add_transition(model.start, state_0, 1.0)
    model.add_transition(state_0, state_0, 0.333333)
    model.add_transition(state_0, state_1, 0.333333)
    model.add_transition(state_0, model.end, 0.000001)
    model.add_transition(state_1, state_1, 0.5)
    model.add_transition(state_1, state_2, 0.5)
    model.add_transition(state_1, model.end, 0.000001)
    model.add_transition(state_2, state_2, 0.333333)
    model.add_transition(state_0, state_2, 0.333333)
    model.add_transition(state_2, state_0, 0.333333)
    model.add_transition(state_2, model.end, 0.333333)

    model.bake()
    return model


def filter_short_data(training_data):
    training_data[0] = [training_data[0][ii] for ii in range(len(training_data[0])) if training_data[1][ii] > 2]
    training_data[1] = [training_data[1][ii] for ii in range(len(training_data[1])) if training_data[1][ii] > 2]
    return training_data


def scaling_data(training_data, scaler):
    for ii in range(len(training_data)):
        # print(len(training_data[ii]))
        training_data[ii] = scaler.transform(training_data[ii])
    # pickle.dump(training_data, open('./training_data.pkl', 'wb'), protocol=2)
    # data = np.concatenate(training_data)
    # print(np.mean(data, axis=0))
    # print(np.std(data, axis=0))
    return training_data


def data_initialization(full_set, n_states, n_mix, dim_feature):
    means = np.mean(full_set, axis=0)
    stds = np.std(full_set, axis=0)
    # initial values for all gaussian components
    np.random.seed(None)
    dist_init = np.random.random((n_states, n_mix, dim_feature, 2))
    dist_init[..., 0] -= 0.5  # center means to 0.0
    for feat_i in range(dim_feature):
        # random init mean in range [-2std, 2std)
        dist_init[..., feat_i, 0] *= 4 * stds[feat_i]
        dist_init[..., feat_i, 0] += means[feat_i]
        # random init std in range 1std/n_components
        dist_init[..., feat_i, 1] *= stds[feat_i] / n_mix
    return dist_init


def print_info(training_data, l, model):

    print('label')
    print(l)

    print('training data sample size')
    print(len(training_data[0]))

    print('average length (s)')
    print(np.mean(training_data[1]))

    index = []
    for state_name in ['-start', 'first', 'mid', 'last', '-end']:
        for ii_state, state in enumerate(model.states):
            if state_name in state.name:
                index.append(ii_state)
    print('index')
    print(index)

    # print(model.states[2])

    print('transition matrix')
    print(model.dense_transition_matrix()[index, :][:, index])


if __name__ == '__main__':
    import pickle
    from lexicon import list_phn

    path_training_data = '../training_data/pomegranate_mfcc_delta'
    path_pretrained_models = '../pretrained_models/mfcc_delta'

    scaler = pickle.load(open(os.path.join(path_training_data, 'scaler.pkl'), 'rb'))

    # non-sil phone
    for l in ['m', 'ow']:#list_phn:
        print('load data', l)
        training_data = pickle.load(open(os.path.join(path_training_data, l+'.pkl'), 'rb'))
        training_data = filter_short_data(training_data)
        # print(min(training_data[1]))
        training_data[0] = scaling_data(training_data[0], scaler)
        data_init = data_initialization(np.concatenate(training_data[0]),
                                        n_states=n_states_phn,
                                        n_mix=n_mix,
                                        dim_feature=dim_feature)

        print('build model')
        model = fit_non_sil_phn(data_init=data_init,
                                n_mix=n_mix,
                                dim_feature=dim_feature,
                                name_phn=l,
                                covar_type=covariance_type)

        print('fitting')
        model.fit(training_data[0],
                  algorithm='baum-welch',
                  n_jobs=4,
                  stop_threshold=1e-9,
                  max_iterations=100,
                  verbose=True)

        print('dump model')
        pickle.dump(model, open(os.path.join(path_pretrained_models, l+'.pkl'), 'wb'), protocol=2)

        print_info(training_data, l, model)

    # sil
    training_data = pickle.load(open(os.path.join(path_training_data, 'sil.pkl'), 'rb'))
    training_data = filter_short_data(training_data)
    training_data[0] = scaling_data(training_data[0], scaler)
    data_init = data_initialization(np.concatenate(training_data[0]),
                                    n_states=n_states_sil,
                                    n_mix=n_mix,
                                    dim_feature=dim_feature)

    model = fit_sil(data_init=data_init, n_mix=n_mix, dim_feature=dim_feature, covar_type=covariance_type)

    model.fit(training_data[0],
              algorithm='baum-welch',
              n_jobs=4,
              stop_threshold=1e-9,
              max_iterations=100,
              verbose=True)

    pickle.dump(model, open(os.path.join(path_pretrained_models, 'sil.pkl'), 'wb'), protocol=2)

    print_info(training_data, 'sil', model)