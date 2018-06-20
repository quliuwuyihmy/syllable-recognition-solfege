import matplotlib.pyplot as plt
import os
import pickle
import pomegranate
from lexicon import *


dir_path = os.path.dirname(os.path.realpath(__file__))
path_pretrained_model = os.path.join(dir_path, '../pretrained_models/mfcc_delta')


def change_state_name(model, syl, phn):
    for name in [phn+'-start', phn+'-first', phn+'-mid', phn+'-last', phn+'-end']:
        for state in model.states:
            if state.name == name:
                state.name = syl+'-'+state.name


def concatenative_hmm_recogntion(path_pretrained_model):

    hmm_do_d = pickle.load(open(os.path.join(path_pretrained_model, 'd.pkl'), 'rb'))
    change_state_name(hmm_do_d, 'do', 'd')
    hmm_do_ow = pickle.load(open(os.path.join(path_pretrained_model, 'ow.pkl'), 'rb'))
    change_state_name(hmm_do_ow, 'do', 'ow')

    hmm_re_r = pickle.load(open(os.path.join(path_pretrained_model, 'r.pkl'), 'rb'))
    change_state_name(hmm_re_r, 're', 'r')
    hmm_re_ey = pickle.load(open(os.path.join(path_pretrained_model, 'ey.pkl'), 'rb'))
    change_state_name(hmm_re_ey, 're', 'ey')


    hmm_mi_m = pickle.load(open(os.path.join(path_pretrained_model, 'm.pkl'), 'rb'))
    change_state_name(hmm_mi_m, 'mi', 'm')
    hmm_mi_iy = pickle.load(open(os.path.join(path_pretrained_model, 'iy.pkl'), 'rb'))
    change_state_name(hmm_mi_iy, 'mi', 'iy')

    hmm_fa_f = pickle.load(open(os.path.join(path_pretrained_model, 'f.pkl'), 'rb'))
    change_state_name(hmm_fa_f, 'fa', 'f')
    hmm_fa_aa = pickle.load(open(os.path.join(path_pretrained_model, 'aa.pkl'), 'rb'))
    change_state_name(hmm_fa_aa, 'fa', 'aa')

    hmm_sol0_s = pickle.load(open(os.path.join(path_pretrained_model, 's.pkl'), 'rb'))
    change_state_name(hmm_sol0_s, 'sol0', 's')
    hmm_sol0_ow = pickle.load(open(os.path.join(path_pretrained_model, 'ow.pkl'), 'rb'))
    change_state_name(hmm_sol0_ow, 'sol0', 'ow')
    hmm_sol0_l = pickle.load(open(os.path.join(path_pretrained_model, 'l.pkl'), 'rb'))
    change_state_name(hmm_sol0_l, 'sol0', 'l')

    hmm_sol1_s = pickle.load(open(os.path.join(path_pretrained_model, 's.pkl'), 'rb'))
    change_state_name(hmm_sol1_s, 'sol1', 's')
    hmm_sol1_ao = pickle.load(open(os.path.join(path_pretrained_model, 'ao.pkl'), 'rb'))
    change_state_name(hmm_sol1_ao, 'sol1', 'ao')
    hmm_sol1_l = pickle.load(open(os.path.join(path_pretrained_model, 'l.pkl'), 'rb'))
    change_state_name(hmm_sol1_l, 'sol1', 'l')

    hmm_la_l = pickle.load(open(os.path.join(path_pretrained_model, 'l.pkl'), 'rb'))
    change_state_name(hmm_la_l, 'la', 'l')
    hmm_la_aa = pickle.load(open(os.path.join(path_pretrained_model, 'aa.pkl'), 'rb'))
    change_state_name(hmm_la_aa, 'la', 'aa')

    hmm_si_s = pickle.load(open(os.path.join(path_pretrained_model, 's.pkl'), 'rb'))
    change_state_name(hmm_si_s, 'si', 's')
    hmm_si_iy = pickle.load(open(os.path.join(path_pretrained_model, 'iy.pkl'), 'rb'))
    change_state_name(hmm_si_iy, 'si', 'iy')

    hmm_sil = pickle.load(open(os.path.join(path_pretrained_model, 'sil.pkl'), 'rb'))

    hmm_conc = pomegranate.HiddenMarkovModel("hmm_conc")

    hmm_conc.add_model(hmm_do_d)
    hmm_conc.add_model(hmm_do_ow)

    hmm_conc.add_model(hmm_re_r)
    hmm_conc.add_model(hmm_re_ey)

    hmm_conc.add_model(hmm_mi_m)
    hmm_conc.add_model(hmm_mi_iy)

    hmm_conc.add_model(hmm_fa_f)
    hmm_conc.add_model(hmm_fa_aa)

    hmm_conc.add_model(hmm_sol0_s)
    hmm_conc.add_model(hmm_sol0_ow)
    hmm_conc.add_model(hmm_sol0_l)


    hmm_conc.add_model(hmm_sol1_s)
    hmm_conc.add_model(hmm_sol1_ao)
    hmm_conc.add_model(hmm_sol1_l)

    hmm_conc.add_model(hmm_la_l)
    hmm_conc.add_model(hmm_la_aa)

    hmm_conc.add_model(hmm_si_s)
    hmm_conc.add_model(hmm_si_iy)

    hmm_conc.add_model(hmm_sil)

    # phrase start to phn start transitions
    hmm_conc.add_transition(hmm_conc.start, hmm_do_d.start, 0.111111)
    hmm_conc.add_transition(hmm_conc.start, hmm_re_r.start, 0.111111)
    hmm_conc.add_transition(hmm_conc.start, hmm_mi_m.start, 0.111111)
    hmm_conc.add_transition(hmm_conc.start, hmm_fa_f.start, 0.111111)
    hmm_conc.add_transition(hmm_conc.start, hmm_sol0_s.start, 0.111111)
    hmm_conc.add_transition(hmm_conc.start, hmm_sol1_s.start, 0.111111)
    hmm_conc.add_transition(hmm_conc.start, hmm_la_l.start, 0.111111)
    hmm_conc.add_transition(hmm_conc.start, hmm_si_s.start, 0.111111)

    hmm_conc.add_transition(hmm_conc.start, hmm_sil.start, 0.111111)

    # # phn end to phrase end transitions
    # hmm_conc.add_transition(hmm_ow.end, hmm_conc.end, 0.2)
    # hmm_conc.add_transition(hmm_ey.end, hmm_conc.end, 0.2)
    # hmm_conc.add_transition(hmm_iy.end, hmm_conc.end, 0.2)
    # hmm_conc.add_transition(hmm_aa.end, hmm_conc.end, 0.2)
    # hmm_conc.add_transition(hmm_ao.end, hmm_conc.end, 0.2)

    # consonant to vowel transitions
    hmm_conc.add_transition(hmm_do_d.end, hmm_do_ow.start, 1.0)
    hmm_conc.add_transition(hmm_re_r.end, hmm_re_ey.start, 1.0)
    hmm_conc.add_transition(hmm_mi_m.end, hmm_mi_iy.start, 1.0)
    hmm_conc.add_transition(hmm_fa_f.end, hmm_fa_aa.start, 1.0)
    hmm_conc.add_transition(hmm_sol0_s.end, hmm_sol0_ow.start, 1.0)
    hmm_conc.add_transition(hmm_sol0_ow.end, hmm_sol0_l.start, 0.5)
    hmm_conc.add_transition(hmm_sol0_ow.end, hmm_sol0_l.end, 0.5)
    hmm_conc.add_transition(hmm_sol1_s.end, hmm_sol1_ao.start, 1.0)
    hmm_conc.add_transition(hmm_sol1_ao.end, hmm_sol1_l.start, 0.5)
    hmm_conc.add_transition(hmm_sol1_ao.end, hmm_sol1_l.end, 0.5)
    hmm_conc.add_transition(hmm_la_l.end, hmm_la_aa.start, 1.0)
    hmm_conc.add_transition(hmm_si_s.end, hmm_si_iy.start, 1.0)

    # syllable end to phrase start
    hmm_conc.add_transition(hmm_do_ow.end, hmm_conc.start, 1.0)
    hmm_conc.add_transition(hmm_re_ey.end, hmm_conc.start, 1.0)
    hmm_conc.add_transition(hmm_mi_iy.end, hmm_conc.start, 1.0)
    hmm_conc.add_transition(hmm_fa_aa.end, hmm_conc.start, 1.0)
    hmm_conc.add_transition(hmm_sol0_l.end, hmm_conc.start, 1.0)
    hmm_conc.add_transition(hmm_sol1_l.end, hmm_conc.start, 1.0)
    hmm_conc.add_transition(hmm_la_aa.end, hmm_conc.start, 1.0)
    hmm_conc.add_transition(hmm_si_iy.end, hmm_conc.start, 1.0)
    hmm_conc.add_transition(hmm_sil.end, hmm_conc.start, 1.0)

    hmm_conc.bake(merge=False)

    pickle.dump(hmm_conc, open(os.path.join(path_pretrained_model, 'hmm_conc.pkl'), 'wb'), protocol=2)

    hmm_conc.plot()
    plt.savefig('topo.png', dpi=3000)


def concatenative_hmm_alignment(trans):
    """concatenate hmm from transcription"""
    # initialize the syllable counter dictionary
    dict_syl_counter = dict()
    for l in syl_2_phn.keys():
        dict_syl_counter[l] = 0

    hmm_conc = pomegranate.HiddenMarkovModel("hmm_conc")
    hmm_precedent = []
    p_first = True
    for syl in trans:
        phns_syl = syl_2_phn[syl]
        if len(phns_syl) == 1:
            for p in phns_syl[0]:
                hmm_p = pickle.load(open(os.path.join(path_pretrained_model, p + '.pkl'), 'rb'))
                change_state_name(hmm_p, syl+'-'+str(dict_syl_counter[syl]), p)
                hmm_conc.add_model(hmm_p)
                if p_first:
                    hmm_conc.add_transition(hmm_conc.start, hmm_p.start, 1.0)
                    p_first = False
                else:
                    for ii_hmm_precedent in range(len(hmm_precedent)):
                        hmm_conc.add_transition(hmm_precedent[ii_hmm_precedent].end, hmm_p.start, 1.0)
                hmm_precedent = [hmm_p]
        else:
            hmm_branch_precedent = hmm_precedent
            hmm_in_branch_precedent = None
            hmm_precedent = []
            for ii_phns, phns in enumerate(phns_syl):
                for ii_p, p in enumerate(phns):
                    hmm_p = pickle.load(open(os.path.join(path_pretrained_model, p + '.pkl'), 'rb'))
                    change_state_name(hmm_p, syl + '-' + str(ii_phns) + '-' + str(dict_syl_counter[syl]), p)
                    hmm_conc.add_model(hmm_p)
                    if p_first:
                        hmm_conc.add_transition(hmm_conc.start, hmm_p.start, 1.0)
                        if ii_phns == len(phns) - 1:
                            p_first = False
                    elif ii_p == 0:
                        for ii_hmm_precedent in range(len(hmm_branch_precedent)):
                            hmm_conc.add_transition(hmm_branch_precedent[ii_hmm_precedent].end, hmm_p.start, 1.0)
                    else:
                        hmm_conc.add_transition(hmm_in_branch_precedent.end, hmm_p.start, 1.0)
                    hmm_in_branch_precedent = hmm_p
                    if ii_p == len(phns) - 1:
                        hmm_precedent.append(hmm_p)
        dict_syl_counter[syl] += 1

    for ii_hmm_precedent in range(len(hmm_precedent)):
        hmm_conc.add_transition(hmm_precedent[ii_hmm_precedent].end, hmm_conc.end, 1.0)

    hmm_conc.bake()
    # hmm_conc.plot()
    # plt.savefig('topo.png', dpi=3000)
    return hmm_conc


if __name__ == '__main__':
    concatenative_hmm_recogntion(path_pretrained_model)