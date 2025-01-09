import os
import numpy as np
import eval_metrics as em
import matplotlib.pyplot as plt

def compute_eer_and_tdcf(cm_score_file, path_to_database):
    asv_score_file = os.path.join(path_to_database, 'E:/datas/ASVspoof_2019/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt')

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 1]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(float)

    other_cm_scores = -cm_scores

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == '1']
    spoof_cm = cm_scores[cm_keys == '0']

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == '1'], other_cm_scores[cm_keys == '0'])[0]

    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    if eer_cm < other_eer_cm:
        # Compute t-DCF
        tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)

        # Minimum t-DCF
        min_tDCF_index = np.argmin(tDCF_curve)
        min_tDCF = tDCF_curve[min_tDCF_index]

    else:
        tDCF_curve, CM_thresholds = em.compute_tDCF(other_cm_scores[cm_keys == '1'], other_cm_scores[cm_keys == '0'],
                                                    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)

        # Minimum t-DCF
        min_tDCF_index = np.argmin(tDCF_curve)
        min_tDCF = tDCF_curve[min_tDCF_index]
    eer =min(eer_cm, other_eer_cm)
    out_data = "min_tDCF: %.4f\n" % min_tDCF
    out_data += "eer: %.2f\n" % (100 * eer)
    print(out_data, end="")
    with open(os.path.dirname(cm_score_file) + '/eer.txt', 'w') as file:
        file.write(out_data)

    return eer, min_tDCF

def compute_eer_A1_A20(cm_score_file,path):

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 1]
    cm_sources = cm_data[:, 1]
    unique_vals, counts, indices_dict = unique(cm_sources)
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(float)
    other_cm_scores = -cm_scores
    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == '1']
    spoof_cm = cm_scores[cm_keys == '0']
    eer_cm ,thresholds= em.compute_eer(bona_cm, spoof_cm)
    other_eer_cm,other_thresholds = em.compute_eer(other_cm_scores[cm_keys == '1'], other_cm_scores[cm_keys == '0'])
    eer_all = min(eer_cm, other_eer_cm)
    if eer_all == eer_cm:
        thresholds = thresholds
    else: thresholds = other_thresholds
    EER = {}
    for key, value in indices_dict.items():
        if key == "-":
            bona_cm = np.array([])
            spoof_cm = cm_scores[value]
            far = compute_far(spoof_cm, thresholds)
            EER[key] = 1-far
        else:
            bona_cm = np.array([])
            spoof_cm = cm_scores[value]
            far = compute_far(spoof_cm, thresholds)
            EER[key] = far
    with open(path+ '/A07_A20_EER.txt', 'w') as A07_A20_EER_file:
        for key, value in EER.items():
            A07_A20_EER_file.write('%s:\t%s' % (key, value*100))
            A07_A20_EER_file.write('\t%\n')







    return eer_all,EER


def compute_eer_A1_A20_max(cm_score_file,path):

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 0]
    unique_vals, counts, indices_dict = unique(cm_sources)
    cm_keys = cm_data[:, 1]
    cm_scores = cm_data[:, 2].astype(float)
    other_cm_scores = -cm_scores
    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']
    eer_cm ,thresholds= em.compute_eer(bona_cm, spoof_cm)
    other_eer_cm,other_thresholds = em.compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_keys == 'spoof'])
    eer_all = min(eer_cm, other_eer_cm)
    if eer_all == eer_cm:
        thresholds = thresholds
    else: thresholds = other_thresholds
    EER = {}
    for key, value in indices_dict.items():
        if key == "A20":
            bona_cm = np.array([])
            spoof_cm = cm_scores[value]
            far = compute_far(spoof_cm, thresholds)
            EER[key] = 1-far
        else:
            bona_cm = np.array([])
            spoof_cm = cm_scores[value]
            far = compute_far(spoof_cm, thresholds)
            EER[key] = far
    with open(path+ '/A07_A20_EER_max.txt', 'w') as A07_A20_EER_file:
        for key, value in EER.items():
            A07_A20_EER_file.write('%s:\t%s' % (key, value*100))
            A07_A20_EER_file.write('\t%\n')





    print('\nCM SYSTEM')
    print('\n   EER_all            = {:8.5f} % (Equal error rate for countermeasure)'.format(eer_all * 100))
    print('\nEER:',EER)




    return eer_all,EER

def unique(arr):
    unique_vals, counts = np.unique(arr, return_counts=True)

    # 返回每个种类的对应元素的索引
    indices_dict = {}
    for val in unique_vals:
        indices_dict[val] = np.where(arr == val)[0]

    return unique_vals,counts,indices_dict

def compute_far(nontarget_scores,threshold):
    fa_s = np.sum(nontarget_scores > threshold)
    far = fa_s/nontarget_scores.size


    return  far

