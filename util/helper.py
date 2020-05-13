import numpy as np, os, sys
sys.path.append(os.path.abspath("./datasets"))
from physionet2020 import PhysioNet2020Dataset
from torch.utils.data import DataLoader

def translate_x(x_data):
    result = []
    for i in range(x_data.shape[1]):
        result.append(x_data[:, i, :])
    return result

def translate_y(y_data):
    result = []
    for i in range(y_data.shape[0]):
        result.append(np.tile(y_data[i, :], (25,1)))
    return result

def get_data_from_physionet2020Dataset(records, singal_type='singal', awni_ecg=False):
    ds = PhysioNet2020Dataset(
#         "Training_WFDB", max_seq_len=6400, records=records, ensure_equal_len=True, proc=0
        "Training_WFDB", max_seq_len=4096, records=records, ensure_equal_len=True, proc=0
    )

    dl = DataLoader(
        ds,
        batch_size=len(ds),
        num_workers=0,
        collate_fn=PhysioNet2020Dataset.collate_fn,
    )
    batch = next(iter(dl))
    dev_y = batch['target'].numpy()
    print (dev_y.shape)
    if (awni_ecg):
        dev_y = np.array(translate_y(dev_y), np.int32)

    if singal_type == 'singal':
        dev_x = batch['signal'].numpy()
    else:
        dev_x = batch['d_signal'].numpy()
    if (awni_ecg):
        dev_x = np.array(translate_x(dev_x), np.int32)
        
    dev_x = np.array(translate_x(dev_x))

    # dev_x = np.repeat(dev_x[:, :, np.newaxis], 1, axis=2)
    print (dev_x.shape)
    print (dev_y.shape)
    return dev_x, dev_y

def epoch_evaluation(probs, dev_y):
    
    p = probs.sum(axis=1)
    p = p/25
    labels = dev_y[:, 1, :]

    threshold = 0.5

    probs_class = p
    probs_class[probs_class > threshold] = 0.99
    probs_class[probs_class <= threshold] = 0.01
    probs_test = probs_class

    sys.path.append(os.path.abspath("./evaluation-2020/"))
    import evaluate_12ECG_score
    auroc,auprc = evaluate_12ECG_score.compute_auc(labels, probs_test, 9)

    probs_class = p
    probs_class[probs_class > threshold] = 1
    probs_class[probs_class <= threshold] = 0
    probs_test = probs_class
    accuracy,f_measure,f_beta,g_beta = evaluate_12ECG_score.compute_beta_score(labels, probs_test, 2, 9)
#     print ([auroc,auprc,accuracy,f_measure,f_beta,g_beta])
    
    return {
        'auroc': auroc,
        'auprc': auprc,
        'accuracy': accuracy,
        'f_measure': f_measure,
        'f_beta': f_beta,
        'g_beta': g_beta
    }