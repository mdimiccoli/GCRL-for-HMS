import numpy as np
import sys
from Hungarian import *
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
import BestMap

def correct_predictions_recursive(contingency_table, bins1, bins0):
    max_value = np.max(contingency_table)
    
    if (max_value == 0):
        return 0
    
    num_max = np.sum(contingency_table == max_value)
    count_label = np.zeros((num_max))

    k = 0
    for i in range(bins1):
        for j in range(bins0):
            if (contingency_table[i,j] == max_value):
                copy_ct = np.copy(contingency_table)
                count_label[k] += max_value
                # We remove the information associated to max_value
                # from the contingency_table table
                copy_ct[i,:]=0
                copy_ct[:,j]=0
                # Here we perform a recursive call
                count_label[k] += correct_predictions_recursive(copy_ct, bins1, bins0)
                k += 1

    correct_classification_labels = np.max(count_label)

    return correct_classification_labels


def classification_accuracy(idx1,idx0):
    """Classification accuracy   

       classification_accuracy(idx1,idx0)
       idx1 -- results of the experiment
       idx0 -- ground truth

       We assume that labels are integer numbers.

       Examples:
          idx1 = np.array([0,0,0,0,1,1])
          idx0 = np.array([0,0,0,0,1,2])
          classification_accuracy(idx1,idx0)
          result: 0.833333

          Observe the label '1' of idx0 has three correspondences for
          label '0' of idx1 and three for label '1' of idx1. But the
          label '2' of idx0 has four correspondenes with label '1' of idx1.
          The algorithm assigns label '2' of idx0 to label '1' of idx1 and
          label '1' of idx0 to label '0' of idx1.

          idx1 = np.array([0,0,0,0,1,1,1,1,1,1,1])
          idx0 = np.array([0,1,1,1,1,1,1,2,2,2,2])
          classification_accuracy(idx1,idx0)
          result: (4+3)/11 = 0.636363
    """
    # Cast to integer

    idx1 = idx1.astype(np.int)
    idx0 = idx0.astype(np.int)

    # We set the labels to start at 0. Commented since relabelling is performed.

    #idx1 = idx1 - np.min(idx1)
    #idx0 = idx0 - np.min(idx0)

    # Unique labels and relabelling starting from zero

    uids1,idx1_relabeled = np.unique(idx1,return_inverse=1)
    uids0,idx0_relabeled = np.unique(idx0,return_inverse=1)

    # The number of bins of each set

    bins1 = uids1.size
    bins0 = uids0.size

    # Contingengy table: great code found at stackoverflow!

    contingency_table = np.bincount(bins0 * idx1_relabeled + idx0_relabeled, minlength=bins1*bins0).reshape((bins1, bins0))

    # Compute accuracy. Observe that a recursive algorithm is used
    # to take into account that a label can be assigned to different
    # ground truth label.
    correct_classification_labels = correct_predictions_recursive(contingency_table, bins1, bins0)

    accuracy = correct_classification_labels / idx1.size

    return accuracy

def compacc_sk_h(idx1,idx0):
    
    ari=adjusted_rand_score(idx0,idx1)
    nmi=normalized_mutual_info_score(idx0,idx1)
    acc =classification_accuracy(idx0,idx1)
    
    idx1 = BestMap.BestMap(idx0,idx1)
    Hungarianrate = float(np.sum(idx0 == idx1)) / idx0.size
    return ari, nmi, acc, Hungarianrate
