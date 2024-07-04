from sklearn.metrics import confusion_matrix
import numpy as np


def calculate_se_sp(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    return sensitivity, specificity


def specificity_fixed_sensitivity(y_true, y_pred, min_se=0.95):
    initial_step = (y_pred.max() - y_pred.min()) / 10
    min_val = y_pred.min()

    T = min_val + initial_step

    y_pred_class = np.zeros(len(y_pred))
    y_pred_class[y_pred > T] = 1

    se, _ = calculate_se_sp(y_true, y_pred_class)

    while se > min_se:
        T += initial_step

        y_pred_class = np.zeros(len(y_pred))
        y_pred_class[y_pred > T] = 1

        se, _ = calculate_se_sp(y_true, y_pred_class)

    while se <= min_se:
        T -= 0.01
        y_pred_class[y_pred > T] = 1

        se, sp = calculate_se_sp(y_true, y_pred_class)

    return {
        "se": se,
        "sp": sp,
        "g_mean": np.sqrt(se * sp),
        "T": T,
    }


def calculate_se_sp_gmeans_with_threshold(y_true, y_pred, T=0):
    y_pred_class = np.zeros(len(y_pred))
    y_pred_class[y_pred > T] = 1
    se, sp = calculate_se_sp(y_true, y_pred_class)

    return {
        "se": se,
        "sp": sp,
        "g_mean": np.sqrt(se * sp),
    }


def calculate_gmean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    return np.sqrt(specificity * sensitivity)
