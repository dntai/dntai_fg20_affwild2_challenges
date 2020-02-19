import matplotlib.pyplot as plt, numpy as np, os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, classification_report

def model_report(y_true_label, y_pred_label, label_names):
    s_report = classification_report(y_true_label, y_pred_label, target_names=label_names)

    model_cm_int = confusion_matrix(y_true_label, y_pred_label)
    model_cm_norm = (model_cm_int.astype('float') / model_cm_int.sum(axis=1)[:, np.newaxis])

    model_acc = (model_cm_int.diagonal() / model_cm_int.sum(axis=1))

    model_f1 = f1_score(y_true_label, y_pred_label, average=None)
    model_f1_micro = f1_score(y_true_label, y_pred_label, average="micro")
    model_f1_macro = f1_score(y_true_label, y_pred_label, average="macro")
    model_f1_weighted = f1_score(y_true_label, y_pred_label, average="weighted")

    model_recall = recall_score(y_true_label, y_pred_label, average=None)
    model_recall_micro = recall_score(y_true_label, y_pred_label, average="micro")
    model_recall_macro = recall_score(y_true_label, y_pred_label, average="macro")
    model_recall_weighted = recall_score(y_true_label, y_pred_label, average="weighted")

    model_precision = precision_score(y_true_label, y_pred_label, average=None)
    model_precision_micro = precision_score(y_true_label, y_pred_label, average="micro")
    model_precision_macro = precision_score(y_true_label, y_pred_label, average="macro")
    model_precision_weighted = precision_score(y_true_label, y_pred_label, average="weighted")

    model_acc_all = np.sum(model_cm_int.diagonal()) / np.sum(model_cm_int)
    model_support = np.sum(model_cm_int, axis=1)
    model_support_all = np.sum(model_support)
    model_f1_avg_weighted = np.sum(model_f1 * model_support) / model_support_all
    model_recall_avg_weighted = np.sum(model_recall * model_support) / model_support_all
    model_precision_avg_weighted = np.sum(model_precision * model_support) / model_support_all

    model_f1_avg = np.average(model_f1)
    model_recall_avg = np.average(model_recall)
    model_precision_avg = np.average(model_precision)

    names = ["model_cm_int", "model_cm_norm", "model_acc", "model_f1", "model_f1_micro", "model_f1_macro",
             "model_f1_weighted",
             "model_recall", "model_recall_micro", "model_recall_macro", "model_recall_weighted",
             "model_precision", "model_precision_micro", "model_precision_macro", "model_precision_weighted",
             "model_acc_all", "model_support", "model_support_all", "model_f1_avg_weighted",
             "model_recall_avg_weighted",
             "model_precision_avg_weighted", "model_f1_avg", "model_recall_avg", "model_precision_avg", "s_report",
             "label_names"]
    save_summary = {}
    for key in names: save_summary[key] = eval(key)
    return save_summary
# model_report

def buffer_print_string(print_handle_fn, *argc, **kwargs):
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO
    import sys

    # keep track of the original sys.stdout
    origStdout = sys.stdout

    # replace sys.stdout temporarily with our own buffer
    outputBuf = StringIO()
    sys.stdout = outputBuf

    try:
        print_handle_fn(*argc, **kwargs)
    except Exception as e:
        print("Error: ", e)

    # put back the original stdout
    sys.stdout = origStdout

    # get the model summary as a string
    string_buffer = outputBuf.getvalue()
    return string_buffer
# buffer_print_string

def print_summary(save_summary):
    print_opts = np.get_printoptions();
    np.set_printoptions(precision=2);

    print("Classification Report: \n", save_summary["s_report"])
    print("Confustion Matrix Int: \n", save_summary["model_cm_int"])
    print("\nConfustion Matrix Norm: \n", save_summary["model_cm_norm"] * 100.0)

    print("\nLabels:\t\t", save_summary["label_names"])
    print("Accuracy:\t", save_summary["model_acc"], " -- Acc: ", "%.2f%%" % (save_summary["model_acc_all"] * 100.0))
    print("F1:\t\t", save_summary["model_f1"], " -- Avg: ", "Weighted: %.2f - No: %.2f"
          % (save_summary["model_f1_avg_weighted"], save_summary["model_f1_avg"]))
    print("Recall:\t\t", save_summary["model_recall"], " -- Avg: ", "Weighted: %.2f - No: %.2f"
          % (save_summary["model_recall_avg_weighted"], save_summary["model_recall_avg"]))
    print("Precision:\t", save_summary["model_precision"], " -- Avg: ", "Weighted: %.2f - No: %.2f"
          % (save_summary["model_precision_avg_weighted"], save_summary["model_precision_avg"]))
    print("Support:\t", save_summary["model_support"], " -- All: ", save_summary["model_support_all"])

    np.set_printoptions(**print_opts)
# print_summary

def plot_confusion_matrix(y_test, y_pred, classes=None,
                          normalize=True,
                          title='Average accuracy \n',
                          cmap=plt.cm.Blues,
                          verbose=0, precision=0,
                          text_size=10,
                          title_size=25,
                          axis_label_size=16,
                          tick_size=14, save_path=None, 
                          has_colorbar = False):
    """
    https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    from sklearn.metrics import confusion_matrix
    import itertools

    cm = confusion_matrix(y_test, y_pred)
    acc = sum(cm.diagonal() / cm.sum()) * 100.0
    if normalize:
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]) * 100.0
        if verbose == 1: print("Normalized confusion matrix")
    else:
        if verbose == 1: print('Confusion matrix, without normalization')

    if verbose == 1: print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title.format_map({'acc': acc}), fontsize=title_size)
    if has_colorbar == True: plt.colorbar()

    if classes is not None:
        ax = plt.gca()
        # tick_marks = np.arange(len(classes))
        # plt.xticks(tick_marks, classes, rotation=45, fontsize=tick_size)
        # plt.yticks(tick_marks, classes, fontsize=tick_size)
        ax.set_xticklabels(classes, fontsize=tick_size, rotation=45)
        ax.set_yticklabels(classes, fontsize=tick_size)

    fmt = '{:.' + '%d' % (precision) + 'f} %' if normalize else '{:d} %'
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, fmt.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=text_size)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=axis_label_size)
    plt.xlabel('Predicted label', fontsize=axis_label_size)

    if save_path is not None:
        dirname = os.path.dirname(save_path)
        if dirname != "" and os.path.exists(dirname) == False: os.makedirs(dirname)
        plt.savefig(save_path)
# plot_confusion_matrix