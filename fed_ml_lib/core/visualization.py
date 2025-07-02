import random
import torch
import shutil
import yaml
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sn
import pandas as pd
import torch.nn.functional
from collections import OrderedDict
from typing import List

def save_matrix(y_true, y_pred, path, classes):
    """
    Save the confusion matrix in the path given in argument.

    :param y_true: true labels (real labels)
    :param y_pred: predicted labels (labels predicted by the model)
    :param path: path to save the confusion matrix
    :param classes: list of the classes
    """
    # To get the normalized confusion matrix
    y_true_mapped = [classes[label] for label in y_true]
    y_pred_mapped = [classes[label] for label in y_pred]
    # To get the normalized confusion matrix
    cf_matrix_normalized = confusion_matrix(y_true_mapped, y_pred_mapped, labels=classes, normalize='all')
   
    # To round up the values in the matrix
    cf_matrix_round = np.round(cf_matrix_normalized, 2)

    # To plot the matrix
    df_cm = pd.DataFrame(cf_matrix_round, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.xlabel("Predicted label", fontsize=13)
    plt.ylabel("True label", fontsize=13)
    plt.title("Confusion Matrix", fontsize=15)

    plt.savefig(path)
    plt.close()

def save_roc(targets, y_proba, path, nbr_classes):
    """
    Save the roc curve in the path given in argument.

    :param targets: true labels (real labels)
    :param y_proba: predicted labels (labels predicted by the model)
    :param path: path to save the roc curve
    :param nbr_classes: number of classes
    """
    y_true = np.zeros(shape=(len(targets), nbr_classes))  # array-like of shape (n_samples, n_classes)
    for i in range(len(targets)):
        y_true[i, targets[i]] = 1

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(nbr_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nbr_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nbr_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nbr_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    lw = 2
    for i in range(nbr_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw, label='Worst case')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic (ROC) Curve OvR")  # One vs Rest
    plt.legend(loc="lower right")  # loc="best"

    plt.savefig(path)
    plt.close()

def save_graphs(path_save, local_epoch, results, end_file=""):
    """
    Save the graphs in the path given in argument.

    :param path_save: path to save the graphs
    :param local_epoch: number of epochs
    :param results: results of the model (accuracy and loss)
    :param end_file: end of the name of the file
    """
    os.makedirs(path_save, exist_ok=True)  # to create folders results
    print("save graph in ", path_save)
    # plot training curves (train and validation)
    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_acc"], results["val_acc"]],
        "Epochs", "Accuracy (%)",
        curve_labels=["Training accuracy", "Validation accuracy"],
        title="Accuracy curves",
        path=path_save + "Accuracy_curves" + end_file)

    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_loss"], results["val_loss"]],
        "Epochs", "Loss",
        curve_labels=["Training loss", "Validation loss"], title="Loss curves",
        path=path_save + "Loss_curves" + end_file)
    
def save_graphs_multimodal(path_save, local_epoch, results, end_file=""):
    """
    Save the graphs in the path given in argument.

    :param path_save: path to save the graphs
    :param local_epoch: number of epochs
    :param results: results of the model (accuracy and loss)
    :param end_file: end of the name of the file
    """
    os.makedirs(path_save, exist_ok=True)  # to create folders results
    print("save graph in ", path_save)
    # plot training curves (train and validation)
    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_acc_mri"], results["val_acc_mri"]],
        "Epochs", "Accuracy (%)",
        curve_labels=["Training accuracy", "Validation accuracy"],
        title="Accuracy curves",
        path=path_save + "DNA_Accuracy_curves" + end_file)

    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_loss_mri"], results["val_loss_mri"]],
        "Epochs", "Loss",
        curve_labels=["Training loss", "Validation loss"], title="Loss curves",
        path=path_save + "DNA_Loss_curves" + end_file)  

    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_acc_dna"], results["val_acc_dna"]],
        "Epochs", "Accuracy (%)",
        curve_labels=["Training accuracy", "Validation accuracy"],
        title="Accuracy curves",
        path=path_save + "MRI_Accuracy_curves" + end_file)

    plot_graph(
        [[*range(local_epoch)]] * 2,
        [results["train_loss_dna"], results["val_loss_dna"]],
        "Epochs", "Loss",
        curve_labels=["Training loss", "Validation loss"], title="Loss curves",
        path=path_save + "MRI_Loss_curves" + end_file)    

def plot_graph(list_xplot, list_yplot, x_label, y_label, curve_labels, title, path=None):
    """
    Plot the graph of the list of points (list_xplot, list_yplot)
    :param list_xplot: list of list of points to plot (one line per curve)
    :param list_yplot: list of list of points to plot (one line per curve)
    :param x_label: label of the x axis
    :param y_label: label of the y axis
    :param curve_labels: list of labels of the curves (curve names)
    :param title: title of the graph
    :param path: path to save the graph
    """
    lw = 2

    plt.figure()
    for i in range(len(curve_labels)):
        plt.plot(list_xplot[i], list_yplot[i], lw=lw, label=curve_labels[i])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if curve_labels:
        plt.legend(loc="lower right")

    if path:
        plt.savefig(path)
    plt.close()


# New convenience function to save all visualizations at once
def save_all_results(
    train_true,
    train_pred, 
    train_proba,
    test_true,
    test_pred,
    test_proba,
    training_history,
    classes,
    results_path,
    config,
    file_suffix=""
):
    """
    Save all standard visualization outputs with a single function call.
    
    This convenience function eliminates the repetitive pattern of calling
    save_matrix, save_roc, and save_graphs separately across all examples.
    
    Args:
        train_true: True labels for training data
        train_pred: Predicted labels for training data  
        train_proba: Predicted probabilities for training data
        test_true: True labels for test data
        test_pred: Predicted labels for test data
        test_proba: Predicted probabilities for test data
        training_history: Training history dictionary with train_acc, val_acc, train_loss, val_loss
        classes: List of class names for confusion matrix labels
        results_path: Base path to save all results
        config: Configuration dictionary containing epochs and other settings
        file_suffix: Optional suffix to append to filenames (e.g., "_pill", "_dna")
        
    Example:
        # Before (37 lines of repetitive code):
        save_matrix(train_true, train_pred, classes, f"{results_path}/confusion_matrix_train.png")
        save_matrix(test_true, test_pred, classes, f"{results_path}/confusion_matrix_test.png") 
        save_roc(train_true, train_proba, f"{results_path}/roc_train.png", len(classes))
        save_roc(test_true, test_proba, f"{results_path}/roc_test.png", len(classes))
        save_graphs(f"{results_path}/", config['epochs'], training_history, "")
        
        # After (1 line):
        save_all_results(train_true, train_pred, train_proba, test_true, test_pred, 
                        test_proba, training_history, classes, results_path, config)
    """
    # Ensure results directory exists
    os.makedirs(results_path, exist_ok=True)
    
    # Save confusion matrices
    save_matrix(
        y_true=train_true,
        y_pred=train_pred,
        classes=classes,
        path=os.path.join(results_path, f"confusion_matrix_train{file_suffix}.png")
    )
    
    save_matrix(
        y_true=test_true,
        y_pred=test_pred,
        classes=classes,
        path=os.path.join(results_path, f"confusion_matrix_test{file_suffix}.png")
    )
    
    # Save ROC curves
    num_classes = len(classes)
    save_roc(
        targets=train_true,
        y_proba=train_proba,
        path=os.path.join(results_path, f"roc_train{file_suffix}.png"),
        nbr_classes=num_classes
    )
    
    save_roc(
        targets=test_true,
        y_proba=test_proba,
        path=os.path.join(results_path, f"roc_test{file_suffix}.png"),
        nbr_classes=num_classes
    )
    
    # Save training curves
    save_graphs(
        path_save=results_path + "/",
        local_epoch=config.get('epochs', len(training_history.get('train_acc', []))),
        results=training_history,
        end_file=file_suffix
    )
    
    print(f"All visualization results saved to: {results_path}")
    print(f"Generated files:")
    print(f"  - confusion_matrix_train{file_suffix}.png")
    print(f"  - confusion_matrix_test{file_suffix}.png") 
    print(f"  - roc_train{file_suffix}.png")
    print(f"  - roc_test{file_suffix}.png")
    print(f"  - Accuracy_curves{file_suffix}.png") 
    print(f"  - Loss_curves{file_suffix}.png")
