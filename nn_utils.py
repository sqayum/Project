import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os


def plot_history(model_name,
                   history_dict,
                   *,
                   figsize=(16,14),
                   filepath=None):

    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 9

    if '_' in model_name:
        model_name = ' '.join(model_name.split('_')).upper()

    training_losses = history_dict['loss']
    validation_losses = history_dict['val_loss']

    epochs = range(1, len(training_losses)+1)

    fig, [ax1, ax2, ax3] = plt.subplots(figsize=figsize, nrows=3, sharex=True)

    ax1.plot(epochs, training_losses, color='tab:blue', label='Training Set')
    ax1.plot(epochs, validation_losses, color='tab:orange', label='Validation Set')
    ax1.set(title=f"History ({model_name})" , ylabel='Crossentropy Loss')
    ax1.legend()

    training_issue_accuracyes = history_dict['issue_accuracy']
    validation_issue_accuracyes = history_dict['val_issue_accuracy']

    ax2.plot(epochs, training_issue_accuracyes, color='tab:blue', label='Training Set')
    ax2.plot(epochs, validation_issue_accuracyes, color='tab:orange', label='Validation Set')
    ax2.set(ylabel='Accuracy (ISSUE)', ylim=(0,1))
    ax2.yaxis.set_major_locator(MultipleLocator(base=0.1))

    training_stance_accuracy = history_dict['stance_accuracy']
    validation_stance_accuracy = history_dict['val_stance_accuracy']

    ax3.plot(epochs, training_stance_accuracy, color='tab:blue', label='Training Set (category = STANCE)')
    ax3.plot(epochs, validation_stance_accuracy, color='tab:orange', label='Validation Set (category = STANCE)')
    ax3.set(ylabel='Accuracy (STANCE)', xlabel='Epoch', ylim=(0,1))
    ax3.yaxis.set_major_locator(MultipleLocator(base=0.1))
    ax3.xaxis.set_major_locator(MultipleLocator(base=5))

    plt.show()

    dirname = os.path.split(filepath)[0]
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    fig.savefig(filepath)
