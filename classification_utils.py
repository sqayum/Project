import numpy as np
import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, recall_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import validation_curve
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator


def _scale_bar_width(ax, factor, *, horizontal=False):
    from math import isclose

    if not horizontal:
        sorted_patches = sorted(ax.patches, key=lambda x: x.get_x())
        for i, patch in enumerate(sorted_patches):
            current_width = patch.get_width()

            updated_current_width = factor * current_width
            patch.set_width(updated_current_width)

            if i == len(sorted_patches) - 1:
                return

            current_x = patch.get_x()
            next_x = sorted_patches[i+1].get_x()
            if isclose(current_x + current_width, next_x, rel_tol=1e-7, abs_tol=1e-7):
                patch.set_x(next_x - updated_current_width)
    else:
        sorted_patches = sorted(ax.patches, key=lambda x: x.get_y())
        for i, patch in enumerate(sorted_patches):
            current_width = patch.get_width()

            updated_current_width = factor * current_width
            patch.set_width(updated_current_width)

            if i == len(sorted_patches) - 1:
                return

            current_y = patch.get_y()
            next_y = sorted_patches[i+1].get_y()
            if isclose(current_y + current_width, next_y, rel_tol=1e-7, abs_tol=1e-7):
                patch.set_y(next_y - updated_current_width)


def get_classification_metrics(y_true,
                                   y_pred,
                                   *,
                                   target_names,
                                   sample_weight=None,
                                   average_only=False):

    binary = False
    if len(target_names) == 2:
        binary = True

    metrics_dict = classification_report(y_true,
                                             y_pred,
                                             target_names=target_names,
                                             sample_weight=sample_weight,
                                             output_dict=True)

    accuracy = round(metrics_dict["accuracy"], 3)
    num_labels = len(target_names)

    metrics_df = pd.DataFrame(metrics_dict)
    target_names = list(target_names)
    if sample_weight is not None:
        target_names.append("weighted avg")
        metrics_df = metrics_df.loc["precision":"f1-score", target_names].apply(lambda x: round(x, 3))
        metrics_df.rename(columns={"weighted avg": "Weighted Average"}, inplace=True)
    else:
        target_names.append("macro avg")
        metrics_df = metrics_df.loc["precision":"f1-score", target_names].apply(lambda x: round(x, 3))
        metrics_df.rename(columns={"macro avg": "Average"}, inplace=True)
    metrics_df.index = ["Precision", "Recall", "F1-Score"]

    metrics_df.loc["Accuracy", metrics_df.columns[num_labels:]] = accuracy

    if not binary and average_only:
        metrics_df = pd.DataFrame(metrics_df["Average"])

    return metrics_df


def _get_roc_values(estimator,
                       X_train,
                       X_validate,
                       y_train,
                       y_validate):

    if 'predict_proba' in dir(estimator):
        y_train_score = estimator.predict_proba(X_train)[:,1]
        y_validate_score = estimator.predict_proba(X_validate)[:,1]

    else:
        y_train_score = estimator.decision_function(X_train)
        y_validate_score = estimator.decision_function(X_validate)

    train_fpr, train_tpr, thresholds = roc_curve(y_train, y_train_score)
    validate_fpr, validate_tpr, thresholds = roc_curve(y_validate, y_validate_score)

    train_auc = round(auc(train_fpr, train_tpr), 2)
    validate_auc = round(auc(validate_fpr, validate_tpr), 2)

    return train_fpr, train_tpr, validate_fpr, validate_tpr, train_auc, validate_auc

def plot_roc_curve(estimator,
                      X_train,
                      X_validate,
                      y_train,
                      y_validate,
                      *,
                      figsize=None,
                      filepath=None):

    train_fpr, train_tpr, validate_fpr, validate_tpr, train_auc, validate_auc = _get_roc_values(estimator,
                                                                                                           X_train,
                                                                                                           X_validate,
                                                                                                           y_train,
                                                                                                           y_validate)
    if figsize is None:
        figsize = (12,7)

    fig, ax = fig.add_subplots(figsize=figsize)
    ax.plot(train_fpr, train_tpr, color="tab:blue", label=f'Training (AUC = {train_auc})')
    ax.plot(validate_fpr, validate_tpr, color="tab:orange", label=f'Validation (AUC = {validate_auc})')
    ax.plot([0,1], [0,1], color='red', ls=':')
    ax.set(
        title='ROC Curve',
        xlabel='False Positive Rate',
        ylabel='True Positive Rate')
    ax.legend()
    plt.show()

    if filepath is not None:
        fig.savefig(filepath)


def _get_validation_confusion_matrices(y_train_true,
                                            y_train_pred,
                                            y_validate_true,
                                            y_validate_pred,
                                            *,
                                            sample_weight=None,
                                            normalize=None):



    cm_train = confusion_matrix(y_train_true,
                                     y_train_pred,
                                     sample_weight=sample_weight,
                                     normalize=normalize)

    cm_validate = confusion_matrix(y_validate_true,
                                        y_validate_pred,
                                        normalize=normalize)

    cm_train_values = [f"{round(value, 2):.2f}" for value in cm_train.flatten()]
    cm_train_labels = np.asarray(cm_train_values).reshape(cm_train.shape[0], cm_train.shape[1])

    cm_validate_values = [f"{round(value, 2):.2f}" for value in cm_validate.flatten()]
    cm_validate_labels = np.asarray(cm_validate_values).reshape(cm_validate.shape[0], cm_validate.shape[1])

    return cm_train, cm_train_labels, cm_validate, cm_validate_labels

def plot_validation_confusion_matrices(y_train_true,
                                            y_train_pred,
                                            y_validate_true,
                                            y_validate_pred,
                                            *,
                                            target_names=None,
                                            sample_weight=None,
                                            normalize=None,
                                            cbar=False,
                                            mode="validate",
                                            figsize=None,
                                            filepath=None):


    if target_names is None:
        target_names = sorted(list(set(y_train_true)))

    cm_train, cm_train_labels, cm_validate, cm_validate_labels = _get_validation_confusion_matrices(y_train_true,
                                                                                                                   y_train_pred,
                                                                                                                   y_validate_true,
                                                                                                                   y_validate_pred,
                                                                                                                   sample_weight=sample_weight,
                                                                                                                   normalize=normalize)

    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 19

    if figsize is None:
        figsize = (13, 5)

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, figure=fig)
    gs.update(wspace=0.4)

    ax1 = fig.add_subplot(gs[0,0])
    sns.heatmap(cm_train,
                   annot=cm_train_labels,
                   fmt="",
                   cmap="Blues",
                   cbar=cbar,
                   xticklabels=target_names,
                   yticklabels=target_names,
                   ax=ax1)

    ax1.set(title="Confusion Matrix (Train)",
             ylabel ='True Label',
             xlabel ='Predicted Label')

    ax2 = fig.add_subplot(gs[0,1])
    sns.heatmap(cm_validate,
                   annot=cm_validate_labels,
                   fmt="",
                   cmap="Oranges",
                   cbar=cbar,
                   xticklabels=target_names,
                   yticklabels=target_names,
                   ax=ax2)

    if mode == "validate":
        title = "Confusion Matrix (Validation)"
    elif mode == "test":
        title = "Confusion Matrix (Test)"

    ax2.set(title=title,
             ylabel ='True Label',
             xlabel ='Predicted Label')

    if filepath is not None:
        fig.savefig(filepath)



def _get_binary_validation_metrics(y_train_true,
                                       y_train_pred,
                                       y_validate_true,
                                       y_validate_pred,
                                       *,
                                       pos_label=1,
                                       sample_weight=None):


    metrics = {'Train':{}, 'Validate': {}}
    for score_name, score_func in [('Accuracy', accuracy_score), ('Recall', recall_score), ('Precision', precision_score), ('F1', f1_score)]:
        if score_name == 'Precision' or score_name == 'Recall':
            metrics['Train'].update({score_name: score_func(y_train_true, y_train_pred, average='binary', pos_label=pos_label, sample_weight=sample_weight)})
            metrics['Validate'].update({score_name: score_func(y_validate_true, y_validate_pred, average='binary', pos_label=pos_label)})
        else:
            metrics['Train'].update({score_name: score_func(y_train_true, y_train_pred, sample_weight=sample_weight)})
            metrics['Validate'].update({score_name: score_func(y_validate_true, y_validate_pred)})

    return(pd.DataFrame(metrics))

def plot_binary_validation_metrics(y_train_true,
                                       y_train_pred,
                                       y_validate_true,
                                       y_validate_pred,
                                       *,
                                       pos_label=1,
                                       sample_weight=None,
                                       cm_normalize=None,
                                       cm_cbar=False,
                                       mode="validate",
                                       filepath=None):


    metrics_df = _get_binary_validation_metrics(y_train_true,
                                                      y_train_pred,
                                                      y_validate_true,
                                                      y_validate_pred,
                                                      pos_label=pos_label,
                                                      sample_weight=sample_weight)

    metrics_df = metrics_df.unstack().reset_index().rename(columns={'level_0': 'Dataset', 'level_1': 'Score', 0: 'Value'})


    cm_train, cm_train_labels, cm_validate, cm_validate_labels = _get_validation_confusion_matrices(y_train_true,
                                                                                                                   y_train_pred,
                                                                                                                   y_validate_true,
                                                                                                                   y_validate_pred,
                                                                                                                   sample_weight=sample_weight,
                                                                                                                   normalize=cm_normalize)
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 19

    fig = plt.figure(figsize=(12,18))
    gs = GridSpec(nrows=2, ncols=2, figure=fig)
    gs.update(wspace=0.3, hspace=0.3)

    if mode == "validate":
        title = "Validation Metrics"
    elif mode == "test":
        title = "Test Metrics"

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set(title=title, ylim=(0,1))
    ax1.yaxis.set_major_locator(MultipleLocator(base=0.1))
    sns.barplot(data=metrics_df, x="Score", y="Value", hue="Dataset", ax=ax1)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.grid(False)
    sns.heatmap(cm_train,
                   annot=cm_train_labels,
                   fmt="",
                   cmap="Blues",
                   cbar=cm_cbar,
                   xticklabels=target_names,
                   yticklabels=target_names,
                   ax=ax2)

    ax2.set(title="Confusion Matrix (Train)",
             ylabel ='True Label',
             xlabel ='Predicted Label')

    ax3 = fig.add_subplot(gs[1,1])
    sns.heatmap(cm_validate,
                   annot=cm_validate_labels,
                   fmt="",
                   cmap="Oranges",
                   cbar=cm_cbar,
                   xticklabels=target_names,
                   yticklabels=target_names,
                   ax=ax3)

    if mode == "validate":
        title = "Confusion Matrix (Validation)"
    elif mode == "test":
        title = "Confusion Matrix (Test)"

    ax3.set(title=title,
             ylabel ='True Label',
             xlabel ='Predicted Label')

    plt.show()

    if filepath is not None:
        fig.savefig(filepath)



def plot_validation_curve(estimator,
                             X_train,
                             y_train,
                             *,
                             param_name,
                             param_range,
                             scoring,
                             scoring_label,
                             fit_params=None,
                             cv=5,
                             semilogx=False,
                             n_jobs=-1,
                             figsize=(16,12),
                             filepath=None):

    estimator_name = str(estimator)

    train_scores, test_scores = validation_curve(estimator,
                                                       X_train,
                                                       y_train,
                                                       param_name=param_name,
                                                       param_range=param_range,
                                                       scoring=scoring,
                                                       fit_params=fit_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       verbose=2)

    avg_train_scores = np.array([np.average(train_scores[i,:]) for i in range(train_scores.shape[0])])
    avg_test_scores = np.array([np.average(test_scores[i,:]) for i in range(test_scores.shape[0])])

    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.titlesize'] = 18

    fig, (ax1,ax2) = plt.subplots(figsize=figsize, nrows=2, ncols=1, sharex=True)
    if not semilogx:
        ax1.plot(param_range, avg_train_scores, param_range, avg_test_scores)
        ax2.plot(param_range, np.abs(avg_train_scores - avg_test_scores), 'r--')
    else:
        ax1.semilogx(param_range, avg_train_scores, param_range, avg_test_scores)
        ax2.semilogx(param_range, np.abs(avg_train_scores - avg_test_scores), 'r--')

    ax1.set(title=f'{scoring_label} vs {param_name} [{estimator_name}]',
             ylabel=scoring_label)
    ax2.set(xlabel=f'{param_name}',
             ylabel=f'{scoring_label} Deviation')

    ax1.legend(['Train', 'Validate'])

    plt.show()

    if filepath is not None:
        fig.savefig(filepath)

def _get_multilabel_validation_metric(y_train_true,
                                           y_train_pred,
                                           y_validate_true,
                                           y_validate_pred,
                                           *,
                                           score_names,
                                           score_label,
                                           target_names,
                                           sample_weight,
                                           mode):

    N = len(score_names)

    for i, score_name in enumerate(score_names):
        if N == 1:
            axes = [axes]

        if '-' in score_name:
            score_label = '-'.join([x.capitalize() for x in score_name.split('-')])
        else:
            score_label = score_name.capitalize()

        train_metrics_dict = classification_report(y_train_true,
                                                        y_train_pred,
                                                        target_names=target_names,
                                                        sample_weight=sample_weight,
                                                        output_dict=True)

        validate_metrics_dict = classification_report(y_validate_true,
                                                        y_validate_pred,
                                                        target_names=target_names,
                                                        output_dict=True)

        train_metric_df = pd.DataFrame(train_metrics_dict).loc[score_name, target_names]
        train_metric_df.name = f"Train (Accuracy = ${round(train_metrics_dict['accuracy'], 3)}$)"

        validate_metric_df = pd.DataFrame(validate_metrics_dict).loc[score_name, target_names]
        if mode == "validate":
            validate_metric_df.name = f"Validate (Accuracy = ${round(validate_metrics_dict['accuracy'], 3)}$)"
        if mode == "test":
            validate_metric_df.name = f"Test (Accuracy = ${round(validate_metrics_dict['accuracy'], 3)}$)"

        return pd.concat([train_metric_df, validate_metric_df], axis=1).stack().reset_index().rename(columns={'level_0': 'Label', 'level_1': 'Dataset', 0: score_label})

def plot_validation_metrics_by_label(y_train_true,
                                         y_train_pred,
                                         y_validate_true,
                                         y_validate_pred,
                                         *,
                                         score_names,
                                         target_names,
                                         estimator_label,
                                         sample_weight,
                                         mode="validate",
                                         figsize=None,
                                         filepath=None):

    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 19

    N = len(score_names)
    if figsize is None:
        figsize = (11,N*5)
    fig, axes = plt.subplots(figsize=figsize, nrows=N)

    for i, score_name in enumerate(score_names):
        if N == 1:
            axes = [axes]

        if '-' in score_name:
            score_label = '-'.join([x.capitalize() for x in score_name.split('-')])
        else:
            score_label = score_name.capitalize()

        score_df = _get_multilabel_validation_metric(y_train_true,
                                               y_train_pred,
                                               y_validate_true,
                                               y_validate_pred,
                                               score_name=score_name,
                                               score_label=score_label,
                                               target_names=target_names,
                                               sample_weight=sample_weight,
                                               mode=mode)

        sns.barplot(x=score_df["Label"],
                      y=score_df[score_label],
                      hue=score_df["Dataset"],
                      ax=axes[i])

        axes[i].set(xlabel=None, ylim=(0,1))
        _scale_bar_width(axes[i], 0.55)
        axes[i].yaxis.set_major_locator(MultipleLocator(base=0.1))
        if i == 0:
            if mode == "validate":
                title = f"Validation Metrics by Label [{estimator_label}]"
            elif mode == "test":
                title = f"Test Metrics by Label [{estimator_label}]"
            axes[i].set(title=title)
            axes[i].legend(loc="best", prop={'size': 14}, ncol=2)
        else:
            axes[i].get_legend().remove()

    if filepath is not None:
        fig.savefig(filepath)