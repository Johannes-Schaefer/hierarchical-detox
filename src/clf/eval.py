import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from data import datasets
from data import process_data
from collections import defaultdict


def get_loss(model, output, labels, device, criteria):
    if criteria is None:
        return 0., {}
    task_batch_losses = {}
    tox_criterion = criteria.get('toxicity_out', None)
    identity_criterion = criteria.get('identity_present_out', None)
    identity_category_criterion = criteria.get('identity_category_out', None)
    identity_term_criterion = criteria.get('identity_term_out', None)

    # note: this is ugly, but it works. should simply use a loop instead of the following four if statements...
    output_index = 0
    if tox_criterion is not None:
        tox_output = output[output_index]
        tox_labels = labels[:, datasets.DATA_LABEL_POSITION_MAPPING['toxicity'],
                            0:model.output_size['toxicity_out']]
        tox_labels = tox_labels.to(device)
        tox_loss = tox_criterion(torch.squeeze(tox_output), torch.squeeze(tox_labels.float()))
        task_batch_losses['toxicity'] = tox_loss
        output_index += 1

    if identity_criterion is not None:
        identity_output = output[output_index]
        identity_labels = labels[:, datasets.DATA_LABEL_POSITION_MAPPING['identity_present'],
                                 0:model.output_size['identity_present_out']]
        identity_labels = identity_labels.to(device)
        identity_loss = identity_criterion(torch.squeeze(identity_output), torch.squeeze(identity_labels.float()))
        task_batch_losses['identity_present'] = identity_loss
        output_index += 1

    if identity_category_criterion is not None:
        identity_output = output[output_index]
        identity_labels = labels[:, datasets.DATA_LABEL_POSITION_MAPPING['identity_category'],
                                 0:model.output_size['identity_category_out']]
        identity_labels = identity_labels.to(device)
        identity_loss = identity_category_criterion(torch.squeeze(identity_output),
                                                    torch.squeeze(identity_labels.float()))
        task_batch_losses['identity_category'] = identity_loss
        output_index += 1

    if identity_term_criterion is not None:
        identity_output = output[output_index]
        identity_labels = labels[:, datasets.DATA_LABEL_POSITION_MAPPING['identity_term'],
                                 0:model.output_size['identity_term_out']]
        identity_labels = identity_labels.to(device)
        identity_loss = identity_term_criterion(torch.squeeze(identity_output), torch.squeeze(identity_labels.float()))
        task_batch_losses['identity_term'] = identity_loss
        output_index += 1

    total_batch_loss = (task_batch_losses.get('toxicity', 0.) + task_batch_losses.get('identity_present', 0.)
                        + task_batch_losses.get('identity_category', 0.) + task_batch_losses.get('identity_term', 0.))

    return total_batch_loss, task_batch_losses


def calculate_prf_binary(scores):
    # returns: scores, {'pos class prec': 0, 'pos class rec': 1, 'pos class f': 2,
    #                   'neg class prec': 3, 'neg class rec': 4, 'neg class f': 5}
    result_scores = {}
    for index in (0, 1):
        res_class = 'pos class' if index == 0 else 'neg class'
        try:
            result_scores[res_class + ' prec'] = scores[0+4*index]\
                                                 / max(float(scores[0+4*index] + scores[2+4*index]), 1.0)
        except ZeroDivisionError:
            result_scores[res_class + ' prec'] = 0.0
        try:
            result_scores[res_class + ' rec'] = scores[0+4*index]\
                                                 / max(float(scores[0+4*index] + scores[3+4*index]), 1.0)
        except ZeroDivisionError:
            result_scores[res_class + ' rec'] = 0.0
        try:
            result_scores[res_class + ' f'] = \
                2 * result_scores[res_class + ' prec'] * result_scores[res_class + ' rec']\
                / max((result_scores[res_class + ' prec'] + result_scores[res_class + ' rec']), 1.0)
        except ZeroDivisionError:
            result_scores[res_class + ' f'] = 0.0
    return {k: v if v != np.nan else 0.0 for k, v in result_scores.items()}


def eval_pred(model, pred, labels, device):
    result_scores = {}
    output_index = 0
    if 'toxicity_out' in model.output_size:
        tox_output = torch.sigmoid(pred[output_index])
        tox_labels = labels[:, datasets.DATA_LABEL_POSITION_MAPPING['toxicity'],
                            0:model.output_size['toxicity_out']]
        tox_labels = tox_labels.to(device)
        tox_scores = eval_binary_pred_values(torch.squeeze(tox_output), torch.squeeze(tox_labels))
        p_tp, p_tn, p_fp, p_fn, n_tp, n_tn, n_fp, n_fn = tox_scores
        result_scores['tox acc'] = (p_tp + p_tn) / len(tox_labels)
        result_scores['tox_fp'] = p_fp
        result_scores['tox_fn'] = p_fn
        for score_name, score in calculate_prf_binary(tox_scores).items():
            result_scores['tox ' + score_name] = score
        # this calc of roc auc requires values to be on cpu
        tox_labels_cpu = tox_labels.cpu().detach().numpy()
        tox_output_cpu = tox_output.cpu().detach().numpy()
        try:
            result_scores['tox auc'] = roc_auc_score(tox_labels_cpu, tox_output_cpu)
        except ValueError:
            # in cases where only one class is present in y_true. ROC AUC score is not defined in that case.
            result_scores['tox auc'] = 0.
        output_index += 1

    if 'identity_present_out' in model.output_size:
        identity_output = torch.sigmoid(pred[output_index])
        identity_labels = labels[:, datasets.DATA_LABEL_POSITION_MAPPING['identity_present'],
                                 0:model.output_size['identity_present_out']]
        identity_labels = identity_labels.to(device)
        id_scores = eval_binary_pred_values(torch.squeeze(identity_output), torch.squeeze(identity_labels))
        p_tp, p_tn, p_fp, p_fn, n_tp, n_tn, n_fp, n_fn = id_scores
        result_scores['idp acc'] = (p_tp + p_tn) / len(identity_labels)
        for score_name, score in calculate_prf_binary(id_scores).items():
            result_scores['idp ' + score_name] = score
        identity_labels_cpu = identity_labels.cpu().detach().numpy()
        identity_output_cpu = identity_output.cpu().detach().numpy()
        try:
            result_scores['idp auc'] = roc_auc_score(identity_labels_cpu, identity_output_cpu)
        except ValueError:
            # in cases where only one class is present in y_true. ROC AUC score is not defined in that case.
            result_scores['idp auc'] = 0.
        output_index += 1
    elif 'identity_category_out' in model.output_size or 'identity_term_out' in model.output_size:
        # infer idp pred from idc or idt pred
        identity_output = torch.sigmoid(pred[output_index])
        identity_output = identity_output.cpu().detach().numpy()
        identity_output = [[1. if any([v >= 0.5 for v in vec]) else 0.] for vec in identity_output]
        identity_output = torch.FloatTensor(identity_output)
        identity_output = identity_output.to(device)
        
        identity_labels = labels[:, datasets.DATA_LABEL_POSITION_MAPPING['identity_present'], 0:1]
        identity_labels = identity_labels.to(device)
        id_scores = eval_binary_pred_values(torch.squeeze(identity_output), torch.squeeze(identity_labels))
        p_tp, p_tn, p_fp, p_fn, n_tp, n_tn, n_fp, n_fn = id_scores
        result_scores['idp acc'] = (p_tp + p_tn) / len(identity_labels)
        for score_name, score in calculate_prf_binary(id_scores).items():
            result_scores['idp ' + score_name] = score
        identity_labels_cpu = identity_labels.cpu().detach().numpy()
        identity_output_cpu = identity_output.cpu().detach().numpy()
        try:
            result_scores['idp auc'] = roc_auc_score(identity_labels_cpu, identity_output_cpu)
        except ValueError:
            # in cases where only one class is present in y_true. ROC AUC score is not defined in that case.
            result_scores['idp auc'] = 0.

    if 'identity_category_out' in model.output_size:
        identity_output = torch.sigmoid(pred[output_index])
        identity_labels = labels[:, datasets.DATA_LABEL_POSITION_MAPPING['identity_category'],
                                 0:model.output_size['identity_category_out']]
        idc_label_scores = defaultdict(list)
        for index in range(model.output_size['identity_category_out']):
            identity_output_label = identity_output[:, index]
            identity_labels_label = identity_labels[:, index]
            identity_labels_label = identity_labels_label.to(device)
            id_scores = eval_binary_pred_values(torch.squeeze(identity_output_label),
                                                torch.squeeze(identity_labels_label))
            p_tp, p_tn, p_fp, p_fn, n_tp, n_tn, n_fp, n_fn = id_scores
            id_acc = (p_tp + p_tn) / len(identity_labels_label)
            idc_label_scores['acc'].append(id_acc.cpu())
            for score_name, score in calculate_prf_binary(id_scores).items():
                idc_label_scores[score_name].append(score.cpu())
            identity_labels_cpu = identity_labels_label.cpu().detach().numpy()
            identity_output_cpu = identity_output_label.cpu().detach().numpy()
            try:
                idc_label_scores['auc'].append(roc_auc_score(identity_labels_cpu, identity_output_cpu))
            except ValueError:
                # in cases where only one class is present in y_true. ROC AUC score is not defined in that case.
                pass
        # calc macro avg of individual label scores
        for score_name, label_scores in idc_label_scores.items():
            result_scores['idc avg ' + score_name] = np.mean(label_scores)
        if not idc_label_scores['auc']:
            result_scores['idc avg auc'] = 0.
        output_index += 1
    elif 'identity_term_out' in model.output_size:
        # infer idc pred from idt pred
        identity_output = torch.sigmoid(pred[output_index])
        # map these vectors of 24 idt labels each to vectors of 5 idc labels
        category_pred = []
        for instance_pred in identity_output:
            instance_cat_pred = []
            for vids in process_data.IDENTITY_CATEGORIES_ANNOTATIONS_MAPPING:
                if any([instance_pred[vid] >= 0.5 for vid in vids]):
                    instance_cat_pred.append(1.)
                else:
                    instance_cat_pred.append(0.)
            category_pred.append(instance_cat_pred)
        identity_output = torch.FloatTensor(category_pred)
        identity_output = identity_output.to(device)
        
        identity_labels = labels[:, datasets.DATA_LABEL_POSITION_MAPPING['identity_category'], 0:5]
        idc_label_scores = defaultdict(list)
        for index in range(5):
            identity_output_label = identity_output[:, index]
            identity_labels_label = identity_labels[:, index]
            identity_labels_label = identity_labels_label.to(device)
            id_scores = eval_binary_pred_values(torch.squeeze(identity_output_label),
                                                torch.squeeze(identity_labels_label))
            p_tp, p_tn, p_fp, p_fn, n_tp, n_tn, n_fp, n_fn = id_scores
            id_acc = (p_tp + p_tn) / len(identity_labels_label)
            idc_label_scores['acc'].append(id_acc.cpu())
            for score_name, score in calculate_prf_binary(id_scores).items():
                idc_label_scores[score_name].append(score.cpu())
            identity_labels_cpu = identity_labels_label.cpu().detach().numpy()
            identity_output_cpu = identity_output_label.cpu().detach().numpy()
            try:
                idc_label_scores['auc'].append(roc_auc_score(identity_labels_cpu, identity_output_cpu))
            except ValueError:
                # in cases where only one class is present in y_true. ROC AUC score is not defined in that case.
                pass
        # calc macro avg of individual label scores
        for score_name, label_scores in idc_label_scores.items():
            result_scores['idc avg ' + score_name] = np.mean(label_scores)
        if not idc_label_scores['auc']:
            result_scores['idc avg auc'] = 0.

    if 'identity_term_out' in model.output_size:
        identity_output = torch.sigmoid(pred[output_index])
        identity_labels = labels[:, datasets.DATA_LABEL_POSITION_MAPPING['identity_term'],
                                 0:model.output_size['identity_term_out']]
        idt_label_scores = defaultdict(list)
        for index in range(model.output_size['identity_term_out']):
            identity_output_label = identity_output[:, index]
            identity_labels_label = identity_labels[:, index]
            identity_labels_label = identity_labels_label.to(device)
            id_scores = eval_binary_pred_values(torch.squeeze(identity_output_label),
                                                torch.squeeze(identity_labels_label))
            p_tp, p_tn, p_fp, p_fn, n_tp, n_tn, n_fp, n_fn = id_scores
            id_acc = (p_tp + p_tn) / len(identity_labels_label)
            idt_label_scores['acc'].append(id_acc.cpu())
            for score_name, score in calculate_prf_binary(id_scores).items():
                idt_label_scores[score_name].append(score.cpu())
            identity_labels_cpu = identity_labels_label.cpu().detach().numpy()
            identity_output_cpu = identity_output_label.cpu().detach().numpy()
            try:
                idt_label_scores['auc'].append(roc_auc_score(identity_labels_cpu, identity_output_cpu))
            except ValueError:
                # in cases where only one class is present in y_true. ROC AUC score is not defined in that case.
                pass
        # calc macro avg of individual label scores
        for score_name, label_scores in idt_label_scores.items():
            result_scores['idt avg ' + score_name] = np.mean(label_scores)
        if not idt_label_scores['auc']:
            result_scores['idt avg auc'] = 0.
        output_index += 1

    return result_scores


def compute_eval_scores(y_pred, y_true, scores):
    scores.append((y_true * y_pred).sum())
    scores.append(((~ y_true) * (~ y_pred)).sum())
    scores.append(((~ y_true) * y_pred).sum())
    scores.append((y_true * (~ y_pred)).sum())


def eval_binary_pred_values(prediction, labels):
    # returns: p_tp, p_tn, p_fp, p_fn, n_tp, n_tn, n_fp, n_fn
    scores = []
    for label_class in (1, 0):
        y_pred = torch.round(prediction) == label_class
        y_true = labels == label_class
        compute_eval_scores(y_pred, y_true, scores)
    return scores


def eval_multi_class_pred_values(prediction, labels, label_set=(1, 0)):
    scores = []
    for label_class in label_set:
        y_pred = torch.argmax(prediction, dim=1) == label_class
        y_true = labels == label_class
        compute_eval_scores(y_pred, y_true, scores)
    return scores


def eval_pred_auc(prediction, labels, label_subgroups=None, label_classes=None):
    # label_subgroups contains a subgroup id for each label
    # label_classes are the different subgroup ids
    auc_scores = []
    try:
        auc_scores.append(roc_auc_score(labels, prediction))
    except ValueError:
        auc_scores.append(np.nan)

    if label_classes is not None:
        for label_class in label_classes:
            subgroup_prediction = []
            subgroup_labels = []
            for pred, label, label_subgroup in zip(prediction, labels, label_subgroups):
                if label_subgroup == label_class:
                    subgroup_prediction.append(pred)
                    subgroup_labels.append(label)
            try:
                auc_scores.append(roc_auc_score(subgroup_labels, subgroup_prediction))
            except ValueError:
                auc_scores.append(np.nan)
    return auc_scores
