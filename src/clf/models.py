import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.autograd import Function
from data import process_data
from data import datasets
from clf import eval
from sklearn.utils import class_weight
from datetime import datetime
from collections import defaultdict
from urllib.error import HTTPError
from time import sleep
from lime.lime_text import LimeTextExplainer


SEED = 4422


class GradientReversal(Function):

    @staticmethod
    def forward(ctx, x, lambda_value):
        ctx.save_for_backward(x, lambda_value)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, lambda_value = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = - lambda_value * grad_output
        return grad_input, None


class ToxPredictorBERT(nn.Module):

    def __init__(self, input_size, hidden_size, dropout=0., classifier_toxicity=True,
                 classifier_identity_present=False, classifier_identity_category=False, classifier_identity_term=False,
                 classifier_toxicity_adversarial=False, classifier_identity_present_adversarial=False,
                 classifier_identity_category_adversarial=False, classifier_identity_term_adversarial=False,
                 adv_lambda=0.5,
                 bert_model_name='bert-base-uncased'):
        super(ToxPredictorBERT, self).__init__()
        self.input_size = input_size

        self.output_size = {}
        if classifier_toxicity:
            self.output_size['toxicity_out'] = 1
        if classifier_identity_present:
            self.output_size['identity_present_out'] = 1
        if classifier_identity_category:
            self.output_size['identity_category_out'] = len(process_data.IDENTITY_CATEGORIES)
        if classifier_identity_term:
            self.output_size['identity_term_out'] = len(process_data.IDENTITY_ANNOTATIONS)

        print('model output shape:', self.output_size)

        self.adversarial_classifiers = []
        if classifier_toxicity_adversarial:
            self.adversarial_classifiers.append('classifier_toxicity')
        if classifier_identity_present_adversarial:
            self.adversarial_classifiers.append('classifier_identity_present')
        if classifier_identity_category_adversarial:
            self.adversarial_classifiers.append('classifier_identity_category')
        if classifier_identity_term_adversarial:
            self.adversarial_classifiers.append('classifier_identity_term')

        print('model adv layers:', self.adversarial_classifiers)

        self.adv_lambda = adv_lambda

        if self.adversarial_classifiers:
            print('model adv lambda:', self.adv_lambda)

        self.bert_model_name = bert_model_name
        bert_model_loaded = False
        while not bert_model_loaded:
            try:
                self.bert = transformers.AutoModel.from_pretrained(bert_model_name)
                bert_model_loaded = True
            except (HTTPError, ValueError):
                print('Error when loading bert model, sleeping for 10 minutes and trying again.')
                sleep(60*10)

        self.dropout_layer = nn.Dropout(p=dropout)

        if classifier_toxicity:
            self.tox_linear = nn.Linear(hidden_size, self.output_size['toxicity_out'])
        if classifier_identity_present:
            self.identity_present_linear = nn.Linear(hidden_size, self.output_size['identity_present_out'])
        if classifier_identity_category:
            self.identity_category_linear = nn.Linear(hidden_size, self.output_size['identity_category_out'])
        if classifier_identity_term:
            self.identity_term_linear = nn.Linear(hidden_size, self.output_size['identity_term_out'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):

        _, encoding = self.bert(input_ids,
                                attention_mask=torch.squeeze(attention_mask),
                                token_type_ids=token_type_ids,
                                return_dict=False)
        encoding = self.dropout_layer(encoding)
        reverse_encoding = GradientReversal.apply(encoding, torch.tensor([self.adv_lambda], device=self.device))

        output_layers = []

        if 'toxicity_out' in self.output_size:
            if 'classifier_toxicity' in self.adversarial_classifiers:
                out_tox = self.tox_linear(reverse_encoding)
            else:
                out_tox = self.tox_linear(encoding)
            output_layers.append(out_tox)

        if 'identity_present_out' in self.output_size:
            if 'classifier_identity_present' in self.adversarial_classifiers:
                out_identity_present = self.identity_present_linear(reverse_encoding)
            else:
                out_identity_present = self.identity_present_linear(encoding)
            output_layers.append(out_identity_present)

        if 'identity_category_out' in self.output_size:
            if 'classifier_identity_category' in self.adversarial_classifiers:
                out_identity_category = self.identity_category_linear(reverse_encoding)
            else:
                out_identity_category = self.identity_category_linear(encoding)
            output_layers.append(out_identity_category)

        if 'identity_term_out' in self.output_size:
            if 'classifier_identity_term' in self.adversarial_classifiers:
                out_identity_term = self.identity_term_linear(reverse_encoding)
            else:
                out_identity_term = self.identity_term_linear(encoding)
            output_layers.append(out_identity_term)

        return output_layers


class PredictorModel(object):
    # used for LIME explanations

    def __init__(self, model, device, tokenizer, max_len):
        self.model = model
        self.model.eval()
        self.device = device
        self.tokenizer = tokenizer
        self.max_len = max_len

    def predict(self, text):
        tokenized_text = [self.tokenizer.tokenizer(text, padding='max_length', max_length=self.max_len,
                                                   truncation=True, return_tensors="pt")]
        dataset = torch.utils.data.DataLoader(tokenized_text)
        for instance in dataset:
            mask = instance['attention_mask'].to(self.device)
            input_id = instance['input_ids'].squeeze(1).to(self.device)
            output = self.model(input_id, mask)
            return output

    def predict_proba(self, texts):
        # LIME classifier_fn – classifier prediction probability function, which takes a list of d strings and outputs a
        #  (d, k) numpy array with prediction probabilities, where k is the number of classes.
        tokenized_texts = [self.tokenizer.tokenizer(text, padding='max_length', max_length=self.max_len,
                                                    truncation=True, return_tensors="pt") for text in texts]
        dataset = torch.utils.data.DataLoader(tokenized_texts)
        outputs = []
        for instance in dataset:
            mask = instance['attention_mask'].to(self.device)
            input_id = instance['input_ids'].squeeze(1).to(self.device)
            output = self.model(input_id, mask)
            output = torch.sigmoid(output[0]).cpu().detach().numpy()[0]
            outputs.append(np.array([output[0], 1-output[0]]))
        return np.array(outputs)

    def predict_instance_proba(self, text):
        tokenized_text = self.tokenizer.tokenizer(text, padding='max_length', max_length=self.max_len, truncation=True,
                                                  return_tensors="pt")
        dataset = torch.utils.data.DataLoader([tokenized_text])
        outputs = []
        for instance in dataset:
            mask = instance['attention_mask'].to(self.device)
            input_id = instance['input_ids'].squeeze(1).to(self.device)
            output = self.model(input_id, mask)
            outputs.append(list(output[0].cpu().detach().numpy()))
        outputs = [list(outputs[0][0]), [1-float(outputs[0][0])]]
        return outputs


def apply_model(texts, model, device):
    mask = texts['attention_mask'].to(device)
    input_id = texts['input_ids'].squeeze(1).to(device)
    output = model(input_id, mask)
    return output


def write_log(filename, text, print_text=False):
    if print_text:
        print(text)
    if filename is not None:
        with open(filename, mode='a') as logfile:
            logfile.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ': ' + text + '\n')


def model_criteria(class_weights, model_output_sizes):
    criteria = {}
    for task_name, task_output_size in model_output_sizes.items():
        task_class_weights = class_weights.get(task_name, None) if class_weights is not None else None
        if task_output_size == 1:
            # binary classifiers
            criteria[task_name] = nn.BCEWithLogitsLoss(pos_weight=task_class_weights)
        elif task_name in ('toxicity_out', 'identity_present_out'):
            # multi-class classifiers (not multi-label)
            criteria[task_name] = nn.CrossEntropyLoss(weight=task_class_weights)
        else:
            # multi-label classifiers
            criteria[task_name] = nn.BCEWithLogitsLoss(pos_weight=task_class_weights)
    return criteria


def train(model, train_data, val_data, learning_rate, epochs, batch_size, project_path, class_weights=None,
          log_to_file=None, early_stopping=True, eval_intermediate=False, no_add_train_epoch=False):
    saved_model = None
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criteria = model_criteria(class_weights, model.output_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        for task_name, criterion in criteria.items():
            criteria[task_name] = criterion.cuda()

    write_log(log_to_file, 'Starting training')

    temp_model_path = None
    best_val_score = float('-inf')
    epochs_waited = 0
    best_epoch = 0
    for epoch_num in range(epochs):
        model.train()
        if temp_model_path is not None and model.adversarial_classifiers and model.adv_lambda > 0 and\
                not no_add_train_epoch:
            checkpoint = torch.load(temp_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        total_pred_train = []
        total_label_train = []

        total_loss_train = 0.
        total_task_losses_train = defaultdict(float)

        for train_input, train_label in train_dataloader:
            output = apply_model(train_input, model, device)

            # train_label is a tensor (8x4x24)
            # output is a list of two tensors [8x1, 8x5]
            if no_add_train_epoch or (not model.adversarial_classifiers) or model.adv_lambda <= 0\
                    or ((not early_stopping) and epoch_num < epochs - 1) or eval_intermediate:
                total_pred_train.append(output)
                total_label_train.append(train_label)

            batch_loss, batch_task_losses = eval.get_loss(model, output, train_label, device, criteria)
            if no_add_train_epoch or (not model.adversarial_classifiers) or model.adv_lambda <= 0\
                    or ((not early_stopping) and epoch_num < epochs - 1):
                # save loss if training is final
                total_loss_train += batch_loss.item()
                for task_name, batch_task_loss in batch_task_losses.items():
                    total_task_losses_train[task_name] += batch_task_loss.item()

            model.zero_grad()  # makes sure all grads are zero
            batch_loss.backward()
            optimizer.step()

        # total_pred_train: list: [num_batches x list: [num_tasks x Tensor: (batch_size x task_output_size)]]
        total_pred_train_lists = []
        for index in range(len(model.output_size)):
            total_pred_train_lists.append(torch.cat([batch_pred[index] for batch_pred in total_pred_train], dim=0))
        # total_pred_train_lists: list: [num_tasks x Tensor: (train_size x task_output_size)]]

        # total_label_train: list: [num_batches x Tensor: (batch_size x 4 x 24)]
        total_label_train = torch.cat(total_label_train, dim=0)
        # total_label_train: Tensor: (train_size x 4 x 24)
        train_result_scores = eval.eval_pred(model, total_pred_train_lists, total_label_train, device)

        model.eval()

        total_pred_val = []
        total_label_val = []

        total_loss_val = 0.
        total_task_losses_val = defaultdict(float)

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                output = apply_model(val_input, model, device)
                batch_loss, batch_task_losses = eval.get_loss(model, output, val_label, device, criteria)
                total_loss_val += batch_loss.item()
                for task_name, batch_task_loss in batch_task_losses.items():
                    total_task_losses_val[task_name] += batch_task_loss.item()

                total_pred_val.append(output)
                total_label_val.append(val_label)

        total_pred_val_lists = []
        for index in range(len(model.output_size)):
            total_pred_val_lists.append(torch.cat([batch_pred[index] for batch_pred in total_pred_val], dim=0))
        total_label_val = torch.cat(total_label_val, dim=0)
        val_result_scores = eval.eval_pred(model, total_pred_val_lists, total_label_val, device)

        out = f'Epochs: {epoch_num + 1} | Train Loss: {batch_size * total_loss_train / len(train_data.texts): .5f} '
        for task_name, task_batch_losses_sum in total_task_losses_train.items():
            out += f'| Train {task_name} loss: {batch_size * task_batch_losses_sum / len(train_data.texts): .5f} '
        for score_name, score in train_result_scores.items():
            out += f'| Train {score_name}: {score: .3f} '

        out += f'| Val loss: {batch_size * total_loss_val / len(val_data.texts): .5f} '
        for task_name, task_batch_losses_sum in total_task_losses_val.items():
            out += f'| Val {task_name}  loss: {batch_size * task_batch_losses_sum / len(val_data.texts): .5f} '
        for score_name, score in val_result_scores.items():
            out += f'| Val {score_name}: {score: .3f} '

        write_log(log_to_file, out, print_text=True)

        if early_stopping:
            # early stopping check
            patience = 3
            min_delta = 0.005

            val_score = 0.
            val_score_dn = 0
            for task_name in model.output_size:
                if task_name == 'toxicity_out' and 'classifier_toxicity' not in model.adversarial_classifiers:
                    val_score += val_result_scores['tox pos class f']
                    val_score_dn += 1
                if task_name == 'identity_term_out':
                    if 'classifier_identity_term' in model.adversarial_classifiers:
                        if model.adv_lambda > 0:
                            val_score += - (1/3) * val_result_scores['idt avg pos class f']
                            val_score_dn += - 1/3
                    else:
                        val_score += 0.5 * val_result_scores['idt avg pos class f']
                        val_score_dn += 0.5
            val_score = val_score / val_score_dn

            model_path = project_path + '/models/model_' + log_to_file.split('/')[-1].split('.')[0] + '.pt'
            saved_model = model_path

            if val_score < (best_val_score + min_delta):
                write_log(log_to_file,
                          f'Early stopping check: monitored value did not improve substantially '
                          f'{val_score} < {best_val_score+min_delta}',
                          print_text=True)
                if epochs_waited >= patience:
                    es_string = f'Early stopping (patience={patience} and min_delta={min_delta: .5f}), ' \
                                f'reloaded model from epoch {epoch_num - patience}.'
                    write_log(log_to_file, es_string, print_text=True)
                    # load model weights
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    break
                elif epoch_num + 1 == epochs:
                    es_string = f'Max epoch reached without improvement in last epoch, ' +\
                                f'reloaded model from epoch {best_epoch}.'
                    write_log(log_to_file, es_string, print_text=True)
                    # load model weights
                    model.load_state_dict(torch.load(model_path, map_location=device))
                else:
                    # continue training
                    epochs_waited += 1
            else:
                write_log(log_to_file,
                          f'Early stopping check: monitored value improved {val_score} > {best_val_score+min_delta}'
                          f', saving model from epoch {epoch_num + 1}', print_text=True)
                epochs_waited = 0
                best_val_score = val_score
                # save model weights
                torch.save(model.state_dict(), model_path)
                best_epoch = epoch_num + 1
    return saved_model, criteria


def evaluate(model, test_data, batch_size, log_to_file=None, criteria=None, write_pred=False):
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    model.eval()
    total_pred_test = []
    total_label_test = []
    total_loss_test = 0.
    total_task_losses_test = defaultdict(float)
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            output = apply_model(test_input, model, device)

            if criteria is not None:
                batch_loss, batch_task_losses = eval.get_loss(model, output, test_label, device, criteria)
                total_loss_test += batch_loss.item()
                for task_name, batch_task_loss in batch_task_losses.items():
                    total_task_losses_test[task_name] += batch_task_loss.item()

            total_pred_test.append(output)
            total_label_test.append(test_label)

    total_pred_test_lists = []
    for index in range(len(model.output_size)):
        total_pred_test_lists.append(torch.cat([batch_pred[index] for batch_pred in total_pred_test], dim=0))
    total_label_test = torch.cat(total_label_test, dim=0)
    test_result_scores = eval.eval_pred(model, total_pred_test_lists, total_label_test, device)

    if write_pred:
        torch.save((model.output_size, total_pred_test_lists, total_label_test, device), write_pred + '-pred')

    out = f'Test loss: {batch_size * total_loss_test / len(test_data.texts): .5f} '
    for task_name, task_batch_losses_sum in total_task_losses_test.items():
        out += f'| Test {task_name} loss: {batch_size * task_batch_losses_sum / len(test_data.texts): .5f} '
    for score_name, score in test_result_scores.items():
        out += f'| Test {score_name}: {score: .4f} '
    write_log(log_to_file, out, print_text=True)


def calculate_class_weights(use_class_weights, tox_label_set, identity_label_set, train_labels, model, logfile):
    if not use_class_weights:
        return None

    class_weights = {}
    for task_name, task_output_size in model.output_size.items():

        if task_name in ('identity_category_out', 'identity_term_out'):
            label_position = datasets.DATA_LABEL_POSITION_MAPPING[task_name.replace('_out', '')]
            all_labels = [label[label_position] for label in train_labels]
            label_labels = [[lab[position] for lab in all_labels] for position in range(len(all_labels[0]))]
            all_weights = []
            for labels in label_labels:
                y = torch.as_tensor(labels)
                unique_labels = np.unique(y)
                task_class_weights = class_weight.compute_class_weight('balanced', classes=unique_labels, y=y.numpy())
                task_class_weights = torch.tensor(task_class_weights, dtype=torch.float)
                try:
                    all_weights.append(task_class_weights[1] / task_class_weights[0])
                except IndexError:
                    # this case happens when labels have a frequency of 0
                    all_weights.append(torch.tensor(1.0, dtype=torch.float))
            task_class_weights = torch.tensor(all_weights)
            print(f'using {task_name} pos weights: {str(task_class_weights)}')
        else:
            label_position = datasets.DATA_LABEL_POSITION_MAPPING[task_name.replace('_out', '')]
            if label_position == 0:
                y = torch.as_tensor([tox_label_set[label[label_position]] for label in train_labels])
            elif label_position == 1:
                y = torch.as_tensor([identity_label_set[label[label_position]] for label in train_labels])
            else:
                task_labels = []
                for labels in train_labels:
                    for index, label_val in enumerate(labels[label_position]):
                        if label_val:
                            task_labels.append(index)
                y = torch.as_tensor(task_labels)

            unique_labels = np.unique(y)
            task_class_weights = class_weight.compute_class_weight('balanced', classes=unique_labels, y=y.numpy())
            task_class_weights = torch.tensor(task_class_weights, dtype=torch.float)

            if task_output_size == 1:
                task_class_weights = task_class_weights[1] / task_class_weights[0]
                print(f'using {task_name} pos_weight: {task_class_weights: .3f}')
            else:
                if task_output_size != len(unique_labels):
                    write_log(logfile, f'{task_name} class weights: labels not for all categories found (only for '
                                       f'{str(unique_labels)} ), adding max weight for unseen labels', print_text=True)
                    all_weights = []
                    calculated_label_weight_position = 0
                    for label in range(task_output_size):
                        if label not in unique_labels:
                            all_weights.append(max(task_class_weights))
                        else:
                            all_weights.append(task_class_weights[calculated_label_weight_position])
                            calculated_label_weight_position += 1
                    task_class_weights = torch.tensor(all_weights)
                print(f'using {task_name} class weights: {str(task_class_weights)}')
        class_weights[task_name] = task_class_weights
    write_log(logfile, f'class_weights: {str(class_weights)}')
    return class_weights


def prepare_nn_model(data_path, logfile, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.model_name
    input_size = args.input_size
    hidden_size = args.hidden_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    dropout = args.dropout
    learning_rate = args.learning_rate
    use_class_weights = args.use_class_weights
    early_stopping = not args.no_early_stopping

    write_log(logfile, 'data: ' + str(data_path))

    model_tokenizer = process_data.ModelTokenizer(model_name)
    try:
        filter_tokens = args.filter_tokens
        filter_tokens_ids = model_tokenizer.tokenizer.convert_tokens_to_ids(filter_tokens)
    except AttributeError:
        filter_tokens_ids = ()

    model = ToxPredictorBERT(input_size, hidden_size, dropout=dropout,
                             classifier_toxicity=args.classifier_toxicity,
                             classifier_identity_present=args.classifier_identity_present,
                             classifier_identity_category=args.classifier_identity_category,
                             classifier_identity_term=args.classifier_identity_term,
                             classifier_toxicity_adversarial=args.adversarial_classifier_toxicity,
                             classifier_identity_present_adversarial=args.adversarial_classifier_identity_present,
                             classifier_identity_category_adversarial=args.adversarial_classifier_identity_category,
                             classifier_identity_term_adversarial=args.adversarial_classifier_identity_term,
                             adv_lambda=args.adv_lambda,
                             bert_model_name=model_name
                             )
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
    model.to(device)

    tox_label_set = {False: 0, True: 1}
    identity_label_set = {False: 0, True: 1}

    return (data_path, args.fixed_train_size, args.fixed_val_size, identity_label_set, filter_tokens_ids,
            use_class_weights, model, learning_rate, num_epochs, batch_size, early_stopping, args.save_model,
            model_tokenizer, tox_label_set, input_size)


def do_train_test_run(logfile, data_path, fixed_train_size, fixed_val_size, model_tokenizer, tox_label_set, input_size,
                      identity_label_set, filter_tokens_ids, use_class_weights, model, learning_rate, num_epochs,
                      batch_size, project_path, early_stopping, save_model, write_pred):
    write_log(logfile, 'Starting train + val + test run')
    train_instances, train_labels, val_instances, val_labels, test_instances, test_labels = \
        datasets.prepare_train_val_test_ds(data_path, fixed_train_size=fixed_train_size,
                                           fixed_val_size=fixed_val_size)
    train_dataset = datasets.ToxDataset(train_instances, train_labels, model_tokenizer, tox_label_set,
                                        max_len=input_size, identity_label_set=identity_label_set,
                                        filter_tokens_ids=filter_tokens_ids)
    class_weights = calculate_class_weights(use_class_weights, tox_label_set, identity_label_set, train_labels,
                                            model, logfile)
    val_dataset = datasets.ToxDataset(val_instances, val_labels, model_tokenizer, tox_label_set, max_len=input_size,
                                      identity_label_set=identity_label_set, filter_tokens_ids=filter_tokens_ids)

    # train
    saved_model, criteria = train(model, train_dataset, val_dataset, learning_rate, num_epochs, batch_size,
                                  project_path,  class_weights=class_weights, log_to_file=logfile,
                                  early_stopping=early_stopping)

    if save_model:
        model_path = project_path + '/models/model_' + logfile.split('/')[-1].split('.')[0] + '.pt'
        torch.save(model.state_dict(), model_path)

    # test
    test_dataset = datasets.ToxDataset(test_instances, test_labels, model_tokenizer, tox_label_set,
                                       max_len=input_size, identity_label_set=identity_label_set,
                                       filter_tokens_ids=filter_tokens_ids)
    evaluate(model, test_dataset, batch_size, log_to_file=logfile, criteria=criteria, write_pred=write_pred)
    return saved_model


def do_test_only_run(logfile, data_path, model_tokenizer, tox_label_set, input_size, filter_tokens_ids,
                     identity_label_set, model, batch_size, write_pred, criteria=None):
    write_log(logfile, 'Starting prediction/eval only run')
    test_instances, test_labels = datasets.load_test_data(data_path)
    test_dataset = datasets.ToxDataset(test_instances, test_labels, model_tokenizer, tox_label_set,
                                       max_len=input_size, filter_tokens_ids=filter_tokens_ids,
                                       identity_label_set=identity_label_set)
    evaluate(model, test_dataset, batch_size, log_to_file=logfile, criteria=criteria, write_pred=write_pred)


def explain(texts, model_type, num_feats, args, project_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_tokenizer = process_data.ModelTokenizer(args.model_name)

    model = ToxPredictorBERT(args.input_size, args.hidden_size, dropout=args.dropout,
                             classifier_toxicity=args.classifier_toxicity,
                             classifier_identity_present=args.classifier_identity_present,
                             classifier_identity_category=args.classifier_identity_category,
                             classifier_identity_term=args.classifier_identity_term,
                             classifier_toxicity_adversarial=args.adversarial_classifier_toxicity,
                             classifier_identity_present_adversarial=args.adversarial_classifier_identity_present,
                             classifier_identity_category_adversarial=args.adversarial_classifier_identity_category,
                             classifier_identity_term_adversarial=args.adversarial_classifier_identity_term,
                             adv_lambda=args.adv_lambda,
                             bert_model_name=args.model_name
                             )
    model.load_state_dict(torch.load(args.load_model, map_location=device))
    model.to(device)

    p = PredictorModel(model, device, model_tokenizer, args.input_size)

    class_names = ['Toxic', 'Not-Toxic']
    explainer = LimeTextExplainer(class_names=class_names)

    for index, instance in enumerate(texts):
        print(instance)
        exp = explainer.explain_instance(instance, p.predict_proba, num_features=num_feats)
        exp.save_to_file(project_path + f'/logs/exp_{model_type}_selected-{index:02d}.html')
        torch.save(exp, project_path + f'/logs/exp_{model_type}_selected-{index:02d}.exp')


def predict_texts(texts, model_type, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_tokenizer = process_data.ModelTokenizer(args.model_name)
    model = ToxPredictorBERT(args.input_size, args.hidden_size, dropout=args.dropout,
                             classifier_toxicity=args.classifier_toxicity,
                             classifier_identity_present=args.classifier_identity_present,
                             classifier_identity_category=args.classifier_identity_category,
                             classifier_identity_term=args.classifier_identity_term,
                             classifier_toxicity_adversarial=args.adversarial_classifier_toxicity,
                             classifier_identity_present_adversarial=args.adversarial_classifier_identity_present,
                             classifier_identity_category_adversarial=args.adversarial_classifier_identity_category,
                             classifier_identity_term_adversarial=args.adversarial_classifier_identity_term,
                             adv_lambda=args.adv_lambda,
                             bert_model_name=args.model_name
                             )
    model.load_state_dict(torch.load(args.load_model, map_location=device))

    model.to(device)

    pred_model = PredictorModel(model, device, model_tokenizer, args.input_size)

    outputs = pred_model.predict_proba(texts)
    out = model_type
    out_preds = []
    for index, text in enumerate(texts):
        out_preds.append((text, str(f'{outputs[index][0]:.3f}')))
    predz = (out, out_preds)
    return predz
