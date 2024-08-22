from random import shuffle
import torch
import numpy as np
from data import process_data


DATA_LABEL_POSITION_MAPPING = {'toxicity': 0, 'identity_present': 1,  'identity_category': 2, 'identity_term': 3}


def prepare_train_val_test_ds(ds_path, ratio=(.8, .1, .1), fixed_train_size=None, fixed_val_size=None,
                              shuffle_data=False):
    if sum(ratio) != 1:
        raise ValueError('Error: Sum of ratio is not 1.')
    instances, labels = process_data.read_prepared_dataset(ds_path)
    if shuffle_data:
        combined_data = list(zip(instances, labels))
        shuffle(combined_data)
        instances, labels = zip(*combined_data)
    if fixed_train_size is not None and fixed_val_size is not None:
        train_instances = instances[:fixed_train_size]
        train_labels = labels[:fixed_train_size]
        val_instances = instances[fixed_train_size:fixed_train_size+fixed_val_size]
        val_labels = labels[fixed_train_size:fixed_train_size+fixed_val_size]
        test_instances = instances[fixed_train_size+fixed_val_size:]
        test_labels = labels[fixed_train_size+fixed_val_size:]
    else:
        train_instances = instances[:int(round(len(instances)*ratio[0]))]
        train_labels = labels[:int(round(len(instances)*ratio[0]))]
        val_instances = instances[int(round(len(instances)*ratio[0])):int(round(len(instances)*(ratio[0]+ratio[1])))]
        val_labels = labels[int(round(len(instances)*ratio[0])):int(round(len(instances)*(ratio[0]+ratio[1])))]
        test_instances = instances[int(round(len(instances)*(ratio[0]+ratio[1]))):]
        test_labels = labels[int(round(len(instances)*(ratio[0]+ratio[1]))):]
    return train_instances, train_labels, val_instances, val_labels, test_instances, test_labels


def load_test_data(ds_path):
    test_instances, test_labels = process_data.read_prepared_dataset(ds_path)
    return test_instances, test_labels


class ToxDataset(torch.utils.data.Dataset):

    def __init__(self, instances, labels, model_tokenizer, tox_label_set, max_len=236, identity_label_set=None,
                 filter_tokens_ids=()):
        if identity_label_set is not None:
            try:
                self.labels = np.array([np.array([np.array([tox_label_set[label[0]]] + [0]*(len(label[3])-1)),
                                                  np.array([identity_label_set[label[1]]] + [0]*(len(label[3])-1)),
                                                  np.array(label[2] + [0]*(len(label[3]) - len(label[2]))),
                                                  np.array(label[3])])
                                        for label in labels])
            except TypeError:
                # in cases when only tox label is available in data
                self.labels = np.array([np.array([np.array([tox_label_set[label]] + [0]*23),
                                                  np.array([0]*24),
                                                  np.array([0]*24),
                                                  np.array([0]*24)])
                                        for label in labels])
        else:
            self.labels = [np.array(tox_label_set[label[0]]) for label in labels]
        print('tokenizing %d instances' % len(instances))
        self.texts = []
        for text in instances:
            tokenized_text = model_tokenizer.tokenizer(text, padding='max_length', max_length=max_len, truncation=True,
                                                       return_tensors="pt")
            if filter_tokens_ids:
                tokenized_text['attention_mask'] = (
                    torch.tensor([[int(t and (t not in filter_tokens_ids)) for t in tokenized_text['input_ids'][0]]]))
            self.texts.append(tokenized_text)
        print('done.')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_texts = self.texts[idx]
        batch_labels = self.labels[idx]
        return batch_texts, batch_labels


def print_labels_ratios(labels):
    tox_labels = [annotation[0] for annotation in labels]
    print('tox ratio: %d/%d = %2.2f%%' % (sum(tox_labels), len(tox_labels), 100*sum(tox_labels)/float(len(tox_labels))))
    idt_mentioned_labels = [annotation[1] for annotation in labels]
    print('idt_mentioned ratio: %d/%d = %2.2f%%' % (sum(idt_mentioned_labels), len(idt_mentioned_labels),
                                                    100*sum(idt_mentioned_labels)/float(len(idt_mentioned_labels))))
