import numpy as np
from tqdm import tqdm
import json
import torch
from urllib.error import HTTPError as UHTTPError
from requests.exceptions import HTTPError
from time import sleep
import csv
import os


IDENTITY_ANNOTATIONS = ("asian", "atheist", "bisexual", "black", "buddhist", "christian", "female", "heterosexual",
                        "hindu", "homosexual_gay_or_lesbian", "intellectual_or_learning_disability", "jewish", "latino",
                        "male", "muslim", "other_disability", "other_gender", "other_race_or_ethnicity",
                        "other_religion", "other_sexual_orientation", "physical_disability",
                        "psychiatric_or_mental_illness", "transgender", "white")
IDENTITY_CATEGORIES = ("disability",
                       # ("intellectual_or_learning_disability", "other_disability", "physical_disability",
                       #  "psychiatric_or_mental_illness")
                       "gender",
                       # ("female", "male", "other_gender", "transgender")
                       "race_or_ethnicity",
                       # ("asian", "black", "latino", "other_race_or_ethnicity", "white")
                       "religion",
                       # ("atheist", "buddhist", "christian", "hindu", "jewish", "muslim", "other_religion")
                       "sexual_orientation",
                       # ("homosexual_gay_or_lesbian", "bisexual", "heterosexual", "other_sexual_orientation")
                       )
# IDENTITY_ANNOTATIONS_CATEGORIES_MAPPING = (2, 3, 4, 2, 3, 3, 1, 4, 2, 4, 0, 3, 2, 1, 3, 0, 1, 2, 3, 4, 0, 0, 1, 2)
IDENTITY_CATEGORIES_ANNOTATIONS_MAPPING = ((10, 15, 20, 21),
                                           (6, 13, 16, 22),
                                           (0, 3, 12, 17, 23),
                                           (1, 4, 5, 8, 11, 14, 18),
                                           (2, 7, 9, 19))

TOX_LABEL_SET = {False: 0, True: 1}
IDENTITY_LABEL_SET = {False: 0, True: 1}


def read_prepared_dataset(path):
    with open(path) as json_file:
        data = json.loads(json.load(json_file))
    return data[0], data[1]


def prepare_gk_data(path):
    instances = []
    labels = []
    with open(path) as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter='\t', quotechar='"')
        for row in csv_reader:
            instances.append(row["text"])
            labels.append(row["HOF"] == 'Hateful')
    return instances, labels


def prepare_stormfront_data(path):
    annotations = {}
    with open(path + 'annotations_metadata.csv') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            annotations[row["file_id"]] = row["label"] == 'hate'
    instances = []
    labels = []
    for file in os.listdir(path + 'sampled_test/'):
        text = open(os.path.join(path + 'sampled_test/', file)).read()
        instances.append(text)
        labels.append(annotations[file[:-4]])
    return instances, labels


def prepare_olid_data(path):
    annotations = {}
    with open(path + 'labels-levela.csv') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            annotations[row[0]] = row[1] == 'OFF'
    instances = []
    labels = []
    with open(path + 'testset-levela.tsv') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter='\t', quotechar='"')
        for row in csv_reader:
            instances.append(row['tweet'])
            labels.append(annotations[row['id']])
    return instances, labels


def prepare_asianprejudice_data(path):
    instances = []
    labels = []
    with open(path) as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter='\t', quotechar='"')
        for row in csv_reader:
            instances.append(row['text'])
            labels.append(row['primary.actual'] == 'entity_directed_hostility')
    return instances, labels


def prepare_davidson_data(path):
    instances = []
    labels = []
    with open(path) as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            instances.append(row['tweet'])
            labels.append(row['class'] in '01')  # HS + OL ({True: 20620, False: 4163})
            # labels.append(row['class'] == '0')  # only HS ({False: 23353, True: 1430})
    return instances, labels


def prepare_hasoc_data(path):
    annotations = {}
    with open(path + '1A_English_actual_labels.csv') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            annotations[row['id']] = row['label'] == 'HOF'
    instances = []
    labels = []
    with open(path + 'en_Hasoc2021_test_task1.csv') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            instances.append(row['text'])
            labels.append(annotations[row['_id']])
    # labels: {True: 798, False: 483}
    return instances, labels


def prepare_sexist_data(path):
    instances = []
    labels = []
    with open(path) as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            instances.append(row['text'])
            labels.append(row['sexist'] == 'True')
    # labels: {False: 11822, True: 1809}
    return instances, labels


def prepare_sexist_data_tox(path):
    instances = []
    labels = []
    with open(path) as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            instances.append(row['text'])
            labels.append(float(row['toxicity']) >= 0.5)
    # labels: {False: 10702, True: 2929}
    return instances, labels


def prepare_ethos_data(path):
    instances = []
    labels = []
    with open(path) as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=';', quotechar='"')
        for row in csv_reader:
            instances.append(row['comment'])
            labels.append(float(row['isHate']) >= 0.5)
    # labels: {False: 565, True: 433}
    return instances, labels


def prepare_hatecheck_ds(project_path):
    path = project_path + '/data/hatecheck-data-main/test_suite_cases.csv'
    instances = []
    labels = []
    with open(path) as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            instances.append(row['test_case'])
            labels.append(row['label_gold'] == 'hateful')
    # labels: {True: 2563, False: 1165}
    with open(project_path + '/data/hatecheck_ds.json', 'w') as outfile:
        json.dump(json.dumps((instances, labels)), outfile)


def prepare_other_data(project_path):
    # result: list of tuples (instance (str), label (bool))

    # ### data with probably different identity groups: ###

    # - Grimminger/Klinger data (Biden, Trump) https://aclanthology.org/2021.wassa-1.18/
    #   https://www.ims.uni-stuttgart.de/forschung/ressourcen/korpora/stance-hof/
    data_path = project_path + '/data/GrimmingerKlingerWASSA2021/test.tsv'
    instances, labels = prepare_gk_data(data_path)
    with open(project_path + '/data/gk_ds.json', 'w') as outfile:
        json.dump(json.dumps((instances, labels)), outfile)

    # - text extracted from Stormfront, a white supremacist forum: https://github.com/Vicomtech/hate-speech-dataset
    data_path = project_path + '/data/hate-speech-dataset-master/'
    instances, labels = prepare_stormfront_data(data_path)
    with open(project_path + '/data/stormfront_ds.json', 'w') as outfile:
        json.dump(json.dumps((instances, labels)), outfile)

    # - OLID (mainly political keywords) https://aclanthology.org/N19-1144.pdf https://github.com/idontflow/OLID
    data_path = project_path + '/data/OLID/'
    instances, labels = prepare_olid_data(data_path)
    with open(project_path + '/data/olid_ds.json', 'w') as outfile:
        json.dump(json.dumps((instances, labels)), outfile)

    # - asian prejudice https://aclanthology.org/2020.alw-1.19/ https://zenodo.org/record/3816667
    data_path =\
        project_path + '/data/asianprejudice/hs_AsianPrejudice_40kdataset_cleaned_anonymized.tsv'
    instances, labels = prepare_asianprejudice_data(data_path)
    with open(project_path + '/data/asianprejudice_ds.json', 'w') as outfile:
        json.dump(json.dumps((instances, labels)), outfile)

    # ### datasets with probably similar identity groups: ###

    # - davidson https://github.com/t-davidson/hate-speech-and-offensive-language
    data_path = project_path + '/data/davidson/labeled_data.csv'
    instances, labels = prepare_davidson_data(data_path)
    with open(project_path + '/data/davidson_ds.json', 'w') as outfile:
        json.dump(json.dumps((instances, labels)), outfile)

    # - HASOC 2021 Test
    data_path = project_path + '/data/HASOC2021/'
    instances, labels = prepare_hasoc_data(data_path)
    with open(project_path + '/data/hasoc_ds.json', 'w') as outfile:
        json.dump(json.dumps((instances, labels)), outfile)

    # - sexist https://search.gesis.org/research_data/SDN-10.7802-2251
    data_path = project_path + '/data/sexist/sexism_data.csv'
    instances, labels = prepare_sexist_data(data_path)
    with open(project_path + '/data/sexist_ds.json', 'w') as outfile:
        json.dump(json.dumps((instances, labels)), outfile)

    # - sexist (with automatic tox anno) https://search.gesis.org/research_data/SDN-10.7802-2251
    data_path = project_path + '/data/sexist/sexism_data.csv'
    instances, labels = prepare_sexist_data_tox(data_path)
    with open(project_path + '/data/sexist-tox_ds.json', 'w') as outfile:
        json.dump(json.dumps((instances, labels)), outfile)

    # - ETHOS: an Online Hate Speech Detection Dataset (Binary)
    #   Link to publication: https://arxiv.org/pdf/2006.08328.pdf
    #   Link to data: https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset;
    #   Details of task: Gender, Race, National Origin, Disability, Religion, Sexual Orientation
    data_path = project_path + '/data/ETHOS/Ethos_Dataset_Binary.csv'
    instances, labels = prepare_ethos_data(data_path)
    with open(project_path + '/data/ethos_ds.json', 'w') as outfile:
        json.dump(json.dumps((instances, labels)), outfile)


def convert_texts(texts, max_seq_length, tokenizer):
    max_seq_length -= 2
    all_tokens = []
    for text in tqdm(texts):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)


class ModelTokenizer(object):

    def __init__(self, model_name):
        tokenizer_loaded = False
        while not tokenizer_loaded:
            try:
                self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)
                tokenizer_loaded = True
            except (HTTPError, ValueError, UHTTPError, OSError):
                print('Error when loading tokenizer, sleeping for ten minutes and trying again.')
                sleep(10*60)
