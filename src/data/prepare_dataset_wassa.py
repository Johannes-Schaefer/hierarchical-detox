import csv
import json
import random
from collections import defaultdict


IDENTITY_ANNOTATIONS = ["asian",
                        "atheist",
                        "bisexual",
                        "black",
                        "buddhist",
                        "christian",
                        "female",
                        "heterosexual",
                        "hindu",
                        "homosexual_gay_or_lesbian",
                        "intellectual_or_learning_disability",
                        "jewish",
                        "latino",
                        "male",
                        "muslim",
                        "other_disability",
                        "other_gender",
                        "other_race_or_ethnicity",
                        "other_religion",
                        "other_sexual_orientation",
                        "physical_disability",
                        "psychiatric_or_mental_illness",
                        "transgender",
                        "white"]
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
IDENTITY_CATEGORIES_ANNOTATIONS_MAPPING = ((10, 15, 20, 21),
                                           (6, 13, 16, 22),
                                           (0, 3, 12, 17, 23),
                                           (1, 4, 5, 8, 11, 14, 18),
                                           (2, 7, 9, 19))
SEED = 4422


def filter_instances(ds, train_data=False):
    print(f'Processing instances from file {ds}.')
    # ignore instances which are not annotated wrt IdTs
    instances = {}
    with open(ds) as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        total_rows = 0
        for row in csv_reader:
            total_rows += 1
            if int(row["identity_annotator_count"]) > 0:
                instance_id = int(row["id"])
                if train_data:
                    instance_toxicity = float(row["target"])
                else:
                    instance_toxicity = float(row["toxicity"])
                instance_text = row["comment_text"]
                instance_identity_annotations = []
                for identity in IDENTITY_ANNOTATIONS:
                    instance_identity_annotation = float(row[identity])
                    instance_identity_annotations.append(instance_identity_annotation)
                instances[instance_id] = [instance_text, instance_toxicity, instance_identity_annotations]
    print(f'Found {len(instances)} instances (with tox and idt annotation) out of {total_rows} instances.')
    return instances


def read_filtered_jigsaw_data(project_path):
    with open(project_path + '/data/train_filtered.json') as json_file:
        jigsaw_data = json.loads(json.load(json_file))
        print(len(jigsaw_data))
    return jigsaw_data


def prepare_jigsaw_data(jigsaw_data, idt_hierarchical=False, filter_idts=()):
    instances = []
    labels = []
    idt_counts = defaultdict(int)
    for entry in jigsaw_data.values():
        # instance body
        text = entry[0]
        # instance annotations/labels
        tox = entry[1] >= 0.5
        idt_mentioned = any([v >= 0.5 for v in entry[2]])
        if idt_hierarchical:
            if idt_mentioned:
                # idt
                idt = [int(v >= 0.5) for v in entry[2]]
                # filter idts: skip instance if mentioned idt in filter_idts
                if any([IDENTITY_ANNOTATIONS[index] in filter_idts for index, val in enumerate(idt) if val]):
                    continue
                for idt_str in [IDENTITY_ANNOTATIONS[index] for index, val in enumerate(idt) if val]:
                    idt_counts[idt_str] += 1
                # idt_category
                idt_categories = [int(any([idt[index] for index in v]))
                                  for v in IDENTITY_CATEGORIES_ANNOTATIONS_MAPPING]
                labels.append((tox, idt_mentioned, idt_categories, idt))
            else:
                labels.append((tox, idt_mentioned, [0]*len(IDENTITY_CATEGORIES), [0]*len(IDENTITY_ANNOTATIONS)))
        else:
            labels.append((tox, idt_mentioned))
        instances.append(text)
    return instances, labels


def prepare_wassa_ds(jigsaw_data, project_path, filter_idts=()):
    instances, labels = prepare_jigsaw_data(jigsaw_data, idt_hierarchical=True, filter_idts=filter_idts)
    combined_data = list(zip(instances, labels))
    random.seed(SEED)
    random.shuffle(combined_data)
    instances, labels = zip(*combined_data)
    if filter_idts:
        ds_filename = project_path + '/data/full_jigsaw_ds_religion-filtered.json'
    else:
        ds_filename = project_path + '/data/full_jigsaw_ds.json'
    print(f'Writing {len(instances)} instances and labels to file {ds_filename}.')
    with open(ds_filename, 'w') as outfile:
        json.dump(json.dumps((instances, labels)), outfile)


def prepare_wassa_test_ds(test_private_data, test_public_data, project_path):
    instances, labels = prepare_jigsaw_data(test_private_data, idt_hierarchical=True)
    more_instances, more_labels = prepare_jigsaw_data(test_public_data, idt_hierarchical=True)
    ds_filename = project_path + '/data/combined_test_ds.json'
    print(f'Writing {len(instances) + len(more_instances)} instances and labels to file {ds_filename}.')
    with open(ds_filename, 'w') as outfile:
        json.dump(json.dumps((instances + more_instances, labels + more_labels)), outfile)


def prepare_all_datasets(project_path):
    # Requires CivilComments data files from the kaggle challenge:
    # - files can be downloaded from https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data,
    # - used files: 'train.csv', 'test_private_expanded.csv' and 'test_public_expanded.csv'.
    jigsaw_train_data_path = project_path + '/data/train.csv'
    jigsaw_test_pr_data_path = project_path + '/data/test_private_expanded.csv'
    jigsaw_test_pu_data_path = project_path + '/data/test_public_expanded.csv'

    # prepare train (dev) dataset
    dev_instances = filter_instances(jigsaw_train_data_path, train_data=True)

    tox_idt = 0
    tox_noidt = 0
    notox_idt = 0
    notox_noidt = 0
    for inst_id in dev_instances:
        tox = dev_instances[inst_id][1] >= 0.5
        idt = any([v >= 0.5 for v in dev_instances[inst_id][2]])
        if tox:
            if idt:
                tox_idt += 1
            else:
                tox_noidt += 1
        else:
            if idt:
                notox_idt += 1
            else:
                notox_noidt += 1

    print('Toxicity and idt correlation:')
    print(f'\tToxic instances with itd: {tox_idt}')
    print(f'\tToxic instances without itd: {tox_noidt}')
    print(f'\tNon-toxic instances with itd: {notox_idt}')
    print(f'\tNon-toxic instances without itd: {notox_noidt}')

    prepare_wassa_ds(dev_instances, project_path)

    tpr = filter_instances(jigsaw_test_pr_data_path)
    tpu = filter_instances(jigsaw_test_pu_data_path)
    prepare_wassa_test_ds(tpr, tpu, project_path)
