
import csv
import json
from collections import defaultdict
from matplotlib import pyplot


IDENTITY_ANNOTATIONS = ["asian", "atheist", "bisexual", "black", "buddhist", "christian", "female", "heterosexual",
                        "hindu", "homosexual_gay_or_lesbian", "intellectual_or_learning_disability", "jewish", "latino",
                        "male", "muslim", "other_disability", "other_gender", "other_race_or_ethnicity",
                        "other_religion", "other_sexual_orientation", "physical_disability",
                        "psychiatric_or_mental_illness", "transgender", "white"]


def count_instances(project_path):
    instances = 0
    instances_with_toxicity_annotation = 0
    instances_with_identity_annotation = 0
    with open(project_path + '/data/train.csv') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            instances += 1
            if int(row["toxicity_annotator_count"]) > 0:
                instances_with_toxicity_annotation += 1
            if int(row["identity_annotator_count"]) > 0:
                instances_with_identity_annotation += 1
    print(instances, instances_with_toxicity_annotation, instances_with_identity_annotation)


def filter_train_instances(project_path):
    # ignore instances which are not annotated wrt IdTs
    instances = {}
    with open(project_path + '/data/train.csv') as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            if int(row["identity_annotator_count"]) > 0:
                instance_id = int(row["id"])
                instance_toxicity = float(row["target"])
                instance_text = row["comment_text"]
                instance_identity_annotations = []
                for identity in IDENTITY_ANNOTATIONS:
                    instance_identity_annotation = float(row[identity])
                    instance_identity_annotations.append(instance_identity_annotation)
                instances[instance_id] = (instance_text, instance_toxicity, instance_identity_annotations)
    with open(project_path + '/data/train_filtered.json', 'w') as outfile:
        json.dump(json.dumps(instances), outfile)


def count_instance_lengths(datasets):
    inst_lengths = defaultdict(int)
    for ds in datasets:
        for inst in ds.texts:
            inst_lengths[len([index for index in inst['input_ids'].tolist()[0] if index != 0])] += 1
    print(inst_lengths)


def sum_inst_annos(data):
    inst_id_sums = [0]*24
    for inst_id in data:
        inst_annos = data[inst_id][2]
        for index, anno_val in enumerate(inst_annos):
            inst_id_sums[index] += anno_val
    print(inst_id_sums)


def plot_anno_sums_vs_tox(data):
    tox_vals = []
    idt_vals = []
    for inst_id in data:
        tox_vals.append(data[inst_id][1])
        idt_vals.append(sum(data[inst_id][2]))
    pyplot.scatter(tox_vals, idt_vals)
    pyplot.show()


def run_data_experiments(project_path):
    count_instances(project_path)
    filter_train_instances(project_path)
    with open(project_path + '/data/train_filtered.json') as json_file:
        data = json.loads(json.load(json_file))
        print(len(data))
        sum_inst_annos(data)
        plot_anno_sums_vs_tox(data)
