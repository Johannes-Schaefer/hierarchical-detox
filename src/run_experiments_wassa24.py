import os
import argparse
from datetime import datetime
from clf import tox_clf
from data import prepare_dataset_wassa
from pathlib import Path


PROJECT_PATH = os.path.dirname(__file__) + '/..'


def parse_args():
    parser = argparse.ArgumentParser(description='Experiment with toxicity detection on identity annotated data.')
    parser.add_argument('-m', '--model_name', dest='model_name', default='bert-base-uncased',
                        help='LLM name')
    parser.add_argument('-inl', '--input_max_len', dest='input_size', type=int, default=236,
                        help='max length of data instances')
    parser.add_argument('-hs', '--hidden_size', dest='hidden_size', type=int, default=768,
                        help='LLM hidden/output size')
    parser.add_argument('-ne', '--num_epochs', dest='num_epochs', type=int, default=15,
                        help='number of training epochs')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=8,
                        help='training batch size')
    parser.add_argument('-do', '--dropout', dest='dropout', type=float, default=0.2,
                        help='dropout for LLM output')
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=1e-6,
                        help='optimizer learning rate')
    parser.add_argument('-cw', '--no_class_weights', dest='use_class_weights', action='store_false',
                        help='do not use class weights')
    parser.add_argument('-l', '--log_to_file', dest='log_to_file', action='store_false',
                        help='do not log to file')
    parser.add_argument('-es', '--no_early_stopping', dest='no_early_stopping', action='store_true',
                        help='turn off early stopping')

    parser.add_argument('-clf_hs', '--classifier_toxicity', dest='classifier_toxicity',
                        action='store_false', help='deactivate toxicity classifier')
    parser.add_argument('-clf_idp', '--classifier_identity_present', dest='classifier_identity_present',
                        action='store_true',
                        help='activate identity present classifier (+/-)')
    parser.add_argument('-clf_idc', '--classifier_identity_category', dest='classifier_identity_category',
                        action='store_true',
                        help='activate identity category classifier (di/ge/ra/re/se/-)')
    parser.add_argument('-clf_idt', '--classifier_identity_term', dest='classifier_identity_term', 
                        action='store_true', help='activate identity term classifier (24 terms and -)')

    parser.add_argument('-adv_clf_hs', '--adversarial_classifier_toxicity', 
                        dest='adversarial_classifier_toxicity', action='store_true',
                        help='set toxicity classifier as adversarial')
    parser.add_argument('-adv_clf_idp', '--adversarial_classifier_identity_present',
                        dest='adversarial_classifier_identity_present', action='store_true',
                        help='set identity present classifier (+/-) as adversarial')
    parser.add_argument('-adv_clf_idc', '--adversarial_classifier_identity_category',
                        dest='adversarial_classifier_identity_category', action='store_true',
                        help='set identity category classifier (di/ge/ra/re/se) as adversarial')
    parser.add_argument('-adv_clf_idt', '--adversarial_classifier_identity_term',
                        dest='adversarial_classifier_identity_term', action='store_true',
                        help='set identity term classifier (24 terms) as adversarial')

    parser.add_argument('-adv_lambda', '--adversarial_classifier_lambda',
                        dest='adv_lambda', type=float, default=0.5,
                        help='set lambda for adversarial classifier')

    parser.add_argument('-fts', '--fixed_train_size', dest='fixed_train_size',
                        type=int, default=None, help='fixed size of train data')
    parser.add_argument('-fvs', '--fixed_val_size', dest='fixed_val_size',
                        type=int, default=None, help='fixed size of val data')

    parser.add_argument('-sm', '--save_model', dest='save_model', action='store_true',
                        help='save model after training (automatically when early stopping active)')
    parser.add_argument('-lm', '--load_model', dest='load_model', default=None,
                        help='load trained model from given path (no additional training will be done)')

    return parser.parse_args()


def create_logfile(log_to_file, args, run_name='train'):
    if log_to_file:
        logfile_name = PROJECT_PATH + '/logs/' + run_name + '_' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.log'
        with open(logfile_name, mode='a') as logfile:
            logfile.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ': command line arguments:' + str(vars(args))
                          + '\n')
    else:
        logfile_name = None
    return logfile_name


def run_full(data_path, project_path, arguments):
    logfile_name = create_logfile(arguments.log_to_file, arguments)
    trained_model = tox_clf.do_clf(data_path, project_path, logfile_name, arguments)
    return trained_model


def run_test(arguments, project_path, write_pred=False):
    if arguments.load_model is None:
        raise ValueError('Error: No model given for prediction/eval only run.')
    logfile_name = create_logfile(arguments.log_to_file, arguments, run_name='test')
    data_path = PROJECT_PATH + '/data/combined_test_ds.json'
    if write_pred:
        write_pred = logfile_name
    tox_clf.do_clf(data_path, project_path, logfile_name, arguments, write_pred=write_pred)


def run_test_other_data(arguments, project_path, model_type):
    if arguments.load_model is None:
        raise ValueError('Error: No model given for prediction/eval only run.')
    logfile_name = create_logfile(arguments.log_to_file, arguments, run_name='test-other-data')
    with open(logfile_name, mode='a') as logfile:
        logfile.write(datetime.now().strftime("%d/%m/%Y %H:%M:%S") + ': model:' + model_type + '\n')
    data_paths = [PROJECT_PATH + '/data/gk_ds.json',
                  PROJECT_PATH + '/data/stormfront_ds.json',
                  PROJECT_PATH + '/data/olid_ds.json',
                  PROJECT_PATH + '/data/asianprejudice_ds.json',
                  PROJECT_PATH + '/data/davidson_ds.json',
                  PROJECT_PATH + '/data/hasoc_ds.json',
                  PROJECT_PATH + '/data/sexist-tox_ds.json',
                  PROJECT_PATH + '/data/ethos_ds.json']
    arguments.tox_only = True
    for data_path in data_paths:
        tox_clf.do_clf(data_path, project_path, logfile_name, arguments)


def run_test_specific_ds(arguments, project_path, specific_ds):
    if arguments.load_model is None:
        raise ValueError('Error: No model given for prediction/eval only run.')
    tox_clf.do_clf(specific_ds, project_path, None, arguments)


def run_full_experiment_wassa(args):
    args.dataset = 'full'
    dataset_path = PROJECT_PATH + '/data/full_jigsaw_ds.json'
    args.fixed_train_size = 100000
    args.fixed_val_size = 50000

    args.num_epochs = 10
    args.batch_size = 32
    args.dropout = 0.2

    lrs = (5e-5, 2.5e-5, 1e-5, 7.5e-6, 5e-6)
    lambda_values = (0.10, 0.25, 0.5, 1.0)

    # baseline model - TOX (baseline)
    args.classifier_identity_present = True
    args.classifier_identity_category = True
    args.classifier_identity_term = True
    args.adversarial_classifier_identity_present = True
    args.adversarial_classifier_identity_category = True
    args.adversarial_classifier_identity_term = True
    args.adv_lambda = 0.

    for lr in lrs:
        args.learning_rate = lr
        saved_model = run_full(dataset_path, PROJECT_PATH, args)
        args.load_model = saved_model
        run_test(args, PROJECT_PATH)
        args.load_model = None

    # Tox IDT joint - Tox+O
    args.classifier_identity_present = True
    args.classifier_identity_category = False
    args.classifier_identity_term = False
    args.adversarial_classifier_identity_present = False
    args.adversarial_classifier_identity_category = False
    args.adversarial_classifier_identity_term = False
    args.adv_lambda = 0.
    for lr in lrs:
        args.learning_rate = lr
        saved_model = run_full(dataset_path, PROJECT_PATH, args)
        args.load_model = saved_model
        run_test(args, PROJECT_PATH)
        args.load_model = None

    # Tox IDT adv - Tox−O
    args.classifier_identity_present = True
    args.classifier_identity_category = False
    args.classifier_identity_term = False
    args.adversarial_classifier_identity_present = True
    args.adversarial_classifier_identity_category = False
    args.adversarial_classifier_identity_term = False
    for lr in lrs:
        args.learning_rate = lr
        for lambda_value in lambda_values:
            args.adv_lambda = lambda_value
            saved_model = run_full(dataset_path, PROJECT_PATH, args)
            args.load_model = saved_model
            run_test(args, PROJECT_PATH)
            args.load_model = None

    # Tox IDC joint - Tox+C
    args.classifier_identity_present = False
    args.classifier_identity_category = True
    args.classifier_identity_term = False
    args.adversarial_classifier_identity_present = False
    args.adversarial_classifier_identity_category = False
    args.adversarial_classifier_identity_term = False
    args.adv_lambda = 0.
    for lr in lrs:
        args.learning_rate = lr
        saved_model = run_full(dataset_path, PROJECT_PATH, args)
        args.load_model = saved_model
        run_test(args, PROJECT_PATH)
        args.load_model = None

    # Tox IDC adv - Tox-C
    args.classifier_identity_present = False
    args.classifier_identity_category = True
    args.classifier_identity_term = False
    args.adversarial_classifier_identity_present = False
    args.adversarial_classifier_identity_category = True
    args.adversarial_classifier_identity_term = False
    for lr in lrs:
        args.learning_rate = lr
        for lambda_value in lambda_values:
            args.adv_lambda = lambda_value
            saved_model = run_full(dataset_path, PROJECT_PATH, args)
            args.load_model = saved_model
            run_test(args, PROJECT_PATH)
            args.load_model = None

    # Tox IDT joint - Tox+I
    args.classifier_identity_present = False
    args.classifier_identity_category = False
    args.classifier_identity_term = True
    args.adversarial_classifier_identity_present = False
    args.adversarial_classifier_identity_category = False
    args.adversarial_classifier_identity_term = False
    args.adv_lambda = 0.
    for lr in lrs:
        args.learning_rate = lr
        saved_model = run_full(dataset_path, PROJECT_PATH, args)
        args.load_model = saved_model
        run_test(args, PROJECT_PATH)
        args.load_model = None

    # Tox IDT adv - Tox-I
    args.classifier_identity_present = False
    args.classifier_identity_category = False
    args.classifier_identity_term = True
    args.adversarial_classifier_identity_present = False
    args.adversarial_classifier_identity_category = False
    args.adversarial_classifier_identity_term = True
    for lr in lrs:
        args.learning_rate = lr
        for lambda_value in lambda_values:
            args.adv_lambda = lambda_value
            saved_model = run_full(dataset_path, PROJECT_PATH, args)
            args.load_model = saved_model
            run_test(args, PROJECT_PATH)
            args.load_model = None

    # Tox IDC IDT adv - Tox-C-I
    args.classifier_identity_present = False
    args.classifier_identity_category = True
    args.classifier_identity_term = True
    args.adversarial_classifier_identity_present = False
    args.adversarial_classifier_identity_category = True
    args.adversarial_classifier_identity_term = True
    for lr in lrs:
        args.learning_rate = lr
        for lambda_value in lambda_values:
            args.adv_lambda = lambda_value
            saved_model = run_full(dataset_path, PROJECT_PATH, args)
            args.load_model = saved_model
            run_test(args, PROJECT_PATH)
            args.load_model = None

    # Tox+O+C+I
    args.classifier_identity_present = True
    args.classifier_identity_category = True
    args.classifier_identity_term = True
    args.adversarial_classifier_identity_present = False
    args.adversarial_classifier_identity_category = False
    args.adversarial_classifier_identity_term = False
    args.adv_lambda = 0.
    for lr in lrs:
        args.learning_rate = lr
        saved_model = run_full(dataset_path, PROJECT_PATH, args)
        args.load_model = saved_model
        run_test(args, PROJECT_PATH)
        args.load_model = None

    # Tox−O−C−I
    args.classifier_identity_present = True
    args.classifier_identity_category = True
    args.classifier_identity_term = True
    args.adversarial_classifier_identity_present = True
    args.adversarial_classifier_identity_category = True
    args.adversarial_classifier_identity_term = True
    for lr in lrs:
        args.learning_rate = lr
        for lambda_value in lambda_values:
            args.adv_lambda = lambda_value
            saved_model = run_full(dataset_path, PROJECT_PATH, args)
            args.load_model = saved_model
            run_test(args, PROJECT_PATH)
            args.load_model = None

    # Tox+O,C,I
    args.classifier_identity_present = True
    args.classifier_identity_category = True
    args.classifier_identity_term = True
    args.adversarial_classifier_identity_present = False
    args.adversarial_classifier_identity_category = True
    args.adversarial_classifier_identity_term = True
    args.adv_lambda = 0.
    for lr in lrs:
        args.learning_rate = lr
        saved_model = run_full(dataset_path, PROJECT_PATH, args)
        args.load_model = saved_model
        run_test(args, PROJECT_PATH)
        args.load_model = None

    # Tox+O+C,I
    args.classifier_identity_present = True
    args.classifier_identity_category = True
    args.classifier_identity_term = True
    args.adversarial_classifier_identity_present = False
    args.adversarial_classifier_identity_category = False
    args.adversarial_classifier_identity_term = True
    args.adv_lambda = 0.
    for lr in lrs:
        args.learning_rate = lr
        saved_model = run_full(dataset_path, PROJECT_PATH, args)
        args.load_model = saved_model
        run_test(args, PROJECT_PATH)
        args.load_model = None

    # Tox+O−C−I
    args.classifier_identity_present = True
    args.classifier_identity_category = True
    args.classifier_identity_term = True
    args.adversarial_classifier_identity_present = False
    args.adversarial_classifier_identity_category = True
    args.adversarial_classifier_identity_term = True
    for lr in lrs:
        args.learning_rate = lr
        for lambda_value in lambda_values:
            args.adv_lambda = lambda_value
            saved_model = run_full(dataset_path, PROJECT_PATH, args)
            args.load_model = saved_model
            run_test(args, PROJECT_PATH)
            args.load_model = None

    # Tox+O+C−I
    args.classifier_identity_present = True
    args.classifier_identity_category = True
    args.classifier_identity_term = True
    args.adversarial_classifier_identity_present = False
    args.adversarial_classifier_identity_category = False
    args.adversarial_classifier_identity_term = True
    for lr in lrs:
        args.learning_rate = lr
        for lambda_value in lambda_values:
            args.adv_lambda = lambda_value
            saved_model = run_full(dataset_path, PROJECT_PATH, args)
            args.load_model = saved_model
            run_test(args, PROJECT_PATH)
            args.load_model = None
        for lambda_value in (2.0, 3.0):
            args.adv_lambda = lambda_value
            saved_model = run_full(dataset_path, PROJECT_PATH, args)
            args.load_model = saved_model
            run_test(args, PROJECT_PATH)
            args.load_model = None


if __name__ == '__main__':
    # prepare dataset
    prepare_dataset_wassa.prepare_all_datasets(PROJECT_PATH)

    Path(PROJECT_PATH + "/data").mkdir(exist_ok=True)
    Path(PROJECT_PATH + "/logs").mkdir(exist_ok=True)
    Path(PROJECT_PATH + "/models").mkdir(exist_ok=True)

    # run experiment
    run_arguments = parse_args()
    run_full_experiment_wassa(run_arguments)
