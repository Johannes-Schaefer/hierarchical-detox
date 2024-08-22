from clf import models


def do_clf(data, project_path, logfile, args, write_pred=False):
    (data_path, fixed_train_size, fixed_val_size, identity_label_set, filter_tokens_ids, use_class_weights, model,
     learning_rate, num_epochs, batch_size, early_stopping, save_model, model_tokenizer, tox_label_set, input_size)\
        = models.prepare_nn_model(data, logfile, args)

    saved_model = None
    if args.load_model is None:
        saved_model = models.do_train_test_run(logfile, data_path, args.fixed_train_size, args.fixed_val_size,
                                               model_tokenizer, tox_label_set, input_size, identity_label_set,
                                               filter_tokens_ids, use_class_weights, model, learning_rate, num_epochs,
                                               batch_size, project_path, early_stopping, args.save_model, write_pred)
    else:
        models.do_test_only_run(logfile, data_path, model_tokenizer, tox_label_set, input_size, filter_tokens_ids,
                                identity_label_set, model, batch_size, write_pred)
    return saved_model
