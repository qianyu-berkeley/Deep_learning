import time
import importlib
import deepdish as dd
from argparse import ArgumentParser
from shared_lib import model_utils


def main(data_sets, seed, num_epochs, optimizer_params, learning_rate,
         initialization, minibatch_size, lambd, keep_prob, layer_dims):

    # Define a Deep NN Model
    print("")
    print("Define deep NN Model ...")
    model = deep_NN_module.deep_NN_model()
    model.seed = seed

    # Set hyper params
    print("Set Hyper Parameters ...")
    model.set_hyper_params(num_epochs, optimizer_params, learning_rate, minibatch_size,
                           initialization, lambd, keep_prob, layer_dims)

    # contruct and train model
    print("Model Construct and Train ...")
    if model.optimizer == 'gd':
        model.L_layer_model(data_sets['train_x'],
                            data_sets['train_y'],
                            print_cost=True)
    elif model.optimizer == 'adam':
        model.adam_model(data_sets['train_x'],
                         data_sets['train_y'],
                         print_cost=True)
    print("-----------------------------")
    print("")

    # Print train and dev accuracy
    print("Train Set Accuracy")
    model.predict(data_sets['train_x'], data_sets['train_y'])
    print("-----------------------")
    print("Dev Set Accuracy")
    model.predict(data_sets['test_x'], data_sets['test_y'])

    # Save model to string then to a file
    print("")
    print("Save Persistent Model")
    print("-----------------------")
    dd.io.save('./model_persist/model_saved.h5', model.save())


if __name__ == '__main__':

    parser = ArgumentParser(description='Deep NN Modeling')
    parser.add_argument('-m',
                        type=str,
                        required=True,
                        dest='module_name',
                        help="The name of the deep NN module you want to run")
    parser.add_argument('-p',
                        type=str,
                        dest='params_file',
                        required=True,
                        help="Model params defined in a JSON file")

    args = parser.parse_args()
    print("Deep NN Job called with arguments: {}".format(args))

    # Read deep NN module name and import the module based on the argument
    try:
        deep_NN_module = importlib.import_module('{}.deep_NN_model'.format(args.module_name))
    except ImportError as err:
        print('Import Library Error:', err)

    # Read model params inputs
    with open(args.params_file, 'r') as f:
        data_dict, deep_nn_hyper_params, seed = model_utils.process_params_file(f, args.module_name)
    print("Deep NN Hyper Params to use {}: ".format(deep_nn_hyper_params))

    # start the job
    start_time = time.time()

    # Running Deep NN Jobs
    main(data_dict, seed, **deep_nn_hyper_params)

    # end the job
    end_time = time.time()
    run_time_min = (end_time - start_time)/60.0  # run time in secs

    # Show run time in sec and mins...
    print("Module {} took {} minutes\n".format(args.module_name, run_time_min))
