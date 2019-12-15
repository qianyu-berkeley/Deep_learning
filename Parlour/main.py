import json
import time
import h5py
import numpy as np
import deepdish as dd
from argparse import ArgumentParser
from deep_NN.deep_NN_model import deep_NN_model
from deep_NN import data_utils


def process_params_file(fp):
    params = json.load(fp)
    hyper_params = {}
    hyper_params['num_iterations'] = params['num_iterations']
    hyper_params['learning_rate'] = params['learning_rate']
    hyper_params['layer_dims'] = params['layer_dims']
    hyper_params['initialization'] = params['initialization']
    hyper_params['lambd'] = params['lambd']
    hyper_params['keep_prob'] = params['keep_prob']

    if params['data_type'] == 'image':
        print("Image Data")
        data_dict, img_param_size = data_utils.load_image_data(params['train_data'],
                                                               params['test_data'],
                                                               params['color_scale'],
                                                               params['verbose'])
        # insert image param size as the size of first NN layer
        hyper_params['layer_dims'].insert(0, img_param_size)
    elif params['data_type'] == '2D':
        print("2D Data")
        file = params['data_file_mat']
        data_dict, X_dim0 = data_utils.load_2D_dataset(file)
        # insert X dimension as the size of first NN layer
        hyper_params['layer_dims'].insert(0, X_dim0)

    return data_dict, hyper_params


def main(data_sets, num_iterations, learning_rate, initialization,
         lambd, keep_prob, layer_dims):

    # Define a Deep NN Model
    print("")
    print("Define deep NN Model ...")
    model = deep_NN_model()

    # Set hyper params
    print("Set Hyper Parameters ...")
    model.set_hyper_params(num_iterations, learning_rate, initialization,
                           lambd, keep_prob, layer_dims)

    # contruct and train model
    print("Model Construct and Train ...")
    model.L_layer_model(data_sets['train_x'],
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

    parser = ArgumentParser(description='Deep NN Jobs')
    parser.add_argument('-job',
                        type=str,
                        required=True,
                        dest='job_name',
                        help="The name of the job module you want to run")
    parser.add_argument('-p',
                        type=str,
                        dest='params_file',
                        required=True,
                        help="Model params defined in a JSON file")

    args = parser.parse_args()
    print("Deep NN Job called with arguments: {}".format(args))

    # Read model params inputs
    with open(args.params_file, 'r') as f:
        data_dict, deep_nn_hyper_params = process_params_file(f)
    print("Deep NN Hyper Params to use {}: ".format(deep_nn_hyper_params))

    # start the job
    start_time = time.time()

    # Running Deep NN Jobs
    main(data_dict, **deep_nn_hyper_params)

    # end the job
    end_time = time.time()
    run_time_min = (end_time - start_time)/60.0  # run time in secs

    # Show run time in sec and mins...
    print("Job {} took {} minutes\n".format(args.job_name, run_time_min))
