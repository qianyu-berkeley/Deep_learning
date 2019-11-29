import json
import time
import h5py
import numpy as np
import deepdish as dd
from argparse import ArgumentParser
from deep_NN.deep_NN_model import deep_NN_model


def process_params_file(fp):
    params = json.load(fp)
    if params['data_type'] == 'image':
        data_dict, img_param_size = load_image_data(params['train_data'],
                                                    params['test_data'],
                                                    params['color_scale'],
                                                    params['verbose'])
    else:
        load_data()

    hyper_params = {}
    hyper_params['num_iterations'] = params['num_iterations']
    hyper_params['learning_rate'] = params['learning_rate']
    hyper_params['layer_dims'] = params['layer_dims']

    # insert image param size as the size of first NN layer
    hyper_params['layer_dims'].insert(0, img_param_size)

    return data_dict, hyper_params


def load_data():
    pass


def load_image_data(path_train, path_test, color_scale, verbose):
    # load train data
    train_dataset = h5py.File(path_train, "r")
    train_x_orig = np.array(train_dataset["train_set_x"][:])
    train_y_orig = np.array(train_dataset["train_set_y"][:])

    # image data param size
    num_px = train_x_orig.shape[1]
    color_channel = 3
    img_param_size = num_px*num_px*color_channel

    # load test data
    test_dataset = h5py.File(path_test, "r")
    test_x_orig = np.array(test_dataset["test_set_x"][:])
    test_y_orig = np.array(test_dataset["test_set_y"][:])
    # load test class label
    classes = np.array(test_dataset["list_classes"][:])

    # Transform and Reshape train and test data
    # The "-1" makes reshape flatten the remaining dimensions
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/color_scale
    test_x = test_x_flatten/color_scale

    train_y = train_y_orig.reshape((1, train_y_orig.shape[0]))
    test_y = test_y_orig.reshape((1, test_y_orig.shape[0]))

    if verbose == 2:
        image_data_status(train_x_orig, test_x_orig,
                          train_y_orig, test_y_orig,
                          train_x, test_x,
                          train_y, test_y)

    # create a dictionary of data sets
    data_dict = {}
    data_dict['train_x'] = train_x
    data_dict['train_y'] = train_y
    data_dict['test_x'] = test_x
    data_dict['test_y'] = test_y
    data_dict['classes'] = classes

    return data_dict, img_param_size


def image_data_status(train_x_orig, test_x_orig,
                      train_y_orig, test_y_orig,
                      train_x, test_x,
                      train_y, test_y):

    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print("")
    print("Image Data Status")
    print("Each image is of size: ({} x {} x {})".format(num_px, num_px, 3))
    print("Number of training examples: {}".format(m_train))
    print("Number of testing examples: {}".format(m_test))
    print("")
    print("Original Data Shape")
    print("-------------------")
    print("train_x_orig shape: {}".format(train_x_orig.shape))
    print("test_x_orig shape: {}".format(test_x_orig.shape))
    print("train_y_orig shape: {}".format(train_y_orig.shape))
    print("test_y_orig shape: {}".format(test_y_orig.shape))
    print("")
    print("Transformed Data Shape")
    print("----------------------")
    print("train_x's shape: {} ".format(train_x.shape))
    print("test_x's shape: {}".format(test_x.shape))
    print("train_y shape: {}".format(train_y.shape))
    print("test_y shape: {}".format(test_y.shape))
    print("")


def main(data_sets, num_iterations, learning_rate, layer_dims):

    # Define a Deep NN Model
    print("")
    print("Define deep NN Model ...")
    model = deep_NN_model()

    # Set hyper params
    print("Set Hyper Parameters ...")
    model.set_hyper_params(num_iterations, learning_rate, layer_dims)

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
