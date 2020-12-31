import math
import json
import numpy as np
from shared_lib import data_utils


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true label as vector of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)  # keep minibatches same
    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.

    # number of complete mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = math.floor(m / mini_batch_size)
    # print("num of complete minibatches {}".format(num_complete_minibatches))

    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[
            :, k * mini_batch_size : k * mini_batch_size + mini_batch_size
        ]
        mini_batch_Y = shuffled_Y[
            :, k * mini_batch_size : k * mini_batch_size + mini_batch_size
        ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def process_params_file(fp, package):
    params = json.load(fp)
    hyper_params = {}
    hyper_params["seed"] = params["seed"]

    # Data types
    if params["data_type"] == "image":
        print("Image Data")
        data_dict = data_utils.load_image_data(
            params["train_data"], params["test_data"], params["color_scale"]
        )
    elif params["data_type"] == "2D":
        print("2D Data")
        file = params["data_file_mat"]
        data_dict = data_utils.load_2D_dataset(file)

    # Hyperparameters
    hyper_params["num_epochs"] = params["num_epochs"]
    hyper_params["learning_rate"] = params["learning_rate"]
    hyper_params["initialization"] = params["initialization"]
    hyper_params["layer_dims"] = params["layer_dims"]
    hyper_params["minibatch_size"] = params["minibatch_size"]
    # insert input feature size as the size of first NN layer
    hyper_params["layer_dims"].insert(0, data_dict["train_x"].shape[0])

    if package == "Parlour":
        hyper_params["optimizer_params"] = params["optimizer_params"]
        hyper_params["lambd"] = params["lambd"]
        hyper_params["keep_prob"] = params["keep_prob"]

    return data_dict, hyper_params
