import numpy as np
import sklearn
import h5py
import sklearn.datasets
import scipy.io


def load_image_data(path_train, path_test, color_scale):
    # load train data
    train_dataset = h5py.File(path_train, "r")
    train_x_orig = np.array(train_dataset["train_set_x"][:])
    train_y_orig = np.array(train_dataset["train_set_y"][:])

    # image data param size
    num_px = train_x_orig.shape[1]
    color_channel = 3

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

    return data_dict


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


def load_planar_dataset(sample_size, dimensionality, seed, randomness):
    np.random.seed(seed)

    m = sample_size  # number of examples
    N = int(m/2)  # number of points per class
    D = dimensionality  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N*j, N*(j+1))
        t = np.linspace(j*3.12, (j+1)*3.12, N) + np.random.randn(N)*randomness  # theta
        r = a*np.sin(4*t) + np.random.randn(N)*randomness  # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X, Y


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def load_2D_dataset(file):
    data = scipy.io.loadmat(file)
    content_keys = list(data.keys())[3:]

    # create a dictionary of data sets
    data_dict = {}
    for k in content_keys:
        data_dict[k] = data[k]

    data_2D_status(data_dict)

    return data_dict


def data_2D_status(data_dict):
    train_x_shape = data_dict['train_x'].shape
    train_y_shape = data_dict['train_y'].shape
    test_x_shape = data_dict['test_x'].shape
    test_y_shape = data_dict['test_y'].shape

    print("2D Data Shape")
    print("-------------------")
    print("train_x shape: {}".format(train_x_shape))
    print("test_x shape: {}".format(test_x_shape))
    print("train_y shape: {}".format(train_y_shape))
    print("test_y shape: {}".format(test_y_shape))
    print("")
