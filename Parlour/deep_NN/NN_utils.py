import numpy as np


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy

    Arguments:
    -----------
    Z -- numpy array of any shape

    Returns:
    ---------
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1/(1+np.exp(-Z))
    cache = Z

    return A, cache


def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    -----------
    Z -- Output of the linear layer, of any shape

    Returns:
    ---------
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ;
             stored for computing the backward pass efficiently
    """
    A = np.maximum(0, Z)
    cache = Z
    assert(A.shape == Z.shape)

    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    -----------
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    -----------
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    # just converting dz to a correct object.
    dZ = np.array(dA, copy=True)
    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)

    return dZ


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    -----------
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    -----------
    dZ -- Gradient of the cost with respect to Z
    """

    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)

    return dZ


def image2vector(image):
    """
    Convert an image in 3-D pixel to a vector of different shape that works
    with deep learning algorithms

    Argument:
    ---------
    image: a numpy array of shape (length, height, depth)

    Returns:
    --------
    a vector of shape (length*height*depth, 1)
    """
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)
    return v


def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (2D)
    (to have unit length).

    Argument:
    ----------
    x: A numpy matrix of shape (n, m)

    Returns:
    The normalized (by row) numpy matrix
    """
    x_norm = np.linalg.norm(x, axis=1, ord=2, keepdims=True)
    x = x / x_norm
    return x


def softmax(x):
    """Calculates the softmax for each row of the input x.
    softmax is sigmoid for multi-classes

    Argument:
    ---------
    x: A numpy matrix of shape (n,m)

    Returns:
    ---------
    A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


def L1(yhat, y):
    """Calculate L1 loss

    Arguments:
    -----------
    yhat: vector of size m (predicted labels)
    y:  vector of size m (true labels)

    Returns:
    --------
    the value of the L1 loss function defined above
    """
    loss = np.sum(abs(y - yhat))
    return loss


def L2(yhat, y):
    """Calculate L2 loss

    Arguments:
    ----------
    yhat: vector of size m (predicted labels)
    y: vector of size m (true labels)

    Returns:
    ---------
    the value of the L2 loss function defined above
    """
    loss = np.sum(np.dot((y - yhat), (y - yhat)))
    return loss


def invert_dropout(A, keep_prob):
    """Invert dropout parameters

    Arguments:
    ----------
    A: parameters matrix
    keep_prob: invert dropout probability

    Returns:
    ---------
    AD: parameter matrix after dropout appied and normalized
    D: dropout matrix
    """
    D = np.random.rand(A.shape[0], A.shape[1])
    D = (D < keep_prob).astype(float)
    AD = A * D
    AD = AD / keep_prob
    return AD, D
