import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0, :], cmap=plt.cm.Spectral)
    plt.show()


def plot_2D_data(file, train=True):
    data = scipy.io.loadmat(file)
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    if train:
        plt.scatter(train_X[0, :], train_X[1, :],
                    c=train_Y[0, :], s=40, cmap=plt.cm.Spectral)
    else:
        plt.scatter(test_X[0, :], test_X[1, :],
                    c=test_Y[0, :], s=40, cmap=plt.cm.Spectral)
