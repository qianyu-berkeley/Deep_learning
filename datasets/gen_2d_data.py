import numpy as np
from argparse import ArgumentParser
import sklearn.datasets
import scipy.io as sio


def gen_2D_data(n_samples=300, noise=0.2):
    np.random.seed(3)
    train_X, train_Y = sklearn.datasets.make_moons(n_samples=n_samples, noise=noise)
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    return train_X, train_Y


if __name__ == "__main__":

    parser = ArgumentParser(description="Genenrate 2D Data")
    parser.add_argument(
        "-ts",
        type=int,
        required=True,
        dest="n_samples_train",
        help="Num of samples of training set",
    )
    parser.add_argument(
        "-tn",
        type=float,
        dest="noise_train",
        required=True,
        help="Standard deviation of Guassian Noise of training set",
    )
    parser.add_argument(
        "-ds",
        type=int,
        required=True,
        dest="n_samples_dev",
        help="Num of samples of dev set",
    )
    parser.add_argument(
        "-dn",
        type=float,
        dest="noise_dev",
        required=True,
        help="Standard deviation of Guassian Noise of dev set",
    )
    parser.add_argument(
        "-f", type=str, dest="output_file", required=True, help="Output .mat file name"
    )

    args = parser.parse_args()
    print(
        "Generate 2D Moon Trainig Dataset with {} samples and Gussian Noise \
           STD of {}".format(
            args.n_samples_train, args.noise_train
        )
    )
    print(
        "Generate 2D Moon Dev Dataset with {} samples and Gussian Noise \
           STD of {}".format(
            args.n_samples_dev, args.noise_dev
        )
    )

    train_X, train_Y = gen_2D_data(args.n_samples_train, args.noise_train)
    dev_X, dev_Y = gen_2D_data(args.n_samples_dev, args.noise_dev)
    sio.savemat(
        args.output_file,
        {"train_x": train_X, "train_y": train_Y, "test_x": dev_X, "test_y": dev_Y},
    )
