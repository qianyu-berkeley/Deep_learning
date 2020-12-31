import gzip
import pickle
from fastai import datasets
from torch import tensor, nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from databunch import Dataset, DataBunch
from callbacks import AvgStatsCallback, ParamScheduler, Recorder
from trainer import Learner, Trainer
from utils import accuracy
from lr_scheduler import *


def get_data(URL):
    path = datasets.download_data(URL, ext=".gz")
    with gzip.open(path, "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    return map(tensor, (x_train, y_train, x_valid, y_valid))


def get_dls(train_ds, valid_ds, bs, **kwargs):
    """Create a train and validation dataloaders"""
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
        DataLoader(valid_ds, batch_size=bs * 2, **kwargs),
    )


def get_model(data, lr=0.5, nh=50):
    """Create a model and optmizer object"""
    m = data.train_ds.x.shape[1]
    model = nn.Sequential(nn.Linear(m, nh), nn.ReLU(), nn.Linear(nh, data.c))
    return model, optim.SGD(model.parameters(), lr=lr)


def train(train_ds, valid_ds, nh, bs, c, epoches):

    # Define dataset, loss function and model learner
    data = DataBunch(*get_dls(train_ds, valid_ds, bs), c)
    loss_func = F.cross_entropy
    learner = Learner(*get_model(data), loss_func, data)

    # Define callback
    lr_sche = combine_scheds([0.3, 0.7], [sched_cos(0.3, 0.6), sched_cos(0.6, 0.2)])
    acc_cbf = partial(AvgStatsCallback, accuracy)
    cbfs = [
        Recorder,
        partial(AvgStatsCallback, accuracy),
        partial(ParamScheduler, "lr", lr_sche),
    ]

    # Training
    train = Trainer(cb_funcs=cbfs)
    train.fit(epoches, learner)
    train.avg_stats.valid_stats.avg_stats


if __name__ == "__main__":
    MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl"
    x_train, y_train, x_valid, y_valid = get_data(MNIST_URL)
    train_ds, valid_ds = Dataset(x_train, y_train), Dataset(x_valid, y_valid)
    c = y_train.max().item() + 1
    nh, bs, epoches = 50, 64, 5
    train(train_ds, valid_ds, nh, bs, c, epoches)
