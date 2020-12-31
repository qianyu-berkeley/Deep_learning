class Dataset:
    """Dataset class to abstract minibatch operation of a dataset
    - dataset is an iterator
    - the Dataset object is used as an input to torch Dataloader class
    """

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class DataBunch:
    """DataBunch class abstracts training and validation dataloader into a single class
    - dataloader object as input
    - pass number of training class
    """

    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl, self.valid_dl, self.c = train_dl, valid_dl, c

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset
