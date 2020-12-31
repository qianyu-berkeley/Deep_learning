import scipy.io

data = scipy.io.loadmat("./football_field_position_new.mat")
list(data.keys())[3:]

train_X = data["X"].T
train_Y = data["y"].T
test_X = data["Xval"].T
test_Y = data["yval"].T

scipy.io.savemat(
    "football_field_position_new.mat",
    {"train_x": train_X, "train_y": train_Y, "test_x": test_X, "test_y": test_Y},
)
