"""General utility functions"""
import json
import logging
import pandas as pd


class Params:
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, "w") as f:
        # We need to convert the values to float for json
        # (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def sampling_stats(df, name):
    class_count = df["ordered_class"].unique().shape[0]
    subject_count = df["subject"].unique().shape[0]
    print(
        "{} dataset has {} classes and {} subjects".format(
            name, class_count, subject_count
        )
    )
    return


def gen_samples(image_label_file, sample_size, random_state=0):
    image_label_df = pd.read_csv(image_label_file)
    print(
        "Samping {} image from total {} images".format(
            sample_size, image_label_df.shape[0]
        )
    )

    # Sample
    class_min_size = int(sample_size / 160)
    sampled = image_label_df.groupby(["ordered_class"], group_keys=False).apply(
        lambda x: x.sample(min(len(x), class_min_size), random_state=random_state)
    )
    remain_size = sample_size - sampled.shape[0]
    remain_df = image_label_df.sample(n=remain_size, random_state=random_state)
    final_df = pd.concat([sampled, remain_df])

    sampling_stats(final_df, "sampled")

    return final_df
