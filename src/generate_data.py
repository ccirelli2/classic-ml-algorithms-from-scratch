"""
Module to generate data for algorithms

"""

###############################################################################
# Libraries
###############################################################################
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn import datasets
from decouple import config as d_config


###############################################################################
# Directories
###############################################################################
DIR_ROOT = d_config("DIR_ROOT")
DIR_SRC = d_config("DIR_SRC")
sys.path.append(DIR_ROOT)
sys.path.append(DIR_SRC)


###############################################################################
# Generate Data
###############################################################################
@dataclass
class DataGenerator:
    """
    Class object to generate synthetic datasets.

    Examples:

    """
    samples: int = None
    data: np.array = None

    def make_blobs(
            self,
            samples=100,
            features=2,
            centers=2,
            center_box=(-10, 10),
            cluster_std=1,
    ):
        """make_blobs.

        :param n_samples:
        :param n_features:
        :param centers:
        :param center_box:
        :param cluster_std:
        """
        if not samples:
            samples = self.samples

        self.data, _ = datasets.make_blobs(
            samples, features, centers, center_box, cluster_std
        )

        return self

    def make_dataframe(self):
        """
        Convert numpy data to pandas dataframe
        """
        if not self.data:
            raise Exception("No data exists")
        if not isinstance(self.data, np.array):
            raise Exception("self.data not of type np.array")

        return pd.DataFrame(self.data)


if __name__ == "__main__":
    pass
