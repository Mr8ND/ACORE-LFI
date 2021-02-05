import re
import os
import pandas as pd
import numpy as np
import logging


class MuonFeatures:

    # TODO: don't forget to divide in train/test -> we only have simulated data, use test as 'observed'
    # TODO: what is true_param in my case? array of theta from simulated dataset?
    # TODO: ensure or_sample and qr_sample are non-overlapping, unless no more unused data

    def __init__(self,
                 data_path: str,
                 true_param_low: int = 100,  # GeV
                 true_param_high: int = 2000,  # GeV
                 test_size: float = 0.2
                 ):

        logging.basicConfig(
            filename="./debugging.log",
            level=logging.DEBUG,
            format='%(asctime)s %(levelname)s %(message)s')

        self.data = MuonFeatures.read_file(path=data_path)
        self.train_set, self.test_set = MuonFeatures.train_test_split(data=self.data, test_size=test_size)
        self.idx_data_used = set()

        pass

    @staticmethod
    def read_file(path: str):

        if not os.path.isfile(path):
            raise ValueError(f'Please provide a path to a file, got {path}')

        # if ASCII or CSV file
        if re.search(re.compile(r'.+\.(asc|csv)$'), path):
            data = pd.read_csv(path, sep=None, header=None, engine='python',  # infer separator
                               names=[f'v[{i}]' for i in range(0, 16)] + ['true_energy'])  # TODO: make it a param
            logging.info(f'Loaded data with shape {data.shape}')
            return data
        elif re.search(re.compile(r'.+\.hdf5$'), path):  # if HDF5 file
            raise NotImplementedError('HDF5 not supported yet')
        else:
            raise NotImplementedError('File format not supported yet')

    @staticmethod
    def train_test_split(data, test_size: float):
        shuffled_index = np.random.permutation(data.index.to_numpy())
        test_index = shuffled_index[:int(test_size*len(shuffled_index))]
        train_set, test_set = data.loc[~data.index.isin(test_index), :].to_numpy(), data.loc[test_index, :].to_numpy()
        logging.info(f'train size: {len(train_set)}, test size: {len(test_set)}')
        return train_set, test_set

    def sample_param_values(self, sample_size: int, data=None):
        if data is None:
            data = self.train_set
        return np.random.choice(data[:, -1], size=sample_size)

    def sample_sim(self):
        pass

    def set_reference_g(self):
        pass

    def generate_sample(self):
        pass







