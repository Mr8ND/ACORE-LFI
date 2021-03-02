from typing import Union
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
                 test_size: float = 0.2,
                 reference_g=None
                 ):

        # just to ease debugging
        logging.basicConfig(
            filename="./debugging.log",
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(message)s'
        )

        self.data = MuonFeatures.read_file(path=data_path)
        self.train_set, self.test_set = MuonFeatures.train_test_split(data=self.data, test_size=test_size)
        self.train_set_left = self.train_set
        if reference_g is None:
            self.reference_g = self.set_marginal_reference
        else:
            self.reference_g = reference_g

    @staticmethod
    def read_file(path: str):

        if not os.path.isfile(path):
            raise ValueError(f'Please provide a path to a file, got {path}')

        # if ASCII or CSV file
        if re.search(re.compile(r'.+\.(asc|csv)$'), path):
            data = pd.read_csv(path, sep=" ", header=None,
                               names=[f'v[{i}]' for i in range(0, 16)] + ['true_energy'])  # TODO: make it a param
            logging.info(f'Loaded data with shape {data.shape}')
            return data
        elif re.search(re.compile(r'.+\.hdf5$'), path):  # if HDF5 file
            raise NotImplementedError('HDF5 not supported yet')
        else:
            raise NotImplementedError('File format not supported yet')

    @staticmethod
    def train_test_split(data, test_size: float):
        assert isinstance(test_size, float)
        shuffled_index = np.random.permutation(data.index.to_numpy())
        test_index = shuffled_index[:int(test_size*len(shuffled_index))]
        train_set, test_set = data.loc[~data.index.isin(test_index), :].to_numpy(), data.loc[test_index, :].to_numpy()
        logging.info(f'train size: {len(train_set)}, test size: {len(test_set)}')
        assert (len(train_set.index) + len(test_set.index)) == len(data.index)
        return train_set, test_set

    def sample_param_values(self, sample_size: int, data: np.ndarray = None):
        if data is None:
            if sample_size > len(self.train_set_left):
                raise ValueError(f'Only {len(self.train_set_left)} simulations available, got {sample_size}')
            data = self.train_set_left

        logging.debug(f'sampling {sample_size} params from data of shape {data.shape}')

        # last column is true energy; unique needed because some params are equal
        return np.random.choice(np.unique(data[:, -1]), size=sample_size)

    def sample_sim(self, sample_size: int, true_param: np.ndarray, data: np.ndarray = None):
        if data is None:
            using_train_set = True
            data = self.train_set_left
        else:
            using_train_set = False
        logging.debug(f'sampling {sample_size} simulations from data of shape {data.shape}')

        if sample_size > len(data):
            raise ValueError(f'Only {len(data)} simulations available, got {sample_size}')

        # last column is true energy; [0] because returns tuple
        idxs = np.where(np.isin(data[:, -1], true_param))[0]
        # params can be equal for multiple rows -> take only one occurrence for each (total_sims == sample_size)
        idxs_unique = list(set(np.unique(data, return_index=True)[1]).intersection(set(idxs)))
        assert len(idxs_unique) == sample_size, f'{sample_size} samples requested, got {len(idxs_unique)}'
        simulations = data[idxs_unique]
        if using_train_set:
            # avoid reusing same data points until we have unused available
            self.train_set_left = np.delete(data, idxs_unique)
            if len(self.train_set_left) == 0:
                self.train_set_left = self.train_set
                logging.info('no more simulations available, training dataset re-instantiated')
        return simulations

    def sample_reference_g(self, sample_size, data: np.ndarray = None):

        if self.reference_g is None:
            true_param = self.sample_param_values(sample_size=1)
            return self.sample_sim(sample_size, true_param, data=data)
        else:
            self.reference_g.rvs(size=1)

    def generate_sample(self, sample_size, bernoulli_p=0.5, data: np.ndarray = None):
        theta = self.sample_param_values(sample_size, data=data)
        labels = np.random.binomial(n=1, p=bernoulli_p, size=sample_size)
        concat_matrix = np.hstack((theta.reshape(-1, 1),
                                  labels.reshape(-1, 1)))

        sample = np.apply_along_axis(arr=concat_matrix, axis=1,
                                     func1d=lambda row: self.sample_sim(sample_size=1, true_param=row[0], data=data)
                                     if row[1] else self.sample_reference_g(sample_size, data=data))
        return np.hstack((concat_matrix, sample.reshape(-1, 1)))


