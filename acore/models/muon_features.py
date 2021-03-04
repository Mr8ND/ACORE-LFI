import re
import os
import pandas as pd
import numpy as np
import logging
from typing import Iterable


class MuonFeatures:

    # TODO: don't forget to divide in train/test -> we only have simulated data, use test as 'observed'
    # TODO: what is true_param in my case? array of theta from simulated dataset?
    # TODO: ensure or_sample and qr_sample are non-overlapping, unless no more unused data

    def __init__(self,
                 data_path: str,
                 t0_grid_granularity: int,
                 true_param_low: int = 100,  # GeV
                 true_param_high: int = 2000,  # GeV
                 param_dims: int = 1,  # only true energy
                 observed_dims: int = 16,  # 16 features
                 observed_size: float = 0.2,
                 reference_g=None,
                 param_column: int = -1,
                 debug: bool = False):
        if debug:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO

        # just to ease debugging
        logging.basicConfig(
            filename="./debugging.log",
            level=logging_level,
            format='%(asctime)s %(module)s %(levelname)s %(message)s'
        )

        self.data = MuonFeatures.read_file(path=data_path)
        # keep observed split to follow ACORE code structure
        self.train_set, self.obs_x, self.obs_param = MuonFeatures.train_obs_split(data=self.data,
                                                                                  observed_size=observed_size,
                                                                                  param_column=param_column)
        self.train_set_left = self.train_set
        self.d = param_dims
        self.observed_dims = observed_dims
        self.param_grid = np.linspace(true_param_low, true_param_high, t0_grid_granularity)
        self.reference_g = reference_g

        self.param_column = param_column

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
    def train_obs_split(data, observed_size: float, param_column: int = -1):
        assert isinstance(observed_size, float)
        shuffled_index = np.random.permutation(data.index.to_numpy())
        obs_index = shuffled_index[:int(observed_size*len(shuffled_index))]
        train_set, obs_set = data.loc[~data.index.isin(obs_index), :].to_numpy(), data.loc[obs_index, :].to_numpy()
        logging.info(f'train size: {len(train_set)}, observed size: {len(obs_set)}')
        assert (len(train_set.index) + len(obs_set.index)) == len(data.index)
        return train_set, obs_set[:param_column], obs_set[param_column]

    def sample_param_values(self, sample_size: int, data: np.ndarray = None):
        if data is None:
            if sample_size > len(self.train_set_left):
                raise ValueError(f'Only {len(self.train_set_left)} simulations available, got {sample_size}')
            data = self.train_set_left

        logging.debug(f'sampling {sample_size} params from data of shape {data.shape}')

        # unique needed because some params are equal
        return np.random.choice(np.unique(data[:, self.param_column]), size=sample_size)

    def sample_sim(self, sample_size: int, true_param: np.ndarray, data: np.ndarray = None):
        if data is None:
            using_train_set = True
            data = self.train_set_left
        else:
            using_train_set = False
        logging.debug(f'sampling {sample_size} simulations from data of shape {data.shape}')

        if sample_size > len(data):
            raise ValueError(f'Only {len(data)} simulations available, got {sample_size}')

        if not isinstance(true_param, Iterable):
            true_param = np.array([true_param])

        # [0] because np.where returns tuple
        idxs = np.where(np.isin(data[:, self.param_column], true_param))[0]
        # params can be equal for multiple rows -> take only one occurrence for each (total_sims == sample_size)
        idxs_unique = list(  # TODO: can avoid this step and use np.unique directly in idxs above?
            set(np.unique(data[:, self.param_column], return_index=True)[1]).intersection(set(idxs))  # [1] for idx obj
        )
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
            true_param = self.sample_param_values(sample_size=sample_size)
            return self.sample_sim(sample_size, true_param, data=data)
        else:
            self.reference_g.rvs(size=sample_size)

    def generate_sample(self, sample_size, bernoulli_p=0.5, data: np.ndarray = None):
        theta = self.sample_param_values(sample_size, data=data)
        labels = np.random.binomial(n=1, p=bernoulli_p, size=sample_size)
        concat_matrix = np.hstack((theta.reshape(-1, 1),
                                  labels.reshape(-1, 1)))

        sample = np.apply_along_axis(arr=concat_matrix, axis=1,
                                     func1d=lambda row: self.sample_sim(sample_size=1, true_param=row[0], data=data)
                                     if row[1] else self.sample_reference_g(sample_size, data=data))
        return np.hstack((concat_matrix, sample.reshape(-1, 1)))

    def sample_msnh(self, b_prime: int, sample_size: int):

        # TODO: theta_matrix should be a tensor of dimensions (b_prime, d, confidence_band_size)
        theta_matrix = self.sample_param_values(sample_size=b_prime).reshape(-1, self.d)
        assert theta_matrix.shape == (b_prime, self.d)

        sample_matrix = np.apply_along_axis(arr=theta_matrix, axis=1,
                                            func1d=lambda row: self.sample_sim(sample_size=sample_size, true_param=row))
        return theta_matrix, sample_matrix
