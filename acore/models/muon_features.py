import re
import os
import pandas as pd
import numpy as np
import logging
import warnings
from typing import Iterable, Union
from tqdm import tqdm


class MuonFeatures:

    # TODO: don't forget to divide in train/test -> we only have simulated data, use test as 'observed'
    # TODO: what is true_param in my case? array of theta from simulated dataset?
    # TODO: ensure or_sample and qr_sample are non-overlapping, unless no more unused data

    def __init__(self,
                 t0_grid_granularity: int,
                 data: Union[str, pd.DataFrame, None] = None,
                 simulated_data: Union[str, pd.DataFrame, None] = None, 
                 observed_data: Union[str, pd.DataFrame, None] = None,
                 true_param_low: int = 100,  # GeV
                 true_param_high: int = 2000,  # GeV
                 param_dims: int = 1,  # only true energy
                 observed_dims: int = 16,  # 16 features
                 observed_sample_fraction: float = 0.2,
                 reference_g='marginal',
                 param_column: Union[int, list, np.array] = -1,
                 verbose=True,
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
        
        if isinstance(data, str):
            self.data = MuonFeatures.read_file(path=data)
        elif isinstance(data, pd.DataFrame):
            self.data = data
        elif data is None:
            if simulated_data is not None:
                self.data = simulated_data
            else:
                raise ValueError("Either data or simulated_data must be specified")
        else:
            raise ValueError(f'data must be either a filepath (str type) or a pandas DataFrame, got {type(data)}')
        
        if isinstance(param_column, int):
            assert param_dims == len([param_column])
            assert observed_dims == (self.data.shape[1] - 1)
            # check that param_column is the correct one
            theoretical_range = true_param_high - true_param_low
            observed_range = self.data.iloc[:, param_column].max() - self.data.iloc[:, param_column].min()
            if abs(theoretical_range - observed_range) > 0.01*theoretical_range:
                warnings.warn(f"Are you sure you set the correct parameter column? Got {param_column}")
        else:
            assert param_dims == len(list(param_column))
            assert observed_dims == (self.data.shape[1] - len(list(param_column)))
            warnings.warn("Check the code to make sure it's consistent for multidimensional parameter", )

        # keep observed split to follow ACORE code structure
        if data is None:
            self.train_set = simulated_data.to_numpy()
            self.obs_x = observed_data.drop(observed_data.columns[param_column], axis=1).to_numpy()
            self.obs_param = observed_data.iloc[:, param_column].to_numpy()
        else:
            self.train_set, self.obs_x, self.obs_param = \
                MuonFeatures.train_obs_split(data=self.data,
                                             observed_fraction=observed_sample_fraction,
                                             param_column=param_column)
        self.train_set_left = self.train_set
        self.d = param_dims
        self.observed_dims = observed_dims
        self.true_param_low = true_param_low
        self.true_param_high = true_param_high
        self.param_grid = np.linspace(true_param_low, true_param_high, t0_grid_granularity)
        self.reference_g = reference_g

        self.param_column = param_column
        # handy for sample_sim TODO: valid for full calo data?
        self.no_param_mask = np.full(self.data.shape[1], True)
        self.no_param_mask[param_column] = False

        self.verbose = verbose
        self.sampling_progress_bar = None

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
    def train_obs_split(data, observed_fraction: float, param_column: int = -1):
        assert isinstance(observed_fraction, float)

        shuffled_index = np.random.permutation(data.index.to_numpy())
        train_index = shuffled_index[int(observed_fraction * len(shuffled_index)):]
        obs_index = shuffled_index[:int(observed_fraction * len(shuffled_index))]
        train_set, obs_set = data.iloc[train_index, :].to_numpy(), data.iloc[obs_index, :].to_numpy()

        logging.info(f'train size: {len(train_set)}, observed size: {len(obs_set)}')
        assert (len(train_set) + len(obs_set)) == len(data.index)

        return train_set, np.delete(obs_set, param_column, axis=1), obs_set[:, param_column]

    def sample_param_values(self, sample_size: int, data: np.ndarray = None):
        if data is None:
            if sample_size > len(self.train_set_left):
                raise ValueError(f'Only {len(self.train_set_left)} simulations available, got {sample_size}')
            data = self.train_set_left

        logging.debug(f'sampling {sample_size} params from data of shape {data.shape}')

        idxs = np.arange(0, data.shape[0], 1)
        sampled_param_idxs = np.random.choice(idxs, size=sample_size)
        return sampled_param_idxs, data[sampled_param_idxs, self.param_column]

    # TODO: think about how to rewrite this more efficiently
    def sample_sim(self, sample_size: int, true_param_idx: np.ndarray, data: Union[np.ndarray, None] = None):
        if data is None:
            using_train_set = True
            data = self.train_set_left
        else:
            using_train_set = False
        logging.debug(f'sampling {sample_size} simulations from data of shape {data.shape}')

        if sample_size > len(data):
            raise ValueError(f'Only {len(data)} simulations available, got {sample_size}')

        if not isinstance(true_param_idx, np.ndarray):
            true_param = np.array([true_param_idx])

        simulations = data[true_param_idx, self.no_param_mask]
        if using_train_set:
            # avoid reusing same data points until we have unused available
            self.train_set_left = np.delete(data, true_param_idx, axis=0)
            if len(self.train_set_left) == 0:
                self.train_set_left = self.train_set
                logging.info('no more simulations available, training dataset re-instantiated')
        return simulations

    def sample_reference_g(self, sample_size, data: Union[np.ndarray, None] = None):

        if self.reference_g == 'marginal':
            # sample another simulation; will be matched with the (different) param sampled in label_dependent_sampling
            # -> EMPIRICAL MARGINAL, ensures independence between x and theta
            idx, true_param = self.sample_param_values(sample_size=sample_size, data=data)
            return self.sample_sim(sample_size, idx, data=data)
        else:
            self.reference_g.rvs(size=sample_size)

    def label_dependent_sampling(self, label: int, data: Union[np.ndarray, None] = None):
        # TODO: should reshape every time as (sample_size, param_dims) and (sample_size, observed_dims)?
        # TODO: when sampling from reference, should theta come from self.param_grid or from self.sample_param_values?
        if self.verbose:
            self.sampling_progress_bar.update(1)

        idx, theta = self.sample_param_values(sample_size=1, data=data)
        if label == 1:
            sim = self.sample_sim(sample_size=1, true_param_idx=idx, data=data)
        elif label == 0:
            sim = self.sample_reference_g(sample_size=1, data=data)
        else:
            raise ValueError(f'label must be either 0 or 1, got {label}')
        return np.concatenate((theta.reshape(-1, self.d),
                               sim.reshape(-1, self.observed_dims)), axis=1)

    # need kwargs cause some functions in ACORE call generate_sample with args we have not specified or don't need
    def generate_sample(self, sample_size, p=0.5, data: Union[np.ndarray, None] = None, **kwargs):
        if self.verbose:
            self.sampling_progress_bar = tqdm(total=sample_size, desc='Sampling %s simulations' % sample_size)

        labels = np.random.binomial(n=1, p=p, size=sample_size).reshape(-1, 1)
        param_and_sample = np.apply_along_axis(arr=labels, axis=1,
                                               func1d=lambda label: self.label_dependent_sampling(label, data=data)
                                               ).reshape(-1, self.d + self.observed_dims)
        out = np.concatenate((labels, param_and_sample), axis=1)

        logging.debug(f'generated sample dimensions: {out.shape}')
        if self.verbose:
            self.sampling_progress_bar.close()
            self.sampling_progress_bar = None
        return out

    def sample_msnh(self, b_prime: int, sample_size: int, data: Union[np.ndarray, None] = None):

        # TODO: use of label-dependent sampling will not work if obs_sample_size > 1.
        #  Will need multiple rows with same param and different obs (?)
        # TODO: this can be done without label-dependent sampling. Simply sample B' obs from data
        if sample_size > 1:
            raise NotImplementedError("Label-dependent sampling will not work if obs_sample_size > 1. See TODO above")

        logging.info("Sampling for many simple null hypothesis")

        if self.verbose:
            self.sampling_progress_bar = tqdm(total=b_prime, desc='Sampling %s simulations' % b_prime)

        # need to sample one at a time -> label dependent sampling with label always equal to 1
        labels = np.random.binomial(n=1, p=1, size=b_prime).reshape(-1, 1)
        assert labels.shape == (b_prime, 1)

        theta_sample_matrix = np.apply_along_axis(arr=labels, axis=1,
                                                  func1d=lambda label: self.label_dependent_sampling(label=label)
                                                  ).reshape(-1, self.d + self.observed_dims)
        if self.verbose:
            self.sampling_progress_bar.close()
            self.sampling_progress_bar = None

        theta_matrix, sample_matrix = theta_sample_matrix[:, :self.d], theta_sample_matrix[:, self.d:]
        return theta_matrix, sample_matrix
