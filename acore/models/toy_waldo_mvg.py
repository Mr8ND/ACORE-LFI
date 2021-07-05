import numpy as np

from models.toy_mvn_multid_isotropic import ToyMVG


class ToyWaldoMVG(ToyMVG):

    def __init__(self,
                 observed_dims=2,
                 true_param=0.0, true_std=1.0,
                 param_grid_width=10, grid_sample_size=1000,
                 prior_type='uniform', normal_mean_prior=0.0, normal_std_prior=2.0,
                 empirical_marginal=True):

        super().__init__(observed_dims,
                         true_param, true_std,
                         param_grid_width, grid_sample_size,
                         prior_type, normal_mean_prior, normal_std_prior,
                         empirical_marginal)

    def generate_sample(self, sample_size, **kwargs):
        theta = self.sample_param_values(sample_size=sample_size)
        assert theta.shape == (sample_size, self.d)
        sample = np.apply_along_axis(arr=theta, axis=1,
                                     func1d=lambda row: self.sample_sim(sample_size=1,
                                                                        true_param=row[:self.d]))
        assert sample.shape == (sample_size, self.observed_dims)
        return np.hstack((theta, sample))
