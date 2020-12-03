import numpy as np
from math import ceil
import sys
sys.path.append('..')

from scipy.stats import multivariate_normal, uniform, norm
from scipy.optimize import Bounds
from itertools import product


class ToyMVNMultiDLoader:

    def __init__(self, d_obs=2, mean_instrumental=0.0, std_instrumental=4.0, low_int=0.0, high_int=10.0,
                 true_param=5.0, true_std=1.0, mean_prior=5.0, std_prior=2.0, uniform_grid_sample_size=2500,
                 out_dir='toy_mvn/', prior_type='uniform',
                 marginal=False, size_marginal=5000, empirical_marginal=True, **kwargs):

        self.low_int = low_int
        self.high_int = high_int
        self.out_directory = out_dir
        self.d = d_obs
        self.d_obs = d_obs
        self.bounds_opt = Bounds([self.low_int] * self.d, [self.high_int] * self.d)

        if prior_type == 'uniform':
            self.prior_distribution = uniform(loc=self.low_int, scale=(self.high_int - self.low_int))
        elif prior_type == 'normal':
            self.prior_distribution = norm(loc=mean_prior, scale=std_prior**2)
        else:
            raise ValueError('The variable prior_type needs to be either uniform or normal.'
                             ' Currently %s' % prior_type)

        self.mean_instrumental = np.repeat(mean_instrumental, self.d_obs) if isinstance(mean_instrumental, float) \
            else mean_instrumental
        self.cov_instrumental = std_instrumental * np.eye(self.d_obs) if isinstance(std_instrumental, float) \
            else std_instrumental
        self.g_distribution = multivariate_normal(self.mean_instrumental, self.cov_instrumental)

        self.true_param = np.repeat(true_param, self.d) if isinstance(true_param, float) \
            else np.array(true_param)
        if self.true_param.shape[0] != self.d:
            raise ValueError('The true_param variable passed is not of the right dimension. '
                             'Currently %s, while it has to be %s.' % (self.true_param.shape[0], self.d))
        self.true_cov = true_std * np.eye(d_obs) if isinstance(true_std, float) else true_std

        if marginal:
            self.compute_marginal_reference(size_marginal)
        
        self.empirical_marginal = empirical_marginal

        # If it's too high-dimensional, rather than gridding the parameter space we randomly sample
        if self.d < 3:
            self.num_pred_grid = 21
            t0_grid = np.round(np.linspace(start=self.low_int, stop=self.high_int, num=self.num_pred_grid), 2)
            pred_iter_list = [t0_grid] * d_obs
            list_full_product = list(product(*pred_iter_list))
            self.pred_grid = np.array(list_full_product)
            self.idx_row_true_param = list_full_product.index(tuple(self.true_param.tolist()))
        else:
            if not uniform_grid_sample_size % self.d == 0:
                self.num_pred_grid = ceil(uniform_grid_sample_size/self.d) * self.d
            else:
                self.num_pred_grid = uniform_grid_sample_size

            pred_grid = np.random.uniform(
                low=self.low_int, high=self.high_int, size=self.num_pred_grid).reshape(-1, self.d)
            self.pred_grid = np.vstack((self.true_param.reshape(1, self.d), pred_grid))
            self.idx_row_true_param = 0
        self.acore_grid = self.pred_grid

        self.nuisance_flag = False

    def sample_sim(self, sample_size, true_param):
        return multivariate_normal(mean=true_param, cov=self.true_cov).rvs(sample_size).reshape(sample_size, self.d_obs)

    def sample_param_values(self, sample_size):
        unique_theta = self.prior_distribution.rvs(size=sample_size * self.d)
        return np.clip(unique_theta.reshape(sample_size, self.d), a_min=self.low_int, a_max=self.high_int)

    def compute_marginal_reference(self, size_marginal):
        theta_vec_marg = self.sample_param_values(sample_size=size_marginal)
        marginal_sample = np.apply_along_axis(arr=theta_vec_marg.reshape(-1, self.d), axis=1,
                                              func1d=lambda row: self.sample_sim(
                                                  sample_size=1, true_param=row)).reshape(-1, self.d_obs)

        self.mean_instrumental = np.average(marginal_sample, axis=0)
        self.cov_instrumental = np.diag(np.var(marginal_sample, axis=0))
        self.g_distribution = multivariate_normal(mean=self.mean_instrumental, cov=self.cov_instrumental)

    def sample_empirical_marginal(self, sample_size):
        theta_vec_marg = self.sample_param_values(sample_size=sample_size)
        return np.apply_along_axis(arr=theta_vec_marg.reshape(-1, self.d), axis=1,
                                   func1d=lambda row: self.sample_sim(
                                       sample_size=1, true_param=row)).reshape(-1, self.d_obs)
    
    def generate_sample(self, sample_size, p=0.5, **kwargs):
        theta_vec = self.sample_param_values(sample_size=sample_size)
        bern_vec = np.random.binomial(n=1, p=p, size=sample_size)
        concat_mat = np.hstack((theta_vec.reshape(-1, self.d), bern_vec.reshape(-1, 1)))
        
        if self.empirical_marginal:
            sample = np.apply_along_axis(arr=concat_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(
                                             sample_size=1, true_param=row[:self.d]) if row[self.d]
                                         else self.sample_empirical_marginal(sample_size=1))
        else:
            sample = np.apply_along_axis(arr=concat_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(
                                             sample_size=1, true_param=row[:self.d]) if row[self.d]
                                         else self.g_distribution.rvs(size=1))
        return np.hstack((concat_mat, sample.reshape(sample_size, self.d_obs)))

    def sample_msnh_algo5(self, b_prime, sample_size):
        theta_mat = self.sample_param_values(sample_size=b_prime).reshape(-1, self.d)
        assert theta_mat.shape == (b_prime, self.d)
        
        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(
                                             sample_size=sample_size, true_param=row[:self.d]))
        return theta_mat, sample_mat.reshape(b_prime, sample_size, self.d_obs)

    def _compute_multivariate_normal_pdf(self, x, mu):
        return multivariate_normal.pdf(x=x, mean=mu, cov=self.true_cov)

    def compute_exact_or(self, t0, t1, x_obs):
        return self._compute_multivariate_normal_pdf(
            x=x_obs, mu=t0) / self._compute_multivariate_normal_pdf(x=x_obs, mu=t1)
    
    def compute_exact_prob(self, theta_vec, x_vec, p=0.5):
        x_vec = x_vec.reshape(-1, self.d_obs)
        theta_vec = theta_vec.reshape(-1, self.d)
        
        f_val = np.array([self._compute_multivariate_normal_pdf(
            x=x, mu=theta_vec[ii, :]) for ii, x in enumerate(x_vec)]).reshape(-1, )
        g_val = self.g_distribution.pdf(x=x_vec).reshape(-1, )
        return (f_val * p) / (f_val * p + g_val * (1 - p))

    def compute_exact_odds(self, theta_vec, x_vec, p=0.5):
        x_vec = x_vec.reshape(-1, self.d_obs)
        theta_vec = theta_vec.reshape(-1, self.d)

        f_val = np.array([self._compute_multivariate_normal_pdf(
            x=x, mu=theta_vec[ii, :]) for ii, x in enumerate(x_vec)]).reshape(-1, )
        g_val = self.g_distribution.pdf(x=x_vec).reshape(-1, )
        return (f_val * p) / (g_val * (1 - p))

    def compute_exact_likelihood(self, x_obs, true_param):
        return self._compute_multivariate_normal_pdf(x=x_obs, mu=true_param)

    def compute_exact_lr_simplevsimple(self, x_obs, t0, t1):
        ll_gmm_t0 = np.sum(np.log(self._compute_multivariate_normal_pdf(x=x_obs, mu=t0)))
        ll_gmm_t1 = np.sum(np.log(self._compute_multivariate_normal_pdf(x=x_obs, mu=t1)))
        return ll_gmm_t0 - ll_gmm_t1

    @staticmethod
    def compute_mle(x_obs):
        return np.mean(x_obs, axis=1)

    def compute_exact_lr_simplevcomp(self, x_obs, t0, mle):
        ll_gmm_t0 = np.sum(np.log(self._compute_multivariate_normal_pdf(x=x_obs, mu=t0)))
        ll_gmm_t1 = np.sum(np.log(self._compute_multivariate_normal_pdf(x=x_obs, mu=mle)))
        return ll_gmm_t0 - ll_gmm_t1

    def calculate_nuisance_parameters_over_grid(self, *args, **kwargs):
        raise NotImplementedError('No nuisance parameter for this class.')
