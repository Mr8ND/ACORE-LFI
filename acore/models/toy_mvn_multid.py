import numpy as np
from math import ceil
import sys
sys.path.append('..')

from scipy.stats import multivariate_normal, uniform, norm
from itertools import product
from functools import partial
from utils.functions import sample_from_matrix
from scipy.optimize import Bounds, minimize


class ToyMVNMultiDLoader:

    def __init__(self, d_obs=2, mean_instrumental=0.0, std_instrumental=4.0, low_int=-5.0, high_int=5.0,
                 true_param=1.0, true_std=1.0, mean_prior=5.0, std_prior=2.0, uniform_grid_sample_size=125000,
                 out_dir='toy_mvn/', prior_type='uniform',
                 marginal=False, size_marginal=5000, empirical_marginal=True,
                 nuisance_parameters=False, **kwargs):

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
        
        # these are bounds for profiling the likelihood (for d-2 nuisance params)
        low_bounds = np.repeat(self.low_int, self.d-2)
        high_bounds = np.repeat(self.high_int, self.d-2)
        self.bounds_opt = Bounds(low_bounds, high_bounds)

        # first 2 params are of interest
        self.target_params_cols = [0, 1]  # target parameter is always the signal

        # Saves a Gaussian marginal and whether one needs to use the empirical distribution
        if marginal:
            self.compute_marginal_reference(size_marginal)
        self.empirical_marginal = empirical_marginal
        
        # If nuisance parameters are treated as such, then determine which columns are nuisance parameters and
        # which are not
        self.nuisance_flag = self.d > 1 and nuisance_parameters
        self.nuisance_params_cols = np.arange(2, self.d) if self.nuisance_flag else None
        self.t0_grid_nuisance = None
        self.nuisance_global_param_val = None
        
        if self.nuisance_flag:
            self.num_pred_grid = 51
            t0_grid = np.round(np.linspace(start=self.low_int, stop=self.high_int, num=self.num_pred_grid), 2)
            pred_iter_list = [t0_grid] * 2  # 2 true params, d-2 nuisance params
            list_full_product = list(product(*pred_iter_list))
            self.pred_grid = np.array(list_full_product)
            self.idx_row_true_param = list_full_product.index(tuple(self.true_param[self.target_params_cols].tolist()))
            self.acore_grid = None
        else:
            # If it's too high-dimensional, rather than gridding the parameter space we randomly sample
            if self.d < 3:
                self.num_pred_grid = 51
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
        
    def _nuisance_parameter_func(self, nu_params, x_obs, target_params, clf_odds):
        param_mat = np.hstack((
            np.tile(np.concatenate((target_params.reshape(-1,), nu_params.reshape(-1,))),
                    x_obs.shape[0]).reshape(-1, self.d),
            x_obs.reshape(-1, self.d_obs)
        ))
        pred_mat = clf_odds.predict_proba(param_mat)
        return -1 * (np.exp(np.sum(np.log(pred_mat[:, 1] / pred_mat[:, 0]))))

    def nuisance_parameter_minimization(self, x_obs, target_params, clf_odds):
        res_min = minimize(
            fun=partial(self._nuisance_parameter_func, x_obs=x_obs,
                        target_params=target_params, clf_odds=clf_odds),
            x0=np.zeros((1, self.d))[:, self.nuisance_params_cols].reshape(-1, ),
            method='trust-constr', options={'verbose': 0}, bounds=self.bounds_opt)

        return np.concatenate((np.array(res_min.x), np.array([-1 * res_min.fun])))

    def calculate_nuisance_parameters_over_grid(self, t0_grid, clf_odds, x_obs):
        # in this toy example, there is always 2 params of interest
        nuisance_param_grid = np.apply_along_axis(
            arr=t0_grid.reshape(-1, 2), axis=1,
            func1d=lambda row: self.nuisance_parameter_minimization(
                x_obs=x_obs, target_params=row, clf_odds=clf_odds))

        # Now we create the full parameter matrix + likelihood values in the last column
        t0_grid_lik_values = np.hstack((
            t0_grid.reshape(-1, 2), nuisance_param_grid.reshape(-1, (self.d - 2) + 1)
        ))
        idx_global_max = np.argmax(t0_grid_lik_values[:, -1].reshape(-1, ))
        self.nuisance_global_param_val = nuisance_param_grid[idx_global_max, :-1]

        # Return the grids necessary to various sampling and ACORE grid
        t0_grid_out = t0_grid_lik_values[:, :-1]
        self.t0_grid_nuisance = t0_grid_out
        acore_grid_out = np.hstack((
            t0_grid.reshape(-1, 2),
            np.tile(self.nuisance_global_param_val, t0_grid.shape[0]).reshape(
                t0_grid.shape[0], self.nuisance_global_param_val.shape[0])
        ))
        return t0_grid_out, acore_grid_out

    def _complete_theta_param_nuisance(self, t0_val):
        return np.concatenate((np.array(t0_val).reshape(-1,), self.nuisance_global_param_val.reshape(-1,)))
    
    def sample_msnh_algo5(self, b_prime, sample_size):
        # If we have nuisance parameters, we replace the values of those parameters with the parameter sampled
        # with the nuisance parameters
        if self.nuisance_flag:
            theta_mat = sample_from_matrix(t0_grid=self.t0_grid_nuisance, sample_size=b_prime).reshape(-1, self.d)
        else:
            theta_mat = self.sample_param_values(sample_size=b_prime).reshape(-1, self.d)
        assert theta_mat.shape == (b_prime, self.d)

        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(sample_size=sample_size,
                                                                            true_param=row[:self.d]))
        return theta_mat, sample_mat.reshape(b_prime, sample_size, self.d_obs)
    
    def compute_exactodds_nuisance_single_t0(self, obs_sample, t0):
        # always 2 target params, d-2 nuisance params in this example        
        target_sample = obs_sample[:, :2]
        target_mle = target_sample.mean(axis=0)
        t1 = np.append(target_mle, t0[2:])
        odds_t0 = self.compute_exact_or(t0=t0, t1=t1, x_obs=obs_sample)
        assert isinstance(odds_t0, float)
        return odds_t0
    