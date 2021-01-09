import numpy as np
import math
import sys
sys.path.append('..')

from functools import partial
from utils.functions import tensor_4d_mesh, sample_from_matrix
from scipy.stats import multivariate_normal, poisson, rv_discrete, expon
from scipy.linalg import sqrtm
from scipy.optimize import Bounds, minimize


class InfernoToyLoader:

    def __init__(self, s_param=50, r_param=0.0, lambda_param=3.0, b_param=1000, benchmark=None,
                 nuisance_parameters=False, s_low=0, s_high=100, r_low=-5, r_high=5, lambda_low=0, lambda_high=10,
                 b_low=700, b_high=1300, out_dir='inferno_toy/', num_acore_grid=21, num_pred_grid=21,
                 empirical_marginal=True, *args, **kwargs):

        self.true_s = s_param
        self.s_low = s_low
        self.s_high = s_high
        self.sigmas_mat = [np.diag([5, 9]), np.diag([1, 1])]
        self.low_int = -5
        self.high_int = 1300

        if benchmark is not None:
            if benchmark not in [0, 1, 2, 3, 4, 5]:
                raise ValueError('benchmark variable needs to be an integer between 0 and 4, corresponding '
                                 'to the setup of the INFERNO paper by de Castro and Dorigo (2018). '
                                 'Currently %s.' % benchmark)
            if benchmark == 0:
                self.true_r = 0.0
                self.true_lambda = 3.0
                self.true_b = 1000
                self.r_low = 0.0
                self.r_high = 0.0
                self.lambda_low = 3.0
                self.lambda_high = 3.0
                self.b_low = 1000.0
                self.b_high = 1000.0
            if benchmark == 1:
                self.true_r = r_param
                self.true_lambda = 3.0
                self.true_b = 1000
                self.r_low = r_low
                self.r_high = r_high
                self.lambda_low = 3.0
                self.lambda_high = 3.0
                self.b_low = 1000.0
                self.b_high = 1000.0
            if benchmark == 2:
                self.true_r = r_param
                self.true_lambda = lambda_param
                self.true_b = 1000
                self.r_low = r_low
                self.r_high = r_high
                self.lambda_low = lambda_low
                self.lambda_high = lambda_high
                self.b_low = 1000.0
                self.b_high = 1000.0
            if benchmark == 3:
                self.true_r = r_param
                self.true_lambda = lambda_param
                self.true_b = 1000
                self.r_low = -1.2
                self.r_high = 1.2
                self.lambda_low = 0.0
                self.lambda_high = 6.0
                self.b_low = 1000.0
                self.b_high = 1000.0
            if benchmark == 4:
                self.true_r = r_param
                self.true_lambda = lambda_param
                self.true_b = b_param
                self.r_low = -1.2
                self.r_high = 1.2
                self.lambda_low = 0.0
                self.lambda_high = 6.0
                self.b_low = b_low
                self.b_high = b_high
            if benchmark == 5:
                self.true_r = 0.0
                self.true_lambda = 3.0
                self.true_b = b_param
                self.r_low = 0.0
                self.r_high = 0.0
                self.lambda_low = 3.0
                self.lambda_high = 3.0
                self.b_low = b_low
                self.b_high = b_high
        else:
            self.true_r = r_param
            self.true_lambda = lambda_param
            self.true_b = b_param
            self.r_low = r_low
            self.r_high = r_high
            self.lambda_low = lambda_low
            self.lambda_high = lambda_high
            self.b_low = b_low
            self.b_high = b_high

        self.true_param = np.array([self.true_s, self.true_r, self.true_lambda, self.true_b])
        active_params_tuple = [True, False, False, False]
        low_bounds, high_bounds = [], []
        for idx, (low, high) in enumerate([(self.r_low, self.r_high), (self.lambda_low, self.lambda_high),
                                           (self.b_low, self.b_high)]):
            if low != high:
                active_params_tuple[idx + 1] = True
                low_bounds.append(low)
                high_bounds.append(high)
        self.bounds_opt = Bounds(low_bounds, high_bounds)

        self.d = sum(active_params_tuple)
        self.active_params = self.d
        self.active_params_cols = np.where(active_params_tuple)[0]
        self.target_params_cols = [0] # target parameter is always the signal

        # If nuisance parameters are treated as such, then determine which columns are nuisance parameters and
        # which are not
        self.nuisance_flag = self.d > 1 and nuisance_parameters
        self.nuisance_params_cols = np.where(active_params_tuple)[0][1:] if self.nuisance_flag else None
        self.t0_grid_nuisance = None
        self.nuisance_global_param_val = None

        # Prediction grids have to be mindful of how many parameters are active
        self.num_pred_grid = num_pred_grid
        self.num_acore_grid = num_acore_grid

        # For the ACORE grid, it really depends on whether we consider nuisance parameters or not
        if self.nuisance_flag:
            self.pred_grid = np.linspace(start=self.s_low, stop=self.s_high, num=self.num_pred_grid)
            self.idx_row_true_param = np.where((self.pred_grid == self.true_param[0]))[0][0]
            self.acore_grid = None
        else:
            self.pred_grid = np.unique(
                tensor_4d_mesh(np.meshgrid(np.linspace(start=self.s_low, stop=self.s_high, num=self.num_pred_grid),
                                           np.linspace(start=self.r_low, stop=self.r_high, num=self.num_pred_grid),
                                           np.linspace(start=self.lambda_low, stop=self.lambda_high,
                                                       num=self.num_pred_grid),
                                           np.linspace(start=self.b_low, stop=self.b_high, num=self.num_pred_grid))),
                axis=0)[:, self.active_params_cols]
            self.idx_row_true_param = np.where((self.pred_grid == self.true_param[self.active_params_cols]).all(
                axis=1))[0][0]
            self.acore_grid = np.unique(
                tensor_4d_mesh(np.meshgrid(np.linspace(start=self.s_low, stop=self.s_high, num=self.num_pred_grid),
                                           np.linspace(start=self.r_low, stop=self.r_high, num=self.num_pred_grid),
                                           np.linspace(start=self.lambda_low, stop=self.lambda_high,
                                                       num=self.num_pred_grid),
                                           np.linspace(start=self.b_low, stop=self.b_high, num=self.num_acore_grid))),
                axis=0)[:, self.active_params_cols]

        self.regen_flag = False
        self.out_directory = out_dir

        self.empirical_marginal = empirical_marginal
        self.mean_instrumental = np.array([1, 1, 2])
        self.cov_instrumental = np.diag([5, 10, 5])
        self.g_distribution = multivariate_normal(mean=self.mean_instrumental, cov=self.cov_instrumental)

        self.b_sample_vec = [50, 100, 500, 1000, 5000, 10000, 50000, 100000]
        self.b_prime_vec = [500, 1000, 5000, 10000, 50000]
        self.d_obs = 3
        self.nuisance_param_val = None

    @staticmethod
    def _compute_mean_vec(r_param):
        return [np.array([2.0 + r_param, 0]), np.array([1, 1])]

    @staticmethod
    def _compute_mixing_param(b_param, s_param):
        return b_param / (s_param + b_param)

    def set_reference_g(self, size_reference):
        sample_mat_ref = self.generate_sample(sample_size=size_reference, p=1.0)[:, (self.d + 1):]
        self.mean_instrumental = np.average(sample_mat_ref, axis=0)
        self.cov_instrumental = np.diag(np.std(sample_mat_ref, axis=0) ** 2)
        self.g_distribution = multivariate_normal(mean=self.mean_instrumental, cov=self.cov_instrumental)

    def sample_empirical_marginal(self, sample_size):
        theta_vec_marg = self.sample_param_values(sample_size=sample_size)
        sample_mat = np.apply_along_axis(arr=theta_vec_marg.reshape(-1, self.d), axis=1,
                                   func1d=lambda row: self.sample_sim(
                                       sample_size=1, true_param=row)).reshape(-1, self.d_obs)
        return sample_mat

    def sample_param_values(self, sample_size):
        full_mat = np.random.uniform(low=self.s_low, high=self.s_high, size=sample_size).reshape(-1, 1)
        if 1 in self.active_params_cols:
            r_vec = np.random.uniform(low=self.r_low, high=self.r_high, size=sample_size).reshape(-1, 1)
            full_mat = np.hstack((full_mat, r_vec))
        if 2 in self.active_params_cols:
            lambda_vec = np.random.uniform(low=self.lambda_low, high=self.lambda_high, size=sample_size).reshape(-1, 1)
            full_mat = np.hstack((full_mat, lambda_vec))
        if 3 in self.active_params_cols:
            background_vec = np.random.uniform(low=self.b_low, high=self.b_high, size=sample_size).reshape(-1, 1)
            full_mat = np.hstack((full_mat, background_vec))
        return full_mat

    def _create_complete_param_vec(self, true_param):
        if 1 not in self.active_params_cols:
            true_param = np.concatenate((true_param, np.array([self.true_r])))
        if 2 not in self.active_params_cols:
            true_param = np.concatenate((true_param, np.array([self.true_lambda])))
        if 3 not in self.active_params_cols:
            true_param = np.concatenate((true_param, np.array([self.true_b])))
        return true_param

    def sample_sim(self, sample_size, true_param):
        if len(true_param) < 4:
            true_param = self._create_complete_param_vec(true_param)
        mixing_param = np.clip(true_param[3] / (true_param[0] + true_param[3]), 0, 1)
        cluster = np.random.binomial(n=1, p=mixing_param, size=sample_size)

        # Normal samples
        means_vec = self._compute_mean_vec(r_param=true_param[1])
        means = np.array([means_vec[idx] for idx in cluster])
        sigmas = np.array([self.sigmas_mat[idx] for idx in cluster])

        # From https://stackoverflow.com/questions/49681124/
        # vectorized-implementation-for-numpy-random-multivariate-normal
        # Compute the matrix square root of each covariance matrix.
        sqrtcovs = np.array([sqrtm(c) for c in sigmas])

        # Generate samples from the standard multivariate normal distribution.
        u = np.random.multivariate_normal(np.zeros(means[0].shape[0]), np.eye(means[0].shape[0]),
                                          size=(len(means), 1,))

        # Transform u.
        v = np.einsum('ijk,ikl->ijl', u, sqrtcovs)
        m = np.expand_dims(means, 1)
        samples_normal = (v + m).reshape(sample_size, 2)

        # Exponential samples
        exp_param_vec = np.take([true_param[2], 2], cluster)
        samples_exp = np.random.exponential(scale=exp_param_vec).reshape(-1, 1)

        return np.hstack((samples_normal, samples_exp))

    def sample_sim_check(self, sample_size, n):
        theta_mat = self.sample_param_values(sample_size=sample_size)
        assert theta_mat.shape == (sample_size, self.d)

        x_vec = np.array([
            self.sample_sim(sample_size=n, true_param=theta_0).reshape(-1, n) for theta_0 in theta_mat])
        return theta_mat, x_vec

    def generate_sample(self, sample_size, p=0.5, marginal=False):
        theta_mat = self.sample_param_values(sample_size=sample_size)
        assert theta_mat.shape == (sample_size, self.d)

        bern_vec = np.random.binomial(n=1, p=p, size=sample_size)
        concat_mat = np.hstack((theta_mat.reshape(-1, self.d), bern_vec.reshape(-1, 1)))

        if marginal:
            raise ValueError('Marginal not implemented for this example')

        if self.empirical_marginal:
            sample = np.apply_along_axis(arr=concat_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(
                                             sample_size=1, true_param=row[:self.d]) if row[self.d]
                                         else self.sample_empirical_marginal(sample_size=1))
        else:
            sample = np.apply_along_axis(arr=concat_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(
                                             sample_size=1, true_param=row[:self.d]) if row[self.d] else
                                         np.abs(self.g_distribution.rvs(size=1)).reshape(1, self.d_obs))
        return np.hstack((concat_mat, sample.reshape(-1, self.d_obs)))

    @staticmethod
    def _normal_multivariate_likelihood(x, mu, sigma):
        size = len(x)
        if size == len(mu) and (size, size) == sigma.shape:
            det = np.linalg.det(sigma)
            if det == 0:
                raise ValueError("The covariance matrix can't be singular")

            norm_const = 1.0 / (math.pow((2 * math.pi), float(size) / 2) * math.pow(det, 1.0 / 2))
            x_mu = (x - mu)
            inv = np.linalg.inv(sigma)
            result = math.pow(math.e, -0.5 * (x_mu @ inv @ x_mu.T))
            return norm_const * result
        else:
            raise ValueError("The dimensions of the input don't match")

    def _mixture_likelihood_manual(self, x, mu_vec, sigma_mats, mixing_param, lambda_param):
        first_term = self._normal_multivariate_likelihood(x=x[:2], mu=mu_vec[0], sigma=sigma_mats[0])
        first_term_exp = expon.pdf(x=x[2], scale=1/lambda_param)

        second_term = self._normal_multivariate_likelihood(x=x[:2], mu=mu_vec[1], sigma=sigma_mats[1])
        second_term_exp = expon.pdf(x=x[2], scale=1/2)

        return mixing_param * first_term * first_term_exp + (1 - mixing_param) * second_term * second_term_exp

    def compute_exact_prob(self, theta_vec, x_vec, p=0.5):
        x_mat = x_vec.reshape(-1, self.d_obs)
        theta_mat = theta_vec.reshape(-1, self.d)
        if theta_mat.shape[1] < 4:
            theta_mat = np.apply_along_axis(arr=theta_mat.reshape(-1, self.d), axis=1,
                                            func1d=lambda row: self._create_complete_param_vec(row))

        f_val = np.array([
            self._mixture_likelihood_manual(
                x=x, mu_vec=self._compute_mean_vec(theta_mat[ii, 1]),
                sigma_mats=self.sigmas_mat,
                mixing_param=self._compute_mixing_param(s_param=theta_mat[ii, 0], b_param=theta_mat[ii, 3]),
                lambda_param=theta_mat[ii, 2])
            for ii, x in enumerate(x_mat)
        ]).reshape(-1, )
        g_val = self.g_distribution.pdf(x=x_mat).reshape(-1, )
        return (f_val * p) / (f_val * p + g_val * (1 - p))

    def compute_exact_odds(self, theta_vec, x_vec, p=0.5):
        x_mat = x_vec.reshape(-1, self.d_obs)
        theta_mat = theta_vec.reshape(-1, self.d)

        if theta_mat.shape[1] < 4:
            theta_mat = np.apply_along_axis(arr=theta_mat.reshape(-1, self.d), axis=1,
                                            func1d=lambda row: self._create_complete_param_vec(row))

        f_val = np.array([
            self._mixture_likelihood_manual(
                x=x, mu_vec=self._compute_mean_vec(theta_mat[ii, 1]),
                sigma_mats=self.sigmas_mat,
                mixing_param=self._compute_mixing_param(s_param=theta_mat[ii, 0], b_param=theta_mat[ii, 3]),
                lambda_param=theta_mat[ii, 2])
            for ii, x in enumerate(x_mat)
        ]).reshape(-1, )
        g_val = self.g_distribution.pdf(x=x_mat).reshape(-1, )
        return (f_val * p) / (g_val * (1 - p))

    def compute_exact_or(self, t0, t1, x_obs):

        if len(t0) < 4:
            t0 = self._create_complete_param_vec(t0)
        if len(t1) < 4:
            t1 = self._create_complete_param_vec(t1)

        numerator = np.array([
            self._mixture_likelihood_manual(
                x=x, mu_vec=self._compute_mean_vec(t0[1]),
                sigma_mats=self.sigmas_mat,
                mixing_param=self._compute_mixing_param(s_param=t0[0], b_param=t0[3]),
                lambda_param=t0[2]) for x in x_obs
            ])

        denominator = np.array([
            self._mixture_likelihood_manual(
            x=x, mu_vec=self._compute_mean_vec(t1[1]),
            sigma_mats=self.sigmas_mat,
            mixing_param=self._compute_mixing_param(s_param=t1[0], b_param=t1[3]),
            lambda_param=t1[2]) for x in x_obs
        ])

        # Avoid straight up product for underflow
        return np.exp(np.sum(np.log(numerator)) - np.sum(np.log(denominator)))

    # def compute_exact_tau(self, x_obs, t0_val, meshgrid):
    #     return np.min(np.array([np.sum(np.log(self.compute_exact_or(
    #         x_obs=x_obs, t0=t0_val, t1=t1))) for t1 in meshgrid]))
    #
    # def compute_exact_tau_distr(self, t0_val, meshgrid, n_sampled, sample_size_obs):
    #     full_obs_sample = self.sample_sim(sample_size=n_sampled * sample_size_obs, true_param=t0_val)
    #     sample_mat = full_obs_sample.reshape((n_sampled, sample_size_obs, self.d_obs))
    #
    #     tau_sample = np.array([self.compute_exact_tau(
    #         x_obs=sample_mat[kk, :, :], t0_val=t0_val, meshgrid=meshgrid) for kk in range(sample_mat.shape[0])])
    #
    #     return tau_sample

    def _nuisance_parameter_func(self, nu_params, x_obs, target_params, clf_odds):
        param_mat = np.hstack((
            np.tile(np.concatenate((target_params.reshape(-1,), nu_params.reshape(-1,))),
                    x_obs.shape[0]).reshape(-1, self.d),
            x_obs.reshape(-1, self.d_obs)
        ))
        pred_mat = clf_odds.predict_proba(param_mat)
        return -1 * (np.prod(pred_mat[:, 1] / pred_mat[:, 0]))

    def nuisance_parameter_minimization(self, x_obs, target_params, clf_odds):
        res_min = minimize(
            fun=partial(self._nuisance_parameter_func, x_obs=x_obs,
                        target_params=target_params, clf_odds=clf_odds),
            x0=np.array([np.nan, 0, 5, 1000]).reshape(1, 4)[:, self.nuisance_params_cols].reshape(-1,),
            method='trust-constr', options={'verbose': 0}, bounds=self.bounds_opt)

        return np.concatenate((np.array(res_min.x), np.array([-1 * res_min.fun])))

    def calculate_nuisance_parameters_over_grid(self, t0_grid, clf_odds, x_obs):
        # in this toy example, there is always 2 params of interest
        nuisance_param_grid = np.apply_along_axis(
            arr=t0_grid.reshape(-1, 1), axis=1,
            func1d=lambda row: self.nuisance_parameter_minimization(
                x_obs=x_obs, target_params=row, clf_odds=clf_odds))

        # Now we create the full parameter matrix + likelihood values in the last column
        t0_grid_lik_values = np.hstack((
            t0_grid.reshape(-1, 1), nuisance_param_grid
        ))
        idx_global_max = np.argmax(t0_grid_lik_values[:, -1].reshape(-1, ))
        self.nuisance_global_param_val = nuisance_param_grid[idx_global_max, :-1]

        # Return the grids necessary to various sampling and ACORE grid
        t0_grid_out = t0_grid_lik_values[:, :-1]
        self.t0_grid_nuisance = t0_grid_out
        acore_grid_out = np.hstack((
            t0_grid.reshape(-1, 1),
            np.tile(self.nuisance_global_param_val, t0_grid.shape[0]).reshape(
                t0_grid.shape[0], self.nuisance_global_param_val.shape[0])
        ))
        return t0_grid_out, acore_grid_out

    def _complete_theta_param_nuisance(self, t0_val):
        return np.concatenate((np.array(t0_val).reshape(-1,), self.nuisance_param_val.reshape(-1,)))

    def sample_msnh_algo5(self, b_prime, sample_size):
        # If we have nuisance parameters, we replace the values of those parameters with the parameter sampled
        # with the nuisance parameters
        if self.nuisance_flag:
            theta_mat = sample_from_matrix(t0_grid=self.t0_grid_nuisance, sample_size=b_prime).reshape(-1, self.d)
        else:
            theta_mat = self.sample_param_values(sample_size=b_prime).reshape(-1, self.d)
        assert theta_mat.shape == (b_prime, self.d)

        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(sample_size=sample_size, true_param=row))
        return theta_mat, sample_mat.reshape(b_prime, sample_size, self.d_obs)

# if __name__ == '__main__':

    # from xgboost import XGBRFClassifier
    # from utils.functions import train_clf
    #
    # model_obj = InfernoToyLoader(benchmark=1, empirical_marginal=True)
    #
    # sample = model_obj.generate_sample(sample_size=300)
    # x_obs = model_obj.sample_sim(sample_size=10, true_param=model_obj.true_param)
    # gen_sample_func = model_obj.generate_sample
    # clf_odds = train_clf(sample_size=1000, clf_model=XGBRFClassifier(),
    #                      gen_function=gen_sample_func, clf_name='XGB', nn_square_root=True,
    #                      d=model_obj.d)
    #
    # t0_grid = model_obj.calculate_nuisance_parameters_over_grid(
    #     t0_grid=model_obj.pred_grid, clf_odds=clf_odds, x_obs=x_obs)
    #
    # theta_mat, sample_mat = model_obj.sample_msnh_algo5(b_prime=100, sample_size=10)
    # print(theta_mat)
    # print(sample_mat)
