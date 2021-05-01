import numpy as np
import sys
sys.path.append('..')

from scipy.stats import multivariate_normal, poisson
from scipy.optimize import Bounds, minimize
from functools import partial


class HepCountingNuisanceLoader:

    def __init__(self, tau_param=1.0, signal_val=10, background_val=100, empirical_marginal=True,
                 low_int_bg=80, high_int_bg=120, nuisance_parameters=True, num_pred_grid=201,
                 low_int_signal=0, high_int_signal=20, out_dir='hep_counting/',
                 num_acore_grid=101, *args, **kwargs):

        if not nuisance_parameters:
            raise NotImplementedError('Example only works for the background parameter being the nuisace parameter.')

        self.tau_param = tau_param
        self.low_int_bg = low_int_bg
        self.high_int_bg = high_int_bg
        self.low_int_signal = low_int_signal
        self.high_int_signal = high_int_signal
        self.bounds_opt = Bounds([low_int_bg], [high_int_bg])

        self.true_param = np.array([signal_val, background_val])
        self.regen_flag = False
        self.out_directory = out_dir
        self.mean_instrumental = None
        self.cov_instrumental = None
        self.g_distribution = None

        self.b_sample_vec = [50, 100, 500, 1000, 5000, 10000, 50000, 100000]
        self.b_prime_vec = [100, 500, 1000, 5000, 10000, 50000, 100000]

        self.d = 2
        self.d_obs = 2
        self.nuisance_flag = nuisance_parameters
        self.nuisance_params_cols = [1]
        self.target_params_cols = [0]

        self.num_pred_grid = num_pred_grid
        self.num_acore_grid = num_acore_grid
        self.pred_grid = np.linspace(start=self.low_int_signal, stop=self.high_int_signal, num=self.num_pred_grid)
        self.idx_row_true_param = np.where((self.pred_grid == self.true_param[0]))[0][0]
        self.acore_grid = np.linspace(start=self.low_int_signal, stop=self.high_int_signal, num=self.num_acore_grid)

        self.t0_grid_nuisance = None
        self.nuisance_global_param_val = None
        self.empirical_marginal = empirical_marginal

    def set_reference_g(self, size_reference):
        background_vec_ref = np.random.uniform(low=self.low_int_bg, high=self.high_int_bg,
                                               size=size_reference).reshape(-1, 1)
        mu_vec_ref = np.random.uniform(low=self.low_int_signal, high=self.high_int_signal,
                                       size=size_reference).reshape(-1, 1)
        theta_mat_ref = np.hstack((background_vec_ref, mu_vec_ref))
        sample_mat_ref = np.apply_along_axis(arr=theta_mat_ref, axis=1,
                                             func1d=lambda row: np.array([
                                                 np.random.poisson(lam=row[0] + row[1], size=1),
                                                 np.random.poisson(
                                                     lam=self.tau_param * row[1], size=1)]).reshape(-1, 2))
        sample_mat_ref = sample_mat_ref.reshape(-1, 2)
        self.mean_instrumental = np.average(sample_mat_ref, axis=0)
        self.cov_instrumental = np.diag(np.std(sample_mat_ref, axis=0) ** 2)
        self.g_distribution = multivariate_normal(mean=self.mean_instrumental, cov=self.cov_instrumental)

    def set_reference_g_no_sample(self, mean_instrumental, cov_instrumental):
        self.mean_instrumental = mean_instrumental
        self.cov_instrumental = cov_instrumental
        self.g_distribution = multivariate_normal(mean=self.mean_instrumental, cov=self.cov_instrumental)

    def sample_empirical_marginal(self, sample_size):
        theta_vec_marg = self.sample_param_values(sample_size=sample_size)
        sample_mat = np.apply_along_axis(arr=theta_vec_marg.reshape(-1, self.d), axis=1,
                                         func1d=lambda row: self.sample_sim(
                                            sample_size=1, true_param=row)).reshape(-1, self.d_obs)
        return sample_mat

    def sample_param_values(self, sample_size):
        background_vec = np.random.uniform(low=self.low_int_bg, high=self.high_int_bg,
                                           size=sample_size).reshape(-1, 1)
        mu_vec = np.random.uniform(low=self.low_int_signal, high=self.high_int_signal,
                                   size=sample_size).reshape(-1, 1)
        return np.hstack((mu_vec, background_vec))

    def sample_sim(self, sample_size, true_param):
        first_dim = np.random.poisson(lam=true_param[0] + true_param[1], size=sample_size).reshape(-1, 1)
        second_dim = np.random.poisson(lam=self.tau_param * true_param[1], size=sample_size).reshape(-1, 1)
        return np.hstack((first_dim, second_dim))

    def sample_sim_check(self, sample_size, n):
        theta_mat = self.sample_param_values(sample_size=sample_size)
        assert theta_mat.shape == (sample_size, 2)

        x_vec = np.array([
            self.sample_sim(sample_size=n, true_param=theta_0).reshape(-1, n) for theta_0 in theta_mat])
        return theta_mat, x_vec

    def generate_sample(self, sample_size, p=0.5, marginal=False):
        theta_mat = self.sample_param_values(sample_size=sample_size)
        assert theta_mat.shape == (sample_size, 2)

        bern_vec = np.random.binomial(n=1, p=p, size=sample_size)
        concat_mat = np.hstack((theta_mat.reshape(-1, 2), bern_vec.reshape(-1, 1)))

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
                                             sample_size=1, true_param=row[:self.d]) if row[2] else
                                         np.abs(self.g_distribution.rvs(size=1)).astype(int).reshape(1, self.d_obs))
        return np.hstack((concat_mat, sample.reshape(-1, 2)))

    def compute_exact_or(self, t0, t1, x_obs):
        f0_val = poisson.pmf(k=x_obs[:, 0].reshape(-1, ), mu=t0[0] + t0[1]) * \
            poisson.pmf(k=x_obs[:, 1].reshape(-1, ), mu=self.tau_param * t0[1])
        f1_val = poisson.pmf(k=x_obs[:, 0].reshape(-1, ), mu=t1[0] + t1[1]) * \
            poisson.pmf(k=x_obs[:, 1].reshape(-1, ), mu=self.tau_param * t1[1])

        return f0_val / f1_val

    def compute_exact_prob(self, x_vec, *args, **kwargs):
        # This function is hard to compute as we would have to integrate over the entire parameter space for the
        # Poisson likelihood, so instead we just return a dummy variable

        return np.ones(x_vec.shape) * 0.99

    def sample_msnh_algo5(self, b_prime, sample_size):
        theta_mat = self.sample_param_values(sample_size=b_prime)
        assert theta_mat.shape == (b_prime, 2)

        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(sample_size=sample_size, true_param=row))
        return theta_mat, sample_mat

    def _nuisance_parameter_func(self, nu_params, x_obs, target_params, clf_odds):

        param_mat = np.hstack((
            np.tile(np.concatenate((target_params.reshape(-1,), nu_params.reshape(-1,))),
                    x_obs.shape[0]).reshape(-1, self.d),
            x_obs.reshape(-1, self.d_obs)
        ))
        pred_mat = clf_odds.predict_proba(param_mat)
        odds_val = -1 * np.sum(np.log(pred_mat[:, 1] / pred_mat[:, 0]))
        return odds_val

    def nuisance_parameter_minimization(self, x_obs, target_params, clf_odds):
        x0_val = np.array([np.nan, 90])[self.nuisance_params_cols]
        res_min = minimize(
            fun=partial(self._nuisance_parameter_func, x_obs=x_obs,
                        target_params=target_params, clf_odds=clf_odds),
            x0=x0_val, method='trust-constr', options={'verbose': 0}, bounds=self.bounds_opt)
        return np.concatenate((np.array(res_min.x), np.array([-1 * res_min.fun])))

    def calculate_nuisance_parameters_over_grid(self, t0_grid, clf_odds, x_obs):
        # We have t0_grid being shape (num_acore_grid, 1) and the nuisance parameter is a single one here
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
        # acore_grid_out = np.hstack((
        #     t0_grid.reshape(-1, 1),
        #     np.tile(self.nuisance_global_param_val, t0_grid.shape[0]).reshape(
        #         t0_grid.shape[0], self.nuisance_global_param_val.shape[0])
        # ))
        acore_grid_out = t0_grid_out.copy()
        return t0_grid_out, acore_grid_out
