import numpy as np
import sys
sys.path.append('..')

from utils.functions import matrix_mesh
from scipy.stats import multivariate_normal, poisson


class SenPoissonLoader:

    def __init__(self, gamma_param=1.0, low_int_reference_background=80, high_int_reference_background=100,
                 low_int_reference_signal=0, high_int_reference_signal=20, out_dir='sen_poisson_2d/', num_grid=21,
                 num_pred_grid=101, *args, **kwargs):
        self.gamma_param = gamma_param
        self.low_int_reference_background = low_int_reference_background
        self.high_int_reference_background = high_int_reference_background
        self.low_int_reference_signal = low_int_reference_signal
        self.high_int_reference_signal = high_int_reference_signal
        self.regen_flag = False
        self.out_directory = out_dir
        self.mean_instrumental = None
        self.cov_instrumental = None
        self.g_distribution = None
        self.b_sample_vec = [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
        self.b_prime_vec = [100, 500, 1000, 5000, 10000, 50000, 100000]
        self.d = 2
        self.d_obs = 2
        self.num_grid = num_grid
        self.grid = matrix_mesh(np.meshgrid(np.linspace(start=self.low_int_reference_background,
                                                        stop=self.high_int_reference_background,
                                                        num=self.num_grid),
                                           np.linspace(start=self.low_int_reference_signal,
                                                       stop=self.high_int_reference_signal,
                                                       num=self.num_grid)))
        self.num_pred_grid = num_pred_grid
        self.pred_grid = matrix_mesh(np.meshgrid(np.linspace(start=self.low_int_reference_background,
                                                             stop=self.high_int_reference_background,
                                                             num=self.num_pred_grid),
                                           np.linspace(start=self.low_int_reference_signal,
                                                       stop=self.high_int_reference_signal,
                                                       num=self.num_pred_grid)))
        self.true_t0 = [98, 10]

    def set_reference_g(self, size_reference):
        background_vec_ref = np.random.uniform(low=self.low_int_reference_background,
                                               high=self.high_int_reference_background,
                                               size=size_reference).reshape(-1, 1)
        mu_vec_ref = np.random.uniform(low=self.low_int_reference_signal,
                                       high=self.high_int_reference_signal,
                                       size=size_reference).reshape(-1, 1)
        theta_mat_ref = np.hstack((background_vec_ref, mu_vec_ref))
        sample_mat_ref = np.apply_along_axis(arr=theta_mat_ref, axis=1,
                                             func1d=lambda row: np.array([
                                                 np.random.poisson(lam=row[0] + row[1], size=1),
                                                 np.random.poisson(
                                                     lam=self.gamma_param * row[0], size=1)]).reshape(-1, 2))
        sample_mat_ref = sample_mat_ref.reshape(-1, 2)
        self.mean_instrumental = np.average(sample_mat_ref, axis=0)
        self.cov_instrumental = np.diag(np.std(sample_mat_ref, axis=0) ** 2)
        self.g_distribution = multivariate_normal(mean=self.mean_instrumental, cov=self.cov_instrumental)

    def set_reference_g_no_sample(self, mean_instrumental, cov_instrumental):
        self.mean_instrumental = mean_instrumental
        self.cov_instrumental = cov_instrumental
        self.g_distribution = multivariate_normal(mean=self.mean_instrumental, cov=self.cov_instrumental)

    def sample_param_values(self, sample_size):
        background_vec = np.random.uniform(low=self.low_int_reference_background,
                                           high=self.high_int_reference_background,
                                           size=sample_size).reshape(-1, 1)
        mu_vec = np.random.uniform(low=self.low_int_reference_signal,
                                   high=self.high_int_reference_signal,
                                   size=sample_size).reshape(-1, 1)
        return np.hstack((background_vec, mu_vec))

    def sample_sim(self, sample_size, true_param):
        first_dim = np.random.poisson(lam=true_param[0] + true_param[1], size=sample_size).reshape(-1, 1)
        second_dim = np.random.poisson(lam=self.gamma_param * true_param[0], size=sample_size).reshape(-1, 1)
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

        sample = np.apply_along_axis(arr=concat_mat, axis=1,
                                     func1d=lambda row: self.sample_sim(
                                         sample_size=1, true_param=row[:2]) if row[2] else
                                     np.abs(self.g_distribution.rvs(size=1)).astype(int).reshape(1, 2))
        return np.hstack((concat_mat, sample.reshape(-1, 2)))

    def compute_exact_or(self, t0, t1, x_obs):
        f0_val = poisson.pmf(k=x_obs[:, 0].reshape(-1, ), mu=t0[0] + t0[1]) * \
            poisson.pmf(k=x_obs[:, 1].reshape(-1, ), mu=self.gamma_param * t0[0])
        f1_val = poisson.pmf(k=x_obs[:, 0].reshape(-1, ), mu=t1[0] + t1[1]) * \
            poisson.pmf(k=x_obs[:, 1].reshape(-1, ), mu=self.gamma_param * t1[0])

        return f0_val / f1_val

    def compute_exact_tau(self, x_obs, t0_val, meshgrid):
        return np.min(np.array([np.sum(np.log(self.compute_exact_or(
            x_obs=x_obs, t0=t0_val, t1=t1))) for t1 in meshgrid]))

    def compute_exact_tau_distr(self, t0_val, meshgrid, n_sampled, sample_size_obs):
        full_obs_sample = self.sample_sim(sample_size=n_sampled * sample_size_obs, true_param=t0_val)
        sample_mat = full_obs_sample.reshape((n_sampled, sample_size_obs, self.d_obs))

        tau_sample = np.array([self.compute_exact_tau(
            x_obs=sample_mat[kk, :, :], t0_val=t0_val, meshgrid=meshgrid) for kk in range(sample_mat.shape[0])])

        return tau_sample

    def sample_msnh_algo5(self, b_prime, sample_size):
        theta_mat = self.sample_param_values(sample_size=b_prime)
        assert theta_mat.shape == (b_prime, 2)

        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(sample_size=sample_size, true_param=row))
        return theta_mat, sample_mat

    def make_grid_over_param_space(self, n_grid):
        return matrix_mesh(np.meshgrid(np.linspace(start=self.low_int_reference_background,
                                                   stop=self.high_int_reference_background,
                                                   num=n_grid),
                                           np.linspace(start=self.low_int_reference_signal,
                                                       stop=self.high_int_reference_signal,
                                                       num=n_grid)))
