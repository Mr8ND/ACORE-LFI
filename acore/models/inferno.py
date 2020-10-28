import numpy as np
import math
import sys
sys.path.append('..')

from utils.functions import tensor_4d_mesh
from scipy.stats import multivariate_normal, poisson, rv_discrete, expon
from scipy.linalg import sqrtm
from scipy.optimize import Bounds, minimize


class InfernoToyLoader:

    def __init__(self, s_param=50, r_param=0.0, lambda_param=3.0, b_param=1000, benchmark=None,
                 s_low=0, s_high=100, r_low=-5, r_high=5, lambda_low=0, lambda_high=10, b_low=700, b_high=1300,
                 out_dir='inferno_toy/', num_grid=21, num_pred_grid=101, preset_grid_n=21, *args, **kwargs):

        self.true_s = s_param
        self.s_low = s_low
        self.s_high = s_high
        self.sigmas_mat = [np.diag([5, 9]), np.diag([1, 1])]

        if benchmark is not None:
            if benchmark not in [0, 1, 2, 3, 4]:
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

        active_params_tuple = [True, False, False, False]
        low_bounds, high_bounds = [], []
        for idx, (low, high) in enumerate([(self.r_low, self.r_high), (self.lambda_low, self.lambda_high),
                                           (self.b_low, self.b_high)]):
            if low != high:
                active_params_tuple[idx + 1] = True
                low_bounds.append(low)
                high_bounds.append(high)

        self.active_params = sum(active_params_tuple)
        self.nuisance_parameters = self.active_params > 1
        self.active_params_columns = np.where(active_params_tuple)[0]
        self.nuisance_parameters_cols = np.where(active_params_tuple)[0][1:] if self.nuisance_parameters else None

        print(self.active_params)
        print(self.active_params_columns)
        print(self.nuisance_parameters)
        print(self.nuisance_parameters_cols)
        print(low_bounds)
        print(high_bounds)
        print(stop)

        self.regen_flag = False
        self.out_directory = out_dir

        self.mean_instrumental = np.array([1, 1, 2])
        self.cov_instrumental = np.diag([5, 10, 5])
        self.g_distribution = multivariate_normal(mean=self.mean_instrumental, cov=self.cov_instrumental)

        self.b_sample_vec = [50, 100, 500, 1000, 5000, 10000, 50000, 100000, 200000]
        self.b_prime_vec = [100, 500, 1000, 5000, 10000, 50000, 100000]
        self.d = 4
        self.d_obs = 3

        self.num_grid = num_grid if self.active_params <= 2 else preset_grid_n
        self.grid = np.unique(
            tensor_4d_mesh(np.meshgrid(np.linspace(start=self.s_low, stop=self.s_high, num=self.num_grid),
                           np.linspace(start=self.r_low, stop=self.r_high, num=self.num_grid),
                           np.linspace(start=self.lambda_low, stop=self.lambda_high,
                                       num=self.num_grid),
                           np.linspace(start=self.b_low, stop=self.b_high, num=self.num_grid))), axis=0)

        self.num_pred_grid = num_pred_grid if self.active_params <= 2 else preset_grid_n
        self.pred_grid = np.unique(
            tensor_4d_mesh(np.meshgrid(np.linspace(start=self.s_low, stop=self.s_high, num=self.num_grid),
                                       np.linspace(start=self.r_low, stop=self.r_high, num=self.num_grid),
                                       np.linspace(start=self.lambda_low, stop=self.lambda_high,
                                                   num=self.num_grid),
                                       np.linspace(start=self.b_low, stop=self.b_high, num=self.num_grid))), axis=0)
        self.true_t0 = [self.true_s, self.true_r, self.true_lambda, self.true_b]

    def select_active_parameters(self, param_mat):
        return param_mat[:, self.active_params_columns]

    def _nuisance_parameter_func(self, nu_params, x_obs, target_params, fixed_params, clf_odds):
        param_mat = np.hstack((
            np.tile(np.concatenate((target_params, nu_params, fixed_params)), x_obs.shape[0]).reshape(-1, self.d),
            x_obs.reshape(-1, self.d_obs)
        ))
        pred_mat = clf_odds.predict_proba(param_mat)
        return -1 * (np.prod(pred_mat[:, 1] / pred_mat[:, 0]))

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

    def sample_param_values(self, sample_size):
        signal_vec = np.random.uniform(low=self.s_low, high=self.s_high, size=sample_size).reshape(-1, 1)
        r_vec = np.random.uniform(low=self.r_low, high=self.r_high, size=sample_size).reshape(-1, 1)
        lambda_vec = np.random.uniform(low=self.lambda_low, high=self.lambda_high, size=sample_size).reshape(-1, 1)
        background_vec = np.random.uniform(low=self.b_low, high=self.b_high, size=sample_size).reshape(-1, 1)
        return np.hstack((signal_vec, r_vec, lambda_vec, background_vec))

    def sample_sim(self, sample_size, true_param):
        mixing_param = true_param[3] / (true_param[0] + true_param[3])
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

    def compute_exact_prob(self, theta_mat, x_mat, p=0.5):
        x_mat = x_mat.reshape(-1, self.d_obs)
        theta_mat = theta_mat.reshape(-1, self.d)

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

    def compute_exact_odds(self, theta_mat, x_mat, p=0.5):
        x_mat = x_mat.reshape(-1, self.d_obs)
        theta_mat = theta_mat.reshape(-1, self.d)

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
        assert theta_mat.shape == (b_prime, self.d)

        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(sample_size=sample_size, true_param=row))
        return theta_mat, sample_mat

    def make_grid_over_param_space(self, n_grid):
        grid = tensor_4d_mesh(np.meshgrid(np.linspace(start=self.s_low, stop=self.s_high, num=n_grid),
                                          np.linspace(start=self.r_low, stop=self.r_high, num=n_grid),
                                          np.linspace(start=self.lambda_low, stop=self.lambda_high,
                                                      num=n_grid),
                                          np.linspace(start=self.b_low, stop=self.b_high, num=n_grid)))
        # active_grid = self.select_active_parameters(grid)
        return grid


if __name__ == '__main__':

    obj = InfernoToyLoader(benchmark=3)

    sample_params = obj.sample_param_values(100)
    samples = obj.generate_sample(sample_size=10)
    x_mat_sampled = samples[:, 5:]
    theta_mat_sampled = samples[:, :4]
    print(theta_mat_sampled.shape, x_mat_sampled.shape)

    print(obj.compute_exact_or(x_obs=x_mat_sampled, t0=theta_mat_sampled[0, :], t1=theta_mat_sampled[1, :]))