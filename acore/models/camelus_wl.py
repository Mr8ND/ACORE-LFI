import pickle
import numpy as np
import random
from scipy.stats import multivariate_normal
from shapely.geometry import Polygon, Point


class CamelusSimLoader:
    # Simulations are actually stored as lists, so that it is possible
    # to pop from them

    def __init__(self, flnm='data/linc_full_dict_data.pkl', true_index=111,
                 out_dir='camelus_linc/', empirical_marginal=True, num_acore_grid=100, *args, **kwargs):
        linc_data_dict = pickle.load(open(flnm, 'rb'))
        self.data_dict = linc_data_dict
        self.grid = linc_data_dict['grid']

        self.acore_grid = self.get_random_point_in_polygon(poly=Polygon(self.grid), size=num_acore_grid)
        self.pred_grid = self.grid
        self.true_t0 = np.round(self.grid[true_index, ], 2)
        self.idx_row_true_param = true_index
        self.true_param = self.true_t0

        self.empirical_marginal = empirical_marginal
        self.mean_instrumental = None
        self.cov_instrumental = None
        self.g_distribution = None
        self.d = 2
        self.d_obs = 7
        self.regen_flag = True
        self.out_directory = out_dir
        self.b_sample_vec = [50, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5]
        self.b_prime_vec = [100, 500, 1000, 5000, 10000, 50000, 100000]
        self.nuisance_flag = False

    def get_random_point_in_polygon(self, poly, size):
        minx, miny, maxx, maxy = poly.bounds
        points_list = []
        counter = 0
        while counter < size:
            p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
            if poly.contains(p):
                points_list.append([p.x, p.y])
                counter += 1
        return np.array(points_list).reshape(-1, 2)

    def set_reference_g(self, size_reference):
        sim_sample_reference = self.sample_sim_overall(size_reference)
        mean_instrumental = np.average(sim_sample_reference, axis=0)
        cov_instrumental = np.diag((np.std(sim_sample_reference, axis=0) ** 2))

        # Multivariate normal (reference)
        self.mean_instrumental = mean_instrumental
        self.cov_instrumental = cov_instrumental
        self.g_distribution = multivariate_normal(mean=self.mean_instrumental, cov=self.cov_instrumental)

    def set_reference_g_no_sample(self, mean_instrumental, cov_instrumental):
        self.mean_instrumental = mean_instrumental
        self.cov_instrumental = cov_instrumental
        self.g_distribution = multivariate_normal(mean=self.mean_instrumental, cov=self.cov_instrumental)

    def sample_sim(self, sample_size, true_param):
        t0_tuple = (round(true_param[0], 2), round(true_param[1], 2))
        if t0_tuple not in self.data_dict:
            raise ValueError('Parameter Combo (%s, %s) not in data' % (t0_tuple[0], t0_tuple[1]))
        if len(self.data_dict[t0_tuple]) < sample_size:
            raise ValueError('Not enough simulations available. Only %s left' % len(self.data_dict[t0_tuple]))
        return np.array([self.data_dict[t0_tuple].pop() for _ in range(sample_size)])

    def sample_sim_overall(self, sample_size):
        random_t0 = self.sample_param_values(sample_size)
        return np.array([self.sample_sim(1, t0_val).reshape(-1, ) for t0_val in random_t0])

    def sample_param_values(self, sample_size):
        return self.grid[np.random.choice(self.grid.shape[0], size=sample_size)]

    def sample_empirical_marginal(self, sample_size):
        theta_vec_marg = self.sample_param_values(sample_size=sample_size)
        return np.apply_along_axis(arr=theta_vec_marg.reshape(-1, self.d), axis=1,
                                   func1d=lambda row: self.sample_sim(
                                       sample_size=1, true_param=row)).reshape(-1, self.d_obs)

    def sample_sim_check(self, sample_size, n):
        random_t0 = self.sample_param_values(sample_size)
        sample_mat = np.array([self.sample_sim(n, t0_val).reshape(-1, n) for t0_val in random_t0])
        return random_t0, sample_mat

    def generate_sample(self, sample_size, p=0.5, marginal=False):
        # Sampling the parameter values
        theta_mat = self.sample_param_values(sample_size=sample_size)
        assert theta_mat.shape == (sample_size, 2)

        # Sampling the bernoulli variable related to whether it's going to sample
        # from the simulation or from the reference distribution
        bern_vec = np.random.binomial(n=1, p=p, size=sample_size).reshape(sample_size, 1)
        concat_mat = np.hstack((theta_mat, bern_vec))

        if self.empirical_marginal:
            sample = np.apply_along_axis(arr=concat_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(
                                             sample_size=1, true_param=row[:self.d]) if row[self.d]
                                         else self.sample_empirical_marginal(sample_size=1)).reshape(-1, 7)
        else:
            sample = np.apply_along_axis(arr=concat_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(
                                             sample_size=1, true_param=row[:2]) if row[2] else
                                         np.abs(self.g_distribution.rvs(size=1)).astype(int)).reshape(-1, 7)
        return np.hstack((concat_mat, sample))

    def sample_msnh_algo5(self, b_prime, sample_size):
        theta_mat = self.sample_param_values(sample_size=b_prime)
        assert theta_mat.shape == (b_prime, 2)

        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(
                                             sample_size=sample_size, true_param=row))
        return theta_mat, sample_mat

    def compute_exact_tau(self, x_obs, t0_val, meshgrid):
        raise NotImplementedError('True Likelihood not known for this model.')

    def compute_exact_tau_distr(self, t0_val, meshgrid, n_sampled, sample_size_obs):
        raise NotImplementedError('True Likelihood not known for this model.')

    def make_grid_over_param_space(self, n_grid):
        raise NotImplementedError('Grid is fixed on this model for now.')

    def compute_exact_prob(self, theta_vec, x_vec):
        return np.array([0.9 for el in theta_vec])
