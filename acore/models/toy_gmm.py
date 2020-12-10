import numpy as np
import sys
sys.path.append('..')

from sklearn import mixture
from scipy.stats import norm, uniform


class ToyGMMLoader:

    def __init__(self, mean_instrumental=0, std_instrumental=10, low_int=0, high_int=10, true_param=5.0,
                 out_dir='toy_gmm/', sigma_mixture=(1.0, 1.0), mixing_param=0.5, marginal=False, size_marginal=1000,
                 empirical_marginal=True):

        self.mean_instrumental = mean_instrumental
        self.std_instrumental = std_instrumental
        self.low_int = low_int
        self.high_int = high_int
        self.sigma_mixture = sigma_mixture
        self.mixing_param = mixing_param
        self.g_distribution = norm(loc=self.mean_instrumental, scale=self.std_instrumental)
        self.regen_flag = False
        self.out_directory = out_dir
        self.d = 1
        self.d_obs = 1
        self.num_grid = 51
        self.grid = np.linspace(start=self.low_int, stop=self.high_int, num=self.num_grid)
        self.num_pred_grid = 41
        self.pred_grid = np.linspace(start=self.low_int, stop=self.high_int, num=self.num_pred_grid)
        self.true_param = true_param
        self.empirical_marginal = True

        if marginal:
            self.compute_marginal_reference(size_marginal)

    def compute_marginal_reference(self, size_marginal):
        theta_vec_marg = np.random.uniform(low=self.low_int, high=self.high_int, size=size_marginal)
        marginal_sample = np.apply_along_axis(arr=theta_vec_marg.reshape(-1, 1), axis=1,
                                              func1d=lambda row: self._gmm_manual_sampling(
                                                  sample_size=1, mu_param=row)).reshape(-1, )
        mean_mle = np.average(marginal_sample)
        std_mle = np.std(marginal_sample)
        self.mean_instrumental = mean_mle
        self.std_instrumental = std_mle
        self.g_distribution = norm(loc=mean_mle, scale=std_mle)

    def sample_empirical_marginal(self, sample_size):
        theta_vec_marg = self.sample_param_values(sample_size=sample_size)
        return np.apply_along_axis(arr=theta_vec_marg.reshape(-1, self.d), axis=1,
                                   func1d=lambda row: self.sample_sim(
                                       sample_size=1, true_param=row)).reshape(-1, self.d_obs)

    def _gmm_manual_sampling(self, sample_size, mu_param):
        cluster = np.random.binomial(n=1, p=self.mixing_param, size=sample_size)
        means = np.take([-1 * mu_param, mu_param], cluster)
        sigmas = np.take([self.sigma_mixture[0], self.sigma_mixture[1]], cluster)
        return np.random.normal(loc=means, scale=sigmas, size=sample_size)

    def sample_sim(self, sample_size, true_param):
        return self._gmm_manual_sampling(sample_size=sample_size, mu_param=true_param)

    def sample_param_values(self, sample_size):
        return np.random.uniform(low=self.low_int, high=self.high_int, size=sample_size)

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
                                         func1d=lambda row: self.sample_sim(sample_size=1, true_param=row[0]) if
                                         row[1] else self.g_distribution.rvs(size=1))
        return np.hstack((concat_mat, sample.reshape(-1, self.d_obs)))

    def sample_msnh_algo5(self, b_prime, sample_size):
        theta_mat = self.sample_param_values(sample_size=b_prime).reshape(-1, 1)
        assert theta_mat.shape == (b_prime, 1)

        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(sample_size=sample_size, true_param=row))
        return theta_mat, sample_mat

    @staticmethod
    def _normal_likelihood(x, mu, sigma):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

    def _gmm_likelihood_manual(self, x, mu):
        return self.mixing_param * self._normal_likelihood(x=x, mu=mu, sigma=self.sigma_mixture[0]) + \
               (1 - self.mixing_param) * self._normal_likelihood(x=x, mu=-1 * mu, sigma=self.sigma_mixture[1])

    def compute_exact_or(self, t0, t1, x_obs):
        return self._gmm_likelihood_manual(x=x_obs, mu=t0) / self._gmm_likelihood_manual(x=x_obs, mu=t1)

    def compute_exact_prob(self, theta_vec, x_vec, p=0.5):
        x_vec_flat = x_vec.reshape(-1, )
        theta_vec_flat = theta_vec.reshape(-1, )

        f_val = np.array([self._gmm_likelihood_manual(
            x=x, mu=theta_vec_flat[ii]) for ii, x in enumerate(x_vec_flat)]).reshape(-1, )
        g_val = self.g_distribution.pdf(x=x_vec).reshape(-1, )
        return (f_val * p) / (f_val * p + g_val * (1 - p))

    def compute_exact_odds(self, theta_vec, x_vec, p=0.5):
        x_vec_flat = x_vec.reshape(-1, )
        theta_vec_flat = theta_vec.reshape(-1, )

        f_val = np.array([self._gmm_likelihood_manual(
            x=x, mu=theta_vec_flat[ii]) for ii, x in enumerate(x_vec_flat)]).reshape(-1, )
        g_val = self.g_distribution.pdf(x=x_vec)
        return (f_val * p) / (g_val * (1 - p))

    def compute_exact_likelihood(self, x_obs, true_param):
        return self._gmm_likelihood_manual(x=x_obs, mu=true_param)

    def compute_exact_lr_simplevsimple(self, x_obs, t0, t1):
        ll_gmm_t0 = np.sum(np.log(self._gmm_likelihood_manual(x=x_obs, mu=t0)))
        ll_gmm_t1 = np.sum(np.log(self._gmm_likelihood_manual(x=x_obs, mu=t1)))

        return ll_gmm_t0 - ll_gmm_t1

    @staticmethod
    def compute_mle(x_obs):
        gmm_obj = mixture.GaussianMixture(n_components=2, covariance_type='full')
        gmm_obj.fit(X=x_obs)
        return gmm_obj

    def compute_exact_lr_simplevcomp(self, x_obs, t0, mle):
        ll_gmm_t0 = np.sum(np.log(self._gmm_likelihood_manual(x=x_obs, mu=t0)))
        ll_gmm_t1 = np.sum(mle.score_samples(x_obs))
        return ll_gmm_t0 - ll_gmm_t1

    def make_grid_over_param_space(self, n_grid):
        return np.linspace(start=self.low_int, stop=self.high_int, num=n_grid)

    def create_samples_for_or_loss(self, or_loss_samples):
        theta_star_distr = uniform(self.low_int, self.high_int)
        theta_distr = uniform(self.low_int, self.high_int)

        first_term_params = np.hstack((
            theta_distr.rvs(size=or_loss_samples).reshape(-1, 1),
            theta_star_distr.rvs(size=or_loss_samples).reshape(-1, 1)))
        first_term_sims = np.apply_along_axis(arr=first_term_params, axis=1,
                                              func1d=lambda row: self.sample_sim(sample_size=1, true_param=row[1]))
        first_term_sample = np.hstack((first_term_params, first_term_sims))

        second_term_params = np.hstack((
            theta_distr.rvs(size=or_loss_samples).reshape(-1, 1),
            theta_star_distr.rvs(size=or_loss_samples).reshape(-1, 1)))
        second_term_sims = np.apply_along_axis(arr=second_term_params, axis=1,
                                               func1d=lambda row: self.sample_sim(sample_size=1, true_param=row[0]))
        second_term_sample = np.hstack((second_term_params, second_term_sims))

        return first_term_sample, second_term_sample
