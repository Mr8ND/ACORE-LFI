import numpy as np
import sys
sys.path.append('..')

from sklearn import mixture
from scipy.stats import multivariate_normal, uniform



class ToyMVNLoader:

    def __init__(self, d_obs, mean_instrumental=0, std_instrumental=4, low_int=0, high_int=10, true_param=5, true_std=1,
    #def __init__(self, d_obs, mean_instrumental=0, std_instrumental=4, low_int=0, high_int=2, true_param=0, true_std=1,
                 out_dir='toy_mvn/', prior_type='uniform',
                 marginal=False, size_marginal=1000):

        self.mean_instrumental = mean_instrumental #np.repeat(mean_instrumental, d_obs)
        self.std_instrumental = std_instrumental #* np.eye(d_obs)
        self.low_int = low_int
        self.high_int = high_int
        self.prior_type = prior_type
        self.g_distribution = multivariate_normal(np.repeat(self.mean_instrumental, d_obs),
                                                  self.std_instrumental * np.eye(d_obs))
        self.out_directory = out_dir
        self.d = d_obs
        self.d_obs = d_obs
        self.num_grid = 31 #51
        grid_param_t1 = np.linspace(start=self.low_int, stop=self.high_int, num=self.num_grid)
        self.grid = np.repeat(grid_param_t1, d_obs).reshape(-1, d_obs)
        self.num_pred_grid = 21 #41
        t0_grid = np.linspace(start=self.low_int, stop=self.high_int, num=self.num_pred_grid)
        self.pred_grid = np.repeat(t0_grid, d_obs).reshape(-1, d_obs)
        self.true_param = true_param
        self.true_std = true_std

        if marginal:
            self.compute_marginal_reference(size_marginal)

    # (can leave this for later)
    # TODO: change this to MVN, may be tricky to integrate over d dims
    def compute_marginal_reference(self, size_marginal):
        theta_vec_marg = np.random.uniform(low=self.low_int, high=self.high_int, size=size_marginal)
        marginal_sample = np.apply_along_axis(arr=theta_vec_marg.reshape(-1, 1), axis=1,
                                              func1d=lambda row: self._gmm_manual_sampling(
                                                  sample_size=1, mu_param=row)).reshape(-1, self.d_obs)
        mean_mle = np.average(marginal_sample, axis=0)
        std_mle = np.diag(np.std(marginal_sample, axis=0))
        self.mean_instrumental = mean_mle
        self.std_instrumental = std_mle
        self.g_distribution = multivariate_normal(mean=self.mean_instrumental, cov=self.std_instrumental)
    
    # for now, have all theta_i coordinates be the same, to be computationally cheaper
    def sample_sim(self, sample_size, true_param):
        return np.random.normal(loc=true_param, scale=self.true_std, size=sample_size * self.d_obs).reshape(sample_size, self.d_obs)
        
    def sample_param_values(self, sample_size):
        if self.prior_type == 'uniform':
            unique_thetas = np.random.uniform(low=self.low_int, high=self.high_int, size=sample_size)
            return np.repeat(unique_thetas, self.d_obs).reshape(sample_size, self.d_obs)
        elif self.prior_type == 'normal':
            unique_thetas = np.random.normal(loc=self.mean_instrumental, scale=self.std_instrumental, size=sample_size)
            return np.repeat(unique_thetas, self.d_obs).reshape(sample_size, self.d_obs)
        else:
            raise ValueError('The variable prior_type needs to be either uniform or normal.'
                             ' Currently %s' % self.prior_type)
    
    def generate_sample(self, sample_size, p=0.5, **kwargs):
        theta_vec = self.sample_param_values(sample_size=sample_size)
        bern_vec = np.random.binomial(n=1, p=p, size=sample_size)
        concat_mat = np.hstack((theta_vec.reshape(-1, self.d_obs),
                                bern_vec.reshape(-1, 1)))
        
        # since all theta_i coordinates are the same, just take row[0]
        sample = np.apply_along_axis(arr=concat_mat, axis=1,
                                     func1d=lambda row: self.sample_sim(sample_size=1, true_param=row[0]) if row[self.d_obs]
                                     else self.g_distribution.rvs(size=1))
        return np.hstack((concat_mat, sample.reshape(sample_size, self.d_obs)))

    def sample_msnh_algo5(self, b_prime, sample_size):
        theta_mat = self.sample_param_values(sample_size=b_prime).reshape(-1, self.d_obs)
        assert theta_mat.shape == (b_prime, self.d_obs)
        
        # since all theta_i coordinates are the same, just take row[0]
        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(sample_size=sample_size, true_param=row[0]))
        return theta_mat, sample_mat.reshape(b_prime, sample_size, self.d_obs)

    # TODO: get rid of this
    def _gmm_likelihood_manual(self, x, mu):
        term1 = multivariate_normal.pdf(x=x, mean=np.repeat(mu, self.d_obs),
                                        cov=np.diag(np.repeat(self.sigma_mixture[0], self.d_obs)))
        term2 = multivariate_normal.pdf(x=x, mean=np.repeat(-1 * mu, self.d_obs),
                                        cov=np.diag(np.repeat(self.sigma_mixture[1], self.d_obs)))
        return self.mixing_param * (term1 + term2)

    # TODO: change this to MVN
    def compute_exact_or(self, t0, t1, x_obs):
        return self._gmm_likelihood_manual(x=x_obs, mu=t0) / self._gmm_likelihood_manual(x=x_obs, mu=t1)
    
    def compute_exact_prob(self, theta_vec, x_vec, p=0.5):
        x_vec = x_vec.reshape(-1, self.d_obs)
        theta_vec = theta_vec.reshape(-1, self.d_obs)

        #f_val = np.array([self._gmm_likelihood_manual(
        #    x=x, mu=theta_vec_flat[ii]) for ii, x in enumerate(x_vec)]).reshape(-1, )
        
        f_val = np.array([multivariate_normal.pdf(
            x=x, mean=theta_vec[ii], cov=self.true_std * np.eye(self.d_obs)) for ii, x in enumerate(x_vec)]).reshape(-1, )
        
        #f_val = multivariate_normal.pdf(x=x_vec, mean=theta_vec, cov=self.true_std * np.eye(self.d_obs))
        g_val = self.g_distribution.pdf(x=x_vec).reshape(-1, )
        return (f_val * p) / (f_val * p + g_val * (1 - p))

    # TODO: change this to MVN
    # 
    def compute_exact_odds(self, theta_vec, x_vec, p=0.5):
        x_vec = x_vec.reshape(-1, self.d_obs)
        theta_vec_flat = theta_vec.reshape(-1, )

        f_val = np.array([self._gmm_likelihood_manual(
            x=x, mu=theta_vec_flat[ii]) for ii, x in enumerate(x_vec)]).reshape(-1, )
        g_val = self.g_distribution.pdf(x=x_vec).reshape(-1, )
        return (f_val * p) / (g_val * (1 - p))

    # TODO: change this to MVN
    # 
    def compute_exact_likelihood(self, x_obs, true_param):
        return self._gmm_likelihood_manual(x=x_obs, mu=true_param)

    # TODO: change this to MVN
    def compute_exact_lr_simplevsimple(self, x_obs, t0, t1):
        ll_gmm_t0 = np.sum(np.log(self._gmm_likelihood_manual(x=x_obs, mu=t0)))
        ll_gmm_t1 = np.sum(np.log(self._gmm_likelihood_manual(x=x_obs, mu=t1)))

        return ll_gmm_t0 - ll_gmm_t1

    # TODO: change this to MVN
    @staticmethod
    def compute_mle(x_obs):
        gmm_obj = mixture.GaussianMixture(n_components=2, covariance_type='full')
        gmm_obj.fit(X=x_obs)
        return gmm_obj

    # TODO: change this to MVN
    def compute_exact_lr_simplevcomp(self, x_obs, t0, mle):
        ll_gmm_t0 = np.sum(np.log(self._gmm_likelihood_manual(x=x_obs, mu=t0)))
        ll_gmm_t1 = np.sum(mle.score_samples(x_obs))
        return ll_gmm_t0 - ll_gmm_t1

    # TODO: I think we need a d-dimensional grid... or, technically we always have param of form (theta, ..., theta) \in \R^d...
    # but maybe we want to search in \R^d, then also plot confidence set (like Figure 4 HEP example in ICML paper)
    def make_grid_over_param_space(self, n_grid):
        return np.linspace(start=self.low_int, stop=self.high_int, num=n_grid)

    # def create_samples_for_or_loss(self, or_loss_samples):
    #     theta_star_distr = uniform(self.low_int, self.high_int)
    #     theta_distr = uniform(self.low_int, self.high_int)
    #
    #     first_term_params = np.hstack((
    #         theta_distr.rvs(size=or_loss_samples).reshape(-1, 1),
    #         theta_star_distr.rvs(size=or_loss_samples).reshape(-1, 1)))
    #     first_term_sims = np.apply_along_axis(arr=first_term_params, axis=1,
    #                                           func1d=lambda row: self.sample_sim(sample_size=1, true_param=row[1]))
    #     first_term_sample = np.hstack((first_term_params, first_term_sims))
    #
    #     second_term_params = np.hstack((
    #         theta_distr.rvs(size=or_loss_samples).reshape(-1, 1),
    #         theta_star_distr.rvs(size=or_loss_samples).reshape(-1, 1)))
    #     second_term_sims = np.apply_along_axis(arr=second_term_params, axis=1,
    #                                            func1d=lambda row: self.sample_sim(sample_size=1, true_param=row[0]))
    #     second_term_sample = np.hstack((second_term_params, second_term_sims))
    #
    #     return first_term_sample, second_term_sample
