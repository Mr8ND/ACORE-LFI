import numpy as np
from math import ceil, sqrt, pi
from scipy.integrate import quad
import sys

sys.path.append('..')

from scipy.stats import multivariate_normal, uniform, norm
from scipy.stats import binom
from scipy.optimize import Bounds
from scipy.special import erf


class ToyMVG:
    
    def __init__(self, 
                 observed_dims=2, 
                 true_param=0.0, true_std=1.0, 
                 param_grid_width=10, grid_sample_size=1000,
                 prior_type='uniform', normal_mean_prior=0.0, normal_std_prior=2.0,
                 empirical_marginal=True):
        
        self.d = observed_dims
        self.observed_dims = observed_dims
        
        # without loss of generality, we only consider points with equal coordinates, for simplicity
        self.true_param = np.repeat(true_param, self.d)
        self.param_grid_width = param_grid_width
        self.low_int = true_param - (param_grid_width/2)
        self.high_int = true_param + (param_grid_width/2)
        assert (self.high_int - self.low_int) == param_grid_width
        
        self.true_cov = true_std * np.eye(observed_dims)

        self.prior_type = prior_type
        if prior_type == 'uniform':
            self.prior_distribution = uniform(loc=self.low_int, scale=(self.high_int - self.low_int))
        elif prior_type == 'normal':
            self.prior_distribution = norm(loc=normal_mean_prior, scale=normal_std_prior ** 2)
        else:
            raise ValueError('The variable prior_type needs to be either uniform or normal.'
                             ' Currently %s' % prior_type)
            
        if self.d == 1:
            self.param_grid = np.linspace(self.low_int, self.high_int, num=grid_sample_size)
            self.t0_grid_granularity = grid_sample_size
        elif self.d == 2: 
            a = np.linspace(self.low_int, self.high_int, num=grid_sample_size)
            # 2-dimensional grid of (grid_sample_size X grid_sample_size) points
            self.param_grid = np.transpose([np.tile(a, len(a)), np.repeat(a, len(a))])
            self.t0_grid_granularity = grid_sample_size**2
        else:
            # easier to sample from a d-dimensional uniform for d > 2
            self.param_grid = np.random.uniform(low=self.low_int, high=self.high_int, 
                                                size=grid_sample_size * self.d).reshape(-1, self.d)
            self.t0_grid_granularity = grid_sample_size
            
        #if self.true_param not in self.param_grid:
        #    self.param_grid = np.append(self.param_grid.reshape(-1, self.d),
        #                                self.true_param.reshape(-1,self.d), axis=0)
        #    self.t0_grid_granularity += 1
        
        if not empirical_marginal:
            raise NotImplementedError("We are only using empirical marginal for now.")
        self.empirical_marginal = True
    
    def sample_sim(self, sample_size, true_param):
        return multivariate_normal(mean=true_param,
                                   cov=self.true_cov).rvs(sample_size).reshape(sample_size, self.observed_dims)

    def sample_param_values(self, sample_size):
        unique_theta = self.prior_distribution.rvs(size=sample_size * self.d)
        return np.clip(unique_theta.reshape(sample_size, self.d), a_min=self.low_int, a_max=self.high_int)
    
    def sample_param_values_qr(self, sample_size):
        # enlarge support to make sure observed sample is not on the boundaries
        if self.prior_type == "uniform":
            qr_prior_distribution = uniform(loc=self.true_param[0] - self.param_grid_width,
                                            scale=(2*self.param_grid_width))
        else:
            raise NotImplementedError
        unique_theta = qr_prior_distribution.rvs(size=sample_size * self.d)
        return unique_theta.reshape(sample_size, self.d)
    
    def sample_empirical_marginal(self, sample_size):
        theta_vec_marg = self.sample_param_values(sample_size=sample_size)
        return np.apply_along_axis(arr=theta_vec_marg.reshape(-1, self.d), axis=1,
                                   func1d=lambda row: self.sample_sim(
                                   sample_size=1, true_param=row)).reshape(-1, self.observed_dims)
    
    def generate_sample(self, sample_size, p=0.5, **kwargs):
        theta_vec = self.sample_param_values(sample_size=sample_size)
        bern_vec = np.random.binomial(n=1, p=p, size=sample_size)
        concat_mat = np.hstack((
            bern_vec.reshape(-1, 1),
            theta_vec.reshape(-1, self.d)
        ))

        sample = np.apply_along_axis(arr=concat_mat, axis=1,
                                     func1d=lambda row: self.sample_sim(sample_size=1, 
                                                                        true_param=row[1:1+self.d]) 
                                     if row[0] else self.sample_empirical_marginal(sample_size=1))
        
        return np.hstack((concat_mat, sample.reshape(sample_size, self.observed_dims)))

    def sample_msnh(self, b_prime, obs_sample_size):
        theta_mat = self.sample_param_values(sample_size=b_prime).reshape(-1, self.d)
        assert theta_mat.shape == (b_prime, self.d)

        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(sample_size=obs_sample_size, 
                                                                            true_param=row[:self.d]))
        return theta_mat, sample_mat.reshape(b_prime, obs_sample_size, self.observed_dims)
    
    def generate_observed_sample(self, n_samples, obs_sample_size, theta_mat=None):
        if theta_mat is None:
            theta_mat = self.sample_param_values(sample_size=n_samples).reshape(-1, self.d)
        assert theta_mat.shape == (n_samples, self.d)

        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(sample_size=obs_sample_size, 
                                                                            true_param=row[:self.d]))
        return theta_mat, sample_mat.reshape(n_samples, obs_sample_size, self.observed_dims)
    
    @staticmethod
    def compute_mle(x_obs):
        return np.mean(x_obs, axis=0)
    
    @staticmethod
    def clopper_pearson_interval(n, x, alpha):
        theta_grid = np.linspace(0, 1, 1000)
        left = np.min([p for p in theta_grid if binom.cdf(x, n, p) > alpha/2])
        right = np.max([p for p in theta_grid if (1-binom.cdf(x, n, p)) > alpha/2])
        return left, right
    
    def _compute_exact_lr_simplevcomp(self, t0, mle, obs_sample_size):
        return (-1)*(obs_sample_size/2)*(np.linalg.norm((mle - t0).reshape(-1, self.d), ord=2, axis=1)**2)
    
    def _compute_multivariate_normal_pdf(self, x, mu, cov=None, obs_sample_size=1):
        if cov is None:
            cov=np.eye(self.d)/obs_sample_size
        return multivariate_normal(mean=mu, cov=cov).pdf(x)
    
    def _compute_marginal_pdf(self, x_obs):
        '''
        In this calculation we are assuming that the covariance matrix is diagonal with all entries being equal, so
        we only consider the first element for every point.
        '''
        density = 0.5 * (erf((self.high_int - x_obs) / (np.sqrt(2) * self.true_cov[0, 0])) -
                         erf((self.low_int - x_obs) / (np.sqrt(2) * self.true_cov[0, 0])))
        return np.prod(density)
    
    def compute_exact_odds(self, theta_vec, x_vec, p=0.5):
        x_vec = x_vec.reshape(-1, self.observed_dims)
        theta_vec = theta_vec.reshape(-1, self.d)

        f_val = np.array([self._compute_multivariate_normal_pdf(
            x=x, mu=theta_vec[ii, :]) for ii, x in enumerate(x_vec)]).reshape(-1, )

        if self.empirical_marginal:
            g_val = np.array([self._compute_marginal_pdf(x_obs=x_obs) for x_obs in x_vec]).reshape(-1, )
        else:
            g_val = self.g_distribution.pdf(x=x_vec).reshape(-1, )
        return (f_val * p) / (g_val * (1 - p))
    
    def _compute_marginal_bf_denominator(self, x_obs, mu, prior_type='uniform', obs_sample_size=1):
        '''
        In this calculation we are assuming that the covariance matrix is the Identity matrix.
        '''
        if prior_type == 'uniform':
            unif_distr = (1 / ((mu + (self.param_grid_width/2)) - (mu - (self.param_grid_width/2)))) ** self.observed_dims
            density = 0.5 * (erf(((mu + (self.param_grid_width/2)) - x_obs)/np.sqrt(2*obs_sample_size)) - erf(((mu - (self.param_grid_width/2)) - x_obs)/np.sqrt(2*obs_sample_size)))
            assert len(density) == self.observed_dims
            denominator = unif_distr * np.prod(density)
        else:
            raise ValueError("The prior type needs to be 'uniform'. Currently %s" % self.prior_type)
        return denominator
    
    def _compute_marginal_bf_denominator_old(self, x_obs, prior_type='uniform'):
        '''
        In this calculation we are assuming that the covariance matrix is diagonal with all entries being equal, so
        we only consider the first element for every point.
        '''
        if prior_type == 'uniform':
            unif_distr = (1 / (self.high_int - self.low_int)) ** self.observed_dims
            density = 0.5 * unif_distr * (erf((self.high_int - x_obs) / (np.sqrt(2) * self.true_cov[0, 0])) -
                             erf((self.low_int - x_obs) / (np.sqrt(2) * self.true_cov[0, 0])))
        else:
            raise ValueError("The prior type needs to be 'uniform'. Currently %s" % self.prior_type)
        return np.prod(density)

    def compute_exact_bayes_factor_with_marginal(self, x, mu, cov=None, obs_sample_size=1):
        '''
        In this calculation we are assuming that the covariance matrix is the Identity matrix.
        '''
        if self.prior_type == 'uniform':
            x_vec = x.reshape(-1, self.observed_dims)
            theta_vec = mu.reshape(-1, self.d)
            
            f_val = np.array([self._compute_multivariate_normal_pdf(
                x=x_val, mu=theta_vec[ii, :], cov=cov, obs_sample_size=obs_sample_size) for ii, x_val in enumerate(x_vec)]).reshape(-1, )
            g_val = np.array([self._compute_marginal_bf_denominator(x_val, mu, prior_type='uniform', obs_sample_size=obs_sample_size) for x_val in x_vec]
                             ).reshape(-1, )
        else:
            raise ValueError("The prior type needs to be 'uniform'. Currently %s" % self.prior_type)
        return f_val / g_val   
    
    def compute_exact_bayes_factor_with_marginal_old(self, x, mu, cov=None, obs_sample_size=1):
        '''
        In this calculation we are assuming that the covariance matrix is the Identity matrix.
        '''
        if self.prior_type == 'uniform':
            x_vec = x.reshape(-1, self.observed_dims)
            theta_vec = mu.reshape(-1, self.d)
            
            f_val = np.array([self._compute_multivariate_normal_pdf(
                x=x_val, mu=theta_vec[ii, :], cov=cov, obs_sample_size=obs_sample_size) for ii, x_val in enumerate(x_vec)]).reshape(-1, )
            g_val = np.array([self._compute_marginal_bf_denominator_old(x_obs=x_val, prior_type='uniform') for x_val in x_vec]
                             ).reshape(-1, )
        else:
            raise ValueError("The prior type needs to be 'uniform'. Currently %s" % self.prior_type)
        return f_val / g_val  

    
    
def manually_compute_normal_pdf(x, mean, cov, d):
    
    x = x.reshape(1, d)
    mean = mean.reshape(1, d)
    cov = cov.reshape(d, d)
    
    normalization_const = ((2*np.pi)**(-d/2))*(np.linalg.det(cov)**(-1/2))
    return (normalization_const * np.exp((-1/2)*((x-mean) @ np.linalg.inv(cov) @ (x-mean).T))).item()    
    


"""                
class ToyMVNMultiDIsotropicLoader:

    def __init__(self, observed_dims=2, mean_instrumental=0.0, std_instrumental=4.0, low_int=-5.0, high_int=5.0,
                 true_param=0.0, true_std=1.0, mean_prior=0.0, std_prior=2.0, grid_sample_size=1000,
                 out_dir='mvg_example/', prior_type='uniform', diagnostic_flag=False,
                 marginal=False, size_marginal=5000, empirical_marginal=True, **kwargs):

        if true_param != 0.0 and not diagnostic_flag:
            raise ValueError('This class does not work outside of the case when the true parameter is the origin.')

        self.out_directory = out_dir
        self.d = observed_dims
        self.observed_dims = observed_dims
        self.low_int = low_int
        self.high_int = high_int
        self.bounds_opt = Bounds([self.low_int] * self.d, [self.high_int] * self.d)
        
        self.prior_type = prior_type
        if prior_type == 'uniform':
            self.prior_distribution = uniform(loc=self.low_int, scale=(self.high_int - self.low_int))
        elif prior_type == 'normal':
            self.prior_distribution = norm(loc=mean_prior, scale=std_prior ** 2)
        else:
            raise ValueError('The variable prior_type needs to be either uniform or normal.'
                             ' Currently %s' % prior_type)

        self.mean_instrumental = np.repeat(mean_instrumental, self.observed_dims) if isinstance(mean_instrumental, float) \
            else mean_instrumental
        self.cov_instrumental = std_instrumental * np.eye(self.observed_dims) if isinstance(std_instrumental, float) \
            else std_instrumental
        self.g_distribution = multivariate_normal(self.mean_instrumental, self.cov_instrumental)

        self.true_param = np.repeat(true_param, self.d) if isinstance(true_param, float) \
            else np.array(true_param)
        if self.true_param.shape[0] != self.d:
            raise ValueError('The true_param variable passed is not of the right dimension. '
                             'Currently %s, while it has to be %s.' % (self.true_param.shape[0], self.d))
        self.true_cov = true_std * np.eye(observed_dims) if isinstance(true_std, float) else true_std

        self.nuisance_flag = False
        if marginal:
            self.compute_marginal_reference(size_marginal)
        self.empirical_marginal = empirical_marginal
            
        self.num_pred_grid = grid_sample_size
        if self.d == 1:
            self.param_grid = np.linspace(self.true_param[0]-5, self.true_param[0]+5, self.num_pred_grid)
        elif self.d == 2:  # TODO: this assumes true_param = (x,y) with x=y
            a = np.linspace(self.true_param[0]-5, self.true_param[0]+5, num=self.num_pred_grid)
            # 2-dimensional grid of (grid_sample_size X grid_sample_size) points
            self.param_grid = np.transpose([np.tile(a, len(a)), np.repeat(a, len(a))])
        else:
            # easier to sample from a d-dimensional uniform for d > 2
            self.param_grid = np.random.uniform(low=self.low_int, high=self.high_int, 
                                                size=self.num_pred_grid * self.d).reshape(-1, self.d)
        if self.true_param not in self.param_grid:
                self.param_grid = np.append(self.param_grid, self.true_param.reshape(-1,self.d), axis=0)
        
        # prediction grid we care about is the null hypothesis plus all the values on the 45 degree line
        # in d dimension. We are basically looking for power in the direction outside the null hypothesis
        one_d_isotropic_line = np.sqrt(np.linspace(0, 30, 31)).reshape(-1, 1) / np.sqrt(self.d)
        self.pred_grid = np.repeat(one_d_isotropic_line, self.d, axis=1)
        self.idx_row_true_param = 0

        # b analysis values
        self.b_sample_vec = [50, 100, 500, 1000, 5000, 10000, 50000, 100000]
        self.regen_flag = False

    def sample_sim(self, sample_size, true_param):
        return multivariate_normal(mean=true_param, cov=self.true_cov).rvs(sample_size).reshape(sample_size, self.observed_dims)

    def sample_param_values(self, sample_size):
        unique_theta = self.prior_distribution.rvs(size=sample_size * self.d)
        return np.clip(unique_theta.reshape(sample_size, self.d), a_min=self.low_int, a_max=self.high_int)

    def compute_marginal_reference(self, size_marginal):
        theta_vec_marg = self.sample_param_values(sample_size=size_marginal)
        marginal_sample = np.apply_along_axis(arr=theta_vec_marg.reshape(-1, self.d), axis=1,
                                              func1d=lambda row: self.sample_sim(
                                                  sample_size=1, true_param=row)).reshape(-1, self.observed_dims)

        self.mean_instrumental = np.average(marginal_sample, axis=0)
        self.cov_instrumental = np.diag(np.var(marginal_sample, axis=0))
        self.g_distribution = multivariate_normal(mean=self.mean_instrumental, cov=self.cov_instrumental)

    def sample_empirical_marginal(self, sample_size):
        theta_vec_marg = self.sample_param_values(sample_size=sample_size)
        return np.apply_along_axis(arr=theta_vec_marg.reshape(-1, self.d), axis=1,
                                   func1d=lambda row: self.sample_sim(
                                       sample_size=1, true_param=row)).reshape(-1, self.observed_dims)
    
    def generate_sample(self, sample_size, p=0.5, **kwargs):
        theta_vec = self.sample_param_values(sample_size=sample_size)
        bern_vec = np.random.binomial(n=1, p=p, size=sample_size)
        concat_mat = np.hstack((bern_vec.reshape(-1, 1), 
                                theta_vec.reshape(-1, self.d)
                               ))

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
        return np.hstack((concat_mat, sample.reshape(sample_size, self.observed_dims)))

    def sample_msnh(self, b_prime, sample_size):
        theta_mat = self.sample_param_values(sample_size=b_prime).reshape(-1, self.d)
        assert theta_mat.shape == (b_prime, self.d)

        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: self.sample_sim(
                                             sample_size=sample_size, true_param=row[:self.d]))
        return theta_mat, sample_mat.reshape(b_prime, sample_size, self.observed_dims)

    def _compute_multivariate_normal_pdf(self, x, mu):
        # Let's take advantage of the unit covariance matrix
        return np.exp(-0.5 * np.dot(x-mu, x-mu)) * (pi ** (self.observed_dims/2))

    def compute_exact_or(self, t0, t1, x_obs):
        return self._compute_multivariate_normal_pdf(
            x=x_obs, mu=t0) / self._compute_multivariate_normal_pdf(x=x_obs, mu=t1)

    def compute_exact_prob(self, theta_vec, x_vec, p=0.5):
        x_vec = x_vec.reshape(-1, self.observed_dims)
        theta_vec = theta_vec.reshape(-1, self.d)

        f_val = np.array([self._compute_multivariate_normal_pdf(
            x=x, mu=theta_vec[ii, :]) for ii, x in enumerate(x_vec)]).reshape(-1, )

        if self.empirical_marginal:
            g_val = np.array([self._compute_marginal_pdf(x_obs=x_obs) for x_obs in x_vec]).reshape(-1,)
        else:
            g_val = self.g_distribution.pdf(x=x_vec).reshape(-1, )

        return (f_val * p) / (f_val * p + g_val * (1 - p))

    def compute_exact_odds(self, theta_vec, x_vec, p=0.5):
        x_vec = x_vec.reshape(-1, self.observed_dims)
        theta_vec = theta_vec.reshape(-1, self.d)

        f_val = np.array([self._compute_multivariate_normal_pdf(
            x=x, mu=theta_vec[ii, :]) for ii, x in enumerate(x_vec)]).reshape(-1, )

        if self.empirical_marginal:
            g_val = np.array([self._compute_marginal_pdf(x_obs=x_obs) for x_obs in x_vec]).reshape(-1, )
        else:
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
        return np.mean(x_obs, axis=0)
    
    @staticmethod
    def cart2pol(x, y):  # cartesian to polar coordinates in 2D
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
    
    def compute_exact_lr_simplevcomp(self, x_obs, t0, mle):
        ll_gmm_t0 = np.sum(np.log(self._compute_multivariate_normal_pdf(x=x_obs, mu=t0)))
        ll_gmm_t1 = np.sum(np.log(self._compute_multivariate_normal_pdf(x=x_obs, mu=mle)))
        return ll_gmm_t0 - ll_gmm_t1
    
    def _compute_exact_lr_simplevcomp(self, t0, mle, obs_sample_size):
        return (-1)*(obs_sample_size/2)*(np.linalg.norm((mle - t0).reshape(-1, self.d), ord=2, axis=1)**2)

    def calculate_nuisance_parameters_over_grid(self, *args, **kwargs):
        raise NotImplementedError('No nuisance parameter for this class.')

    def _compute_marginal_pdf(self, x_obs):
        '''
        In this calculation we are assuming that the covariance matrix is diagonal with all entries being equal, so
        we only consider the first element for every point.
        '''
        density = 0.5 * (erf((self.high_int - x_obs) / (np.sqrt(2) * self.true_cov[0, 0])) -
                         erf((self.low_int - x_obs) / (np.sqrt(2) * self.true_cov[0, 0])))
        return np.prod(density)

    def _compute_marginal_bf_denominator(self, x_obs, prior_type='uniform'):
        '''
        In this calculation we are assuming that the covariance matrix is diagonal with all entries being equal, so
        we only consider the first element for every point.
        '''
        if prior_type == 'uniform':
            unif_distr = (1 / (self.high_int - self.low_int)) ** self.observed_dims
            density = 0.5 * unif_distr * (erf((self.high_int - x_obs) / (np.sqrt(2) * self.true_cov[0, 0])) -
                             erf((self.low_int - x_obs) / (np.sqrt(2) * self.true_cov[0, 0])))
        else:
            raise ValueError("The prior type needs to be 'uniform'. Currently %s" % self.prior_type)
        return np.prod(density)

    def compute_exact_bayes_factor_with_marginal(self, theta_vec, x_vec):
        if self.prior_type == 'uniform':
            x_vec = x_vec.reshape(-1, self.observed_dims)
            theta_vec = theta_vec.reshape(-1, self.d)
            
            f_val = np.array([self._compute_multivariate_normal_pdf(
                x=x, mu=theta_vec[ii, :]) for ii, x in enumerate(x_vec)]).reshape(-1, )
            g_val = np.array([self._compute_marginal_bf_denominator(x, prior_type='uniform') for x in x_vec]
                             ).reshape(-1, )
        else:
            raise ValueError("The prior type needs to be 'uniform'. Currently %s" % self.prior_type)
        return f_val / g_val
    
    def compute_exact_bayes_factor_single_t0(self, obs_sample, t0):
        results = np.array([self.compute_exact_bayes_factor_with_marginal(theta_vec=t0, x_vec=x)
                            for x in obs_sample])
        exact_bayes_t0 = np.sum(np.log(results)).astype(np.float64)
        assert isinstance(exact_bayes_t0, float)
        return exact_bayes_t0


class ClfOddsExact:

    def __init__(self, toy_mvn_model, d):
        self.toy_mvn_model = toy_mvn_model
        self.d = d

    def predict_proba(self, param_mat):
        prob_mat = np.apply_along_axis(arr=param_mat, axis=1,
                                       func1d=lambda row: self.toy_mvn_model.compute_exact_odds(
                                           theta_vec=row[:self.d], x_vec=row[self.d:]
                                        ))
        return np.hstack((np.ones(prob_mat.shape[0]).reshape(-1, 1), prob_mat.reshape(-1, 1)))
"""