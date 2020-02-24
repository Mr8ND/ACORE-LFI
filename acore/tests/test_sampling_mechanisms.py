import sys
sys.path.append("..")
import pytest
import numpy as np

from models.sen_poisson import SenPoissonLoader
from models.toy_gmm import ToyGMMLoader
from scipy.stats import multivariate_normal, norm


def test__camelus_linc_sampling():

    np.random.seed(7)
    mean_instrumental = np.repeat(0, 7)
    cov_instrumental = np.diag(np.repeat(1, 7))

    instrumental_distr = multivariate_normal(mean=mean_instrumental, cov=cov_instrumental)

    # Create sample concat_mat
    concat_mat = np.array([[1.5, 1.5, 0], [0.5, 0.5, 1]])
    obs_value = np.array([1, 2, 3, 4, 5, 6, 7])
    random_sample = np.abs(instrumental_distr.rvs(size=1)).astype(int)

    # Sample matrix
    sample_mat_1 = np.apply_along_axis(arr=concat_mat, axis=1,
                                       func1d=lambda row: obs_value if row[2] else random_sample)
    sample_mat_2 = np.apply_along_axis(arr=concat_mat, axis=1,
                                       func1d=lambda row: obs_value.reshape(1, 7) if row[2]
                                            else random_sample.reshape(1, 7)).reshape(-1, 7)
    expected_mat = np.vstack((random_sample.reshape(-1, 7), obs_value.reshape(-1, 7)))

    np.testing.assert_array_equal(sample_mat_1, expected_mat)
    np.testing.assert_array_equal(sample_mat_2, expected_mat)


def test__poisson_2d_sampling_msnh():

    def sample_msnh_algo5_poisson_two_params(b_prime, sample_size):
        background_vec = np.random.uniform(low=80, high=100,
                                           size=b_prime).reshape(-1, 1)
        mu_vec = np.random.uniform(low=0, high=20,
                                   size=b_prime).reshape(-1, 1)
        theta_mat = np.hstack((background_vec, mu_vec))
        assert theta_mat.shape == (b_prime, 2)

        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: np.hstack(
                                             (np.random.poisson(lam=row[0] + row[1], size=sample_size).reshape(-1, 1),
                                              np.random.poisson(lam=1 * row[0], size=sample_size).reshape(-1, 1))))
        return theta_mat, sample_mat

    np.random.seed(7)
    t1, s1 = sample_msnh_algo5_poisson_two_params(100, 10)

    model_obj = SenPoissonLoader()
    model_obj.set_reference_g(size_reference=100)
    np.random.seed(7)
    t2, s2 = model_obj.sample_msnh_algo5(100, 10)

    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(s1, s2)


def test__poisson_2d_sampling():

    def generate_sample_poisson_two_params(sample_size, mean_instrumental_poisson, cov_instrumental_poisson,
                                           p=0.5, marginal=False):
        background_vec = np.random.uniform(low=80, high=100,
                                           size=sample_size).reshape(-1, 1)
        mu_vec = np.random.uniform(low=0, high=20,
                                   size=sample_size).reshape(-1, 1)
        theta_mat = np.hstack((background_vec, mu_vec))
        assert theta_mat.shape == (sample_size, 2)

        bern_vec = np.random.binomial(n=1, p=p, size=sample_size)
        concat_mat = np.hstack((theta_mat.reshape(-1, 2),
                                bern_vec.reshape(-1, 1)))

        if marginal:
            raise ValueError('Marginal not implemented for this example')
        else:
            instrumental_distr = multivariate_normal(mean=mean_instrumental_poisson, cov=cov_instrumental_poisson)

        sample = np.apply_along_axis(arr=concat_mat, axis=1,
                                     func1d=lambda row: np.array([
                                         np.random.poisson(lam=row[0] + row[1], size=1),
                                         np.random.poisson(lam=1 * row[0], size=1)]).reshape(1, 2) if row[
                                         2] else
                                     np.abs(instrumental_distr.rvs(size=1)).astype(int).reshape(1, 2))
        return np.hstack((concat_mat, sample.reshape(-1, 2)))

    model_obj = SenPoissonLoader()
    model_obj.set_reference_g(size_reference=100)
    np.random.seed(7)
    t1 = model_obj.generate_sample(100)

    np.random.seed(7)
    t2 = generate_sample_poisson_two_params(100, model_obj.mean_instrumental, model_obj.cov_instrumental)

    np.testing.assert_array_equal(t1, t2)


def test__msnh_algo5_gmm():

    def gmm_manual_sampling(sample_size, mix_param=0.5, mu_param=[-5, 5], sigma_param=[1, 1]):
        cluster = np.random.binomial(n=1, p=mix_param, size=sample_size)
        means = np.take(mu_param, cluster)
        sigmas = np.take(sigma_param, cluster)
        return np.random.normal(loc=means, scale=sigmas, size=sample_size)

    def sample_msnh_algo5_gmm(b_prime, sample_size):
        theta_mat = np.random.uniform(low=0, high=10, size=b_prime).reshape(-1,
                                                                                                                    1)
        sample_mat = np.apply_along_axis(arr=theta_mat, axis=1,
                                         func1d=lambda row: gmm_manual_sampling(
                                             sample_size=sample_size, mu_param=[-row, row]))
        full_mat = np.hstack((theta_mat, sample_mat))
        return theta_mat, full_mat

    def generate_sample_gmm(sample_size=1000, p=0.5, marginal=False, sample_marginal=1000):

        theta_vec = np.random.uniform(low=0, high=10, size=sample_size)  # Reference Distribution
        bern_vec = np.random.binomial(n=1, p=p, size=sample_size)  # Generating Y_1,...,Y_n

        # Chaining in columns the two above and then sample either from F or G
        # according to Y_i for i=1,...,n
        concat_mat = np.hstack((theta_vec.reshape(-1, 1),
                                bern_vec.reshape(-1, 1)))

        if marginal:
            theta_vec_marg = np.random.uniform(low=0, high=10,
                                               size=sample_marginal)
            marginal_sample = np.apply_along_axis(arr=theta_vec_marg.reshape(-1, 1), axis=1,
                                                  func1d=lambda row: gmm_manual_sampling(
                                                      sample_size=1, mu_param=[-row, row])).reshape(-1, )
            mean_mle = np.average(marginal_sample)
            std_mle = np.std(marginal_sample)
            instrumental_distr = norm(loc=mean_mle, scale=std_mle)
        else:
            instrumental_distr = norm(loc=0, scale=10)

        sample = np.apply_along_axis(arr=concat_mat, axis=1,
                                     func1d=lambda row: gmm_manual_sampling(
                                         sample_size=1, mu_param=[-row[0], row[0]]) if row[1] else
                                     instrumental_distr.rvs(size=1))
        return np.hstack((concat_mat, sample.reshape(-1, 1)))

    model_obj = ToyGMMLoader()
    np.random.seed(7)
    t1, s1 = model_obj.sample_msnh_algo5(10, 10)

    np.random.seed(7)
    t2, f2 = sample_msnh_algo5_gmm(10, 10)

    np.testing.assert_array_equal(t1, t2)
    np.testing.assert_array_equal(np.hstack((t1, s1)), f2)

    np.random.seed(7)
    b1 = model_obj.generate_sample(100)

    np.random.seed(7)
    b2 = generate_sample_gmm(100)

    np.testing.assert_array_equal(b1[:, :2], b2[:, :2])
    np.testing.assert_array_equal(b1[:, 2:], b2[:, 2:])