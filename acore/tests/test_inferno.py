import sys
sys.path.append("..")
import pytest
import numpy as np
from scipy.stats import multivariate_normal, poisson, rv_discrete, expon

from models.inferno import InfernoToyLoader


def manual_likelihood(x, mu_vec, sigma_mats, mixing_param, lambda_param):

    mult_normal_bg = multivariate_normal.pdf(x=x[:2], mean=mu_vec[0], cov=sigma_mats[0])
    exp_bg = expon.pdf(x=x[2], scale=1/lambda_param)

    mult_normal_signal = multivariate_normal.pdf(x=x[:2], mean=mu_vec[1], cov=sigma_mats[1])
    exp_signal = expon.pdf(x=x[2], scale=1/2)

    return mixing_param * mult_normal_bg * exp_bg + (1 - mixing_param) * mult_normal_signal * exp_signal


def manual_or(obs_sample, t0_grid, grid_param):

    n = obs_sample.shape[0]
    n_t0 = t0_grid.shape[0]
    grid_t0_obs = np.hstack((
        np.repeat(t0_grid, n, axis=0),
        np.tile(obs_sample, (n_t0, 1))
    ))
    assert grid_t0_obs.shape[0] == n * n_t0

    lik_mat = np.apply_along_axis(arr=grid_t0_obs, axis=1,
                                  func1d=lambda row: manual_likelihood(
                                      x=row[4:], mu_vec=[np.array([2 + row[1], 0]), np.array([1, 1])],
                                      sigma_mats=[np.array([[5, 0], [0, 9]]), np.diag(np.ones(2))],
                                      mixing_param=row[3]/(row[0] + row[3]),
                                      lambda_param=row[2])).reshape(-1, )
    assert lik_mat.shape[0] == n * n_t0

    # Now group and find the maximum
    grouped_sum_t0 = np.array(
        [np.sum(np.log(lik_mat[n * ii:(n * (ii + 1))])) for ii in range(n_t0)])
    assert grouped_sum_t0.shape[0] == n_t0

    if np.array_equal(t0_grid, grid_param):
        max_val = np.max(grouped_sum_t0)
    else:
        # Compute the denominator value
        n_grid = grid_param.shape[0]
        grid_param_obs = np.hstack((
            np.repeat(grid_param, n, axis=0),
            np.tile(obs_sample, (n_grid, 1))
        ))
        assert grid_param_obs.shape[0] == n * n_grid

        lik_mat = np.apply_along_axis(arr=grid_param_obs, axis=1,
                                      func1d=lambda row: manual_likelihood(
                                      x=row[4:], mu_vec=[np.array([2 + row[1], 0]), np.array([1, 1])],
                                      sigma_mats=[np.array([[5, 0], [0, 9]]), np.diag(np.ones(2))],
                                      mixing_param=row[3]/(row[0] + row[3]),
                                      lambda_param=row[2])).reshape(-1, )
        assert lik_mat.shape[0] == n * n_grid

        # Now group and find the maximum
        grouped_sum_t1 = np.array(
            [np.sum(np.log(lik_mat[n * ii:(n * (ii + 1))])) for ii in range(n_grid)])
        assert grouped_sum_t1.shape[0] == n_grid
        max_val = np.max(grouped_sum_t1)

    return grouped_sum_t0 - max_val


def test__inferno_calculations():

    model = InfernoToyLoader(benchmark=1)
    x_obs = np.array([2, 1, 2])

    np.random.seed(7)
    for param_test in np.random.uniform(size=(20, 4)):
        inferno_calc = model._mixture_likelihood_manual(
            x=x_obs, mu_vec=model._compute_mean_vec(param_test[1]),
            sigma_mats=model.sigmas_mat,
            mixing_param=model._compute_mixing_param(s_param=param_test[0], b_param=param_test[3]),
            lambda_param=param_test[2])

        mu_vec = [np.array([2 + param_test[1], 0]), np.array([1, 1])]
        sigma_mats = [np.array([[5, 0], [0, 9]]), np.diag(np.ones(2))]
        mixing_param = param_test[3] / (param_test[0] + param_test[3])

        manual_calc = manual_likelihood(x=x_obs, mu_vec=mu_vec, sigma_mats=sigma_mats,
                                        mixing_param=mixing_param, lambda_param=param_test[2])

        assert np.abs(inferno_calc - manual_calc) <= 1e-6


def test__or_calculations():
    model = InfernoToyLoader(benchmark=1)

    np.random.seed(7)
    x_obs = np.random.uniform(size=(1, 3))
    x_obs = np.clip(x_obs, a_min=0, a_max=20)

    t0_grid = np.zeros((50, 4))
    t0_grid[:, 0] = np.arange(0, 100, 2)
    t0_grid[:, 1] = 0.0
    t0_grid[:, 2] = 5.0
    t0_grid[:, 3] = 1000.0

    or_val = model.compute_exactlr_single_t0(obs_sample=x_obs, t0_grid=t0_grid, grid_param=t0_grid)
    manual_val = manual_or(obs_sample=x_obs, t0_grid=t0_grid, grid_param=t0_grid)

    assert np.sum(np.abs(or_val - manual_val)) <= 1e-6

