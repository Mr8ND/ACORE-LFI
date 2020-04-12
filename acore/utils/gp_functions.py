import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel


def train_gp(sample_size, n_anchor_points, model_obj, t0_grid, sample_type='MC'):

    if sample_type == 'MC':
        min_val, max_val = np.min(t0_grid), np.max(t0_grid)
        anchor_points = np.linspace(min_val, max_val, n_anchor_points)
        anchor_sample_size = int(sample_size/n_anchor_points)

        gp_samples = np.array([
            model_obj.sample_sim(
                sample_size=anchor_sample_size, true_param=anchor) for anchor in anchor_points]).reshape(
            -1, model_obj.d_obs)
        if model_obj.d == 1:
            theta_mat = np.repeat(anchor_points, repeats=anchor_sample_size).reshape(-1, 1)
        else:
            theta_mat = np.tile(anchor_points, repeats=anchor_sample_size).reshape(-1, model_obj.d)
    elif sample_type == 'uniform':
        theta_mat = model_obj.sample_param_values(sample_size=sample_size).reshape(-1, 1)
        gp_samples = np.array([
            model_obj.sample_sim(
                sample_size=1, true_param=theta_val) for theta_val in theta_mat]).reshape(
            -1, model_obj.d_obs)
    else:
        raise ValueError('Sample type not defined. It can either be "MC" or "uniform".')

    gp_model = GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel())
    gp_model.fit(theta_mat, gp_samples)

    return gp_model


def compute_statistics_single_t0_gp(gp_model, obs_sample, t0, grid_param_t1, d, d_obs):

    if d_obs > 1:
        raise NotImplementedError

    params_vec = np.vstack((t0.reshape(1, d), grid_param_t1.reshape(-1, d)))
    mean_vec, std_vec = gp_model.predict(params_vec.reshape(-1, d), return_std=True)
    std_vec[std_vec == 0] = 1e-15

    # Likelihood of the observed sample at t0
    density_obj = norm(loc=mean_vec[0], scale=std_vec[0])
    first_term = np.sum(density_obj.logpdf(obs_sample))

    # Min of likelihood everywhere else
    second_term = np.max([np.sum(norm(
        loc=mu_param, scale=std_vec[jj]).logpdf(obs_sample)) for jj, mu_param in enumerate(mean_vec[1:])])

    return first_term - second_term
