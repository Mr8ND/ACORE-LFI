import numpy as np
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from functools import partial
from scipy.optimize import minimize


neighbor_range = [1, 2, 3, 4, 5, 10, 15, 20, 25] + [50 * x for x in range(1, 21)] + [500 * x for x in range(3, 6)]


def odds_ratio_loss(clf, x_vec, theta_vec, bern_vec, d=1, d_obs=1, p=0.5):

    # Obtain samples from the simulator and from the reference distribution
    # gen_sample = gen_function(sample_size=sample_size, p=p, marginal=marginal)

    # Split them accordingly
    theta_simulator = theta_vec[bern_vec == 1].reshape(-1, d)
    x_simulator = x_vec[bern_vec == 1].reshape(-1, d_obs)
    theta_reference = theta_vec[bern_vec == 0].reshape(-1, d)
    x_reference = x_vec[bern_vec == 0].reshape(-1, d_obs)

    # Predict the odds for both of them
    prob_simulator = clf.predict_proba(np.hstack((theta_simulator, x_simulator)))
    prob_simulator = prob_simulator[~np.any(prob_simulator == 0, axis=1)]
    odds_simulator = (prob_simulator[:, 1]) / (prob_simulator[:, 0])

    prob_reference = clf.predict_proba(np.hstack((theta_reference, x_reference)))
    prob_reference = prob_reference[~np.any(prob_reference == 0, axis=1)]
    odds_reference = (prob_reference[:, 1]) / (prob_reference[:, 0])

    return np.average(odds_reference**2) - 2 * (p/(1-p)) * np.average(odds_simulator)


def clf_prob_value(clf, x_vec, theta_vec, d, d_obs):
    predict_mat = np.hstack((theta_vec.reshape(-1, d),
                            x_vec.reshape(-1, d_obs)))
    prob_mat = clf.predict_proba(predict_mat)
    return prob_mat[:, 1]


def train_pvalue_clf(clf_model, X, y, clf_name='xgboost', cv_nn=5, nn_square_root=False):
    '''
    This function trains a classifier for a classification task by passing in a X and y vectors.
    It checks whether the classifier is a nearest neighbor algorithm, and if so it does set the neighbors either
    to be the square root of the total number of samples (X.shape[0]) or finds them by cross-validation

    :param clf_model: classifier model (sklearn compatible)
    :param X: training data features (numpy array)
    :param y: training data response (numpy array)
    :param clf_name: Name of the classifier used
    :param cv_nn: Number of folds to be used in CV for nearest neighbors
    :param nn_square_root: If true, the number of neighbors for NN is chosen with the square root of the data
    :return: Trained classifier model
    '''

    if 'nn' in clf_name.lower():

        if nn_square_root:
            clf_model = KNeighborsClassifier(n_neighbors=int(np.sqrt(X.shape[0])))
            clf_model.fit(X=X, y=y)

        else:
            grid_params = {'n_neighbors': np.array(neighbor_range)}
            # The following lines makes sure that we are not selecting a number of neighbors which is too
            # large with respect to the data we are going to use for CV
            grid_params['n_neighbors'] = [x for x in grid_params['n_neighbors'] if x < (X.shape[0]*(1 - (1/cv_nn)))]
            gs = GridSearchCV(
                KNeighborsClassifier(),
                grid_params,
                verbose=0,
                cv=cv_nn,
                n_jobs=-1,
                scoring='neg_log_loss',
                iid=True
            )
            gs_results = gs.fit(X, y)
            clf_model = gs_results.best_estimator_
    else:
        clf_model.fit(X=X, y=y)

    return clf_model


def train_clf(sample_size, gen_function, clf_model,
              d=1, p=0.5, clf_name='xgboost', cv_nn=5, nn_square_root=False):
    '''
    This function works for multiple dimensions of the theta_parameters and the generated sample.

    :param sample_size: value of the number of samples to be generated to train the classifier model
    :param gen_function: a function to generate samples from the problem at hand
    :param clf_model: classifier model (sklearn compatible)
    :param d: the dimensionality of the parameter theta
    :param p: probability of Algorithm 1: generate a point from G or F_theta
    :param clf_name: Name of the classifier used
    :param cv_nn: Number of folds to be used in CV for nearest neighbors
    :param nn_square_root: If true, the number of neighbors for NN is chosen with the square root of the data
    :return: Trained classifier model
    '''

    gen_sample = gen_function(sample_size=sample_size, p=p)

    col_selected = [el for el in range(gen_sample.shape[1]) if el != d]
    X, y = gen_sample[:, col_selected], gen_sample[:, d]

    if 'nn' in clf_name.lower():

        if nn_square_root:
            clf_model = KNeighborsClassifier(n_neighbors=int(np.sqrt(X.shape[0])))
            clf_model.fit(X=X, y=y)

        else:
            grid_params = {'n_neighbors': np.array(neighbor_range)}
            # The following lines makes sure that we are not selecting a number of neighbors which is too
            # large with respect to the data we are going to use for CV
            grid_params['n_neighbors'] = [x for x in grid_params['n_neighbors'] if x < (X.shape[0]*(1 - (1/cv_nn)))]
            gs = GridSearchCV(
                KNeighborsClassifier(),
                grid_params,
                verbose=0,
                cv=cv_nn,
                n_jobs=-1,
                scoring='neg_log_loss',
                iid=True
            )
            gs_results = gs.fit(X, y)
            clf_model = gs_results.best_estimator_
    else:
        clf_model.fit(X=X, y=y)

    return clf_model


def compute_odds(clf, obs_data, theta_val, clf_name='xgboost'):
    '''
    Computing odds of the observed data `obs_data`, given a classifier `clf`
    and value of the parameter `theta_val`.
    '''
    predict_mat = np.hstack((
        np.repeat(theta_val, obs_data.shape[0]).reshape(-1, 1),
        obs_data.reshape(-1, 1)
    ))

    if clf_name == 'logistic_regression':
        predict_mat = np.c_[predict_mat,
                            predict_mat[:, 0] * predict_mat[:, 1],
                            predict_mat[:, 0] ** 2, predict_mat[:, 1] ** 2,
                            (predict_mat[:, 0] * predict_mat[:, 1]) ** 2]

    prob_mat = clf.predict_proba(predict_mat)
    prob_mat[prob_mat == 0] = 1e-15
    return (prob_mat[:, 1]) / (prob_mat[:, 0])


def compute_log_odds_ratio(clf, obs_data, t0, t1, clf_name='xgboost'):
    '''
    Computing log-odds ratio of the observed data `obs_data`, given a classifier `clf`
    and the two values of the parameters.
    '''
    n = obs_data.shape[0]
    predict_mat = np.hstack((
        np.vstack((
            np.repeat(t0, n).reshape(-1, 1),
            np.repeat(t1, n).reshape(-1, 1)
        )),
        np.vstack((
            obs_data.reshape(-1, 1),
            obs_data.reshape(-1, 1)
        ))
    ))

    if clf_name == 'logistic_regression':
        predict_mat = np.c_[predict_mat,
                            predict_mat[:, 0] * predict_mat[:, 1],
                            predict_mat[:, 0] ** 2, predict_mat[:, 1] ** 2,
                            (predict_mat[:, 0] * predict_mat[:, 1]) ** 2]

    prob_mat = clf.predict_proba(predict_mat)
    prob_mat[prob_mat == 0] = 1e-15
    odds_vec = (prob_mat[:, 1] / prob_mat[:, 0]).reshape(-1, )
    odds_ratio = odds_vec[:n] / odds_vec[n:]

    return np.sum(np.log(odds_ratio))


def pinball_loss(y_true, y_pred, alpha):
    diff_vec = y_true - y_pred
    diff_mat = np.hstack((
        alpha * diff_vec.reshape(-1, 1),
        (alpha - 1) * diff_vec.reshape(-1, 1)
    ))
    return np.average(np.max(diff_mat, axis=1))


def compute_averageodds_single_t0(clf, obs_sample, t0, d=1, d_obs=1, apply_log=False):

    n = obs_sample.shape[0]

    predict_mat = np.hstack((
            np.tile(t0, n).reshape(-1, d),
            obs_sample.reshape(-1, d_obs),
    ))
    assert predict_mat.shape == (n, d + d_obs)

    # Do the prediction step
    prob_mat = clf.predict_proba(predict_mat)
    prob_mat[prob_mat == 0] = 1e-15
    assert prob_mat.shape == (n, 2)

    # Calculate odds
    # We extract t0 values
    odds_t0 = np.exp(np.sum(np.log(prob_mat[:, 1] / prob_mat[:, 0]))).astype(np.float64)
    assert isinstance(odds_t0, float)

    # return the average odds directly
    return odds_t0 if not apply_log else np.log(odds_t0)


def compute_bayesfactor_single_t0(clf, obs_sample, t0, gen_param_fun,
                                  log_out=False, d=1, d_obs=1, monte_carlo_samples=1000):

    theta_samples = gen_param_fun(sample_size=monte_carlo_samples)
    n = obs_sample.shape[0]

    if d > 1:
        predict_mat = np.hstack((
            np.vstack((
                np.tile(t0, n).reshape(-1, d),
                np.tile(theta_samples, n).reshape(-1, d)
            )),
            np.vstack((
                obs_sample.reshape(-1, d_obs),
                np.tile(obs_sample, (monte_carlo_samples, 1)).reshape(-1, d_obs)
            ))
        ))
    else:
        predict_mat = np.hstack((
            np.vstack((
                np.repeat(t0, n).reshape(-1, 1),
                np.repeat(theta_samples, n).reshape(-1, 1)
            )),
            np.vstack((
                obs_sample.reshape(-1, d_obs),
                np.tile(obs_sample, (monte_carlo_samples, 1)).reshape(-1, d_obs)
            ))
        ))
    assert predict_mat.shape == (n * monte_carlo_samples + n, d + d_obs)

    # Do the prediction step
    prob_mat = clf.predict_proba(predict_mat)
    prob_mat[prob_mat == 0] = 1e-15
    assert prob_mat.shape == (n * monte_carlo_samples + n, 2)

    # Calculate odds
    # We extract t0 values
    odds_t0 = np.exp(np.sum(np.log(prob_mat[0:n, 1] / prob_mat[0:n, 0]))).astype(np.float64)
    assert isinstance(odds_t0, float)

    # We calculate the values for the monte carlo samples to approximate the integral
    odds_t1 = prob_mat[n:, 1] / prob_mat[n:, 0]
    assert odds_t1.shape[0] == n * monte_carlo_samples

    # Calculate sum of logs for each of the value we sampled from Monte Carlo samples
    grouped_sum_t1 = np.array(
        [np.exp(np.sum(np.log(odds_t1[n * ii:(n * (ii + 1))]))) for ii in range(monte_carlo_samples)])
    assert grouped_sum_t1.shape[0] == monte_carlo_samples

    # Using the log of the average rather than the average itself
    if log_out:
        return np.log(odds_t0) - np.log(np.average(grouped_sum_t1.reshape(-1, )))

    # Using the bayes factor directly
    return odds_t0/np.average(grouped_sum_t1.reshape(-1, ))


def compute_statistics_single_t0(clf, obs_sample, t0, grid_param_t1, d=1, d_obs=1):
    # Stack the data
    # First sets of column: we repeat t0 for n times, and then repeat t1 for n times each
    # Second sets of column: We duplicate the data n_t1 + 1 times
    n = obs_sample.shape[0]
    n_t1 = grid_param_t1.shape[0]

    if d > 1:
        predict_mat = np.hstack((
            np.vstack((
                np.tile(t0, n).reshape(-1, d),
                np.tile(grid_param_t1, n).reshape(-1, d)
            )),
            np.vstack((
                obs_sample.reshape(-1, d_obs),
                np.tile(obs_sample, (n_t1, 1)).reshape(-1, d_obs)
            ))
        ))
    else:
        predict_mat = np.hstack((
            np.vstack((
                np.repeat(t0, n).reshape(-1, 1),
                np.repeat(grid_param_t1, n).reshape(-1, 1)
            )),
            np.vstack((
                obs_sample.reshape(-1, d_obs),
                np.tile(obs_sample, (n_t1, 1)).reshape(-1, d_obs)
            ))
        ))
    assert predict_mat.shape == (n * n_t1 + n, d + d_obs)

    # Do the prediction step
    prob_mat = clf.predict_proba(predict_mat)
    prob_mat[prob_mat == 0] = 1e-15
    assert prob_mat.shape == (n * n_t1 + n, 2)

    # Calculate odds
    # We extract t0 values
    odds_t0 = np.log(prob_mat[0:n, 1]) - np.log(prob_mat[0:n, 0])
    assert odds_t0.shape[0] == n

    # We then extract t1_values
    odds_t1 = np.log(prob_mat[n:, 1]) - np.log(prob_mat[n:, 0])
    assert odds_t1.shape[0] == n * n_t1

    # Calculate sum of logs for each of the value
    # of the t1 grid
    grouped_sum_t1 = np.array(
        [np.sum(odds_t1[n * ii:(n * (ii + 1))]) for ii in range(n_t1)])
    assert grouped_sum_t1.shape[0] == n_t1

    return np.sum(odds_t0) - np.max(grouped_sum_t1)


def _clf_odds_function(theta_param, clf, obs_sample, d=1, d_obs=1, mult_min=-1):
    predict_mat_t0 = np.hstack((np.tile(theta_param, obs_sample.shape[0]).reshape(-1, d),
                                obs_sample.reshape(-1, d_obs)))
    prob_mat_t0 = clf.predict_proba(predict_mat_t0)
    prob_mat_t0[prob_mat_t0 == 0] = 1e-15
    return mult_min * np.sum(np.log(prob_mat_t0[:, 1]) - np.log(prob_mat_t0[:, 0]))


def compute_statistics_single_t0_multid(clf, obs_sample, t0, bounds_opt, d=1, d_obs=1):
    # First compute the first term, which is the value of the odds under theta0
    t0_odds = _clf_odds_function(theta_param=t0, clf=clf, obs_sample=obs_sample, d=d, d_obs=d_obs)

    # Then implement some sort of gradient-free optimization routing to find the minimum over multiple dimensions
    opt_function = partial(_clf_odds_function, clf=clf, obs_sample=obs_sample, d=d, d_obs=d_obs, mult_min=-1)

    # Choose a starting value which avoids the exact value of the bounds
    starting_value = t0 + 0.01 if np.sum(t0) == 0.0 else t0 - 0.01
    res = minimize(opt_function, starting_value, method='trust-constr', options={'verbose': 0}, bounds=bounds_opt)
    t1_odds = _clf_odds_function(theta_param=res.x, clf=clf, obs_sample=obs_sample, d=d, d_obs=d_obs)

    return t0_odds - t1_odds


def compute_clf_tau_distr(clf, gen_obs_func, theta_0, t1_linspace, n_sampled=1000, sample_size_obs=200):
    full_obs_sample = gen_obs_func(sample_size=n_sampled * sample_size_obs, true_param=theta_0)
    sample_mat = full_obs_sample.reshape(n_sampled, sample_size_obs)

    tau_sample = np.apply_along_axis(arr=sample_mat, axis=1,
                                     func1d=lambda row: compute_statistics_single_t0(
                                         clf=clf, obs_sample=row, t0=theta_0, grid_param_t1=t1_linspace,
                                         d=len(theta_0)))

    return tau_sample


def compute_exact_tau(or_func, x_obs, t0_val, t1_linspace):
    return np.min(np.array([np.sum(np.log(or_func(x_obs=x_obs, t0=t0_val, t1=t1))) for t1 in t1_linspace]))


def compute_exact_tau_distr(gen_obs_func, or_func, t0_val, t1_linspace, d_obs=1, n_sampled=1000, sample_size_obs=200):
    full_obs_sample = gen_obs_func(sample_size=n_sampled * sample_size_obs, true_param=t0_val)

    if d_obs == 1:
        sample_mat = full_obs_sample.reshape(n_sampled, sample_size_obs)
        tau_sample = np.apply_along_axis(arr=sample_mat, axis=1,
                                         func1d=lambda row: compute_exact_tau(
                                             or_func=or_func, x_obs=row, t0_val=t0_val, t1_linspace=t1_linspace))
    else:
        sample_mat = full_obs_sample.reshape(n_sampled, sample_size_obs, d_obs)
        tau_sample = np.array([compute_exact_tau(
            or_func=or_func, x_obs=sample_mat[kk, :, :], t0_val=t0_val, t1_linspace=t1_linspace)
            for kk in range(n_sampled)])

    return tau_sample


class Suppressor(object):

    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self

    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            pass

    def write(self, x):
        pass


def matrix_mesh(a_tuple):
    return np.hstack((a_tuple[0].reshape(-1, 1), a_tuple[1].reshape(-1, 1)))


def tensor_4d_mesh(a_tuple):
    return np.hstack((
        a_tuple[0].reshape(-1, 1), a_tuple[1].reshape(-1, 1), a_tuple[2].reshape(-1, 1), a_tuple[3].reshape(-1, 1)
    ))


# def or_loss(clf, first_sample, second_sample):
#     num1, den1, num2, den2 = clf.predict_proba(first_sample[:, (0, 1)]), clf.predict_proba(first_sample[:, (0, 2)]), \
#                              clf.predict_proba(second_sample[:, (0, 1)]), clf.predict_proba(second_sample[:, (0, 2)])
#
#     # Some of the classifiers might return odds which are infinity -- we filter those out
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         odds_num1, odds_den1 = (num1[:, 1] / num1[:, 0]).reshape(-1, ), (den1[:, 1] / den1[:, 0]).reshape(-1, )
#         odds_num2, odds_den2 = (num2[:, 1] / num2[:, 0]).reshape(-1, ), (den2[:, 1] / den2[:, 0]).reshape(-1, )
#
#     first_term = np.mean([el / odds_den1[ii] for ii, el in enumerate(odds_num1) if el != np.inf
#                           and odds_den1[ii] != np.inf and odds_den1[ii] > 0])
#     second_term = np.mean([el / odds_den2[ii] for ii, el in enumerate(odds_num2) if el != np.inf
#                           and odds_den2[ii] != np.inf and odds_den2[ii] > 0])
#
#     return first_term - 2 * second_term


# from https://stackoverflow.com/questions/43554819/find-most-frequent-row-or-mode-of-a-matrix-of-vectors-python-numpy
def mode_rows(a):
    a = np.ascontiguousarray(a)
    void_dt = np.dtype((np.void, a.dtype.itemsize * np.prod(a.shape[1:])))
    _, ids, count = np.unique(a.view(void_dt).ravel(), return_index=1, return_counts=1)
    largest_count_id = ids[count.argmax()]
    most_frequent_row = a[largest_count_id]
    return most_frequent_row


def sample_from_matrix(t0_grid, sample_size):
    # Random sampling a sample_size number of rows from a t0_grid matrix with replacement
    return t0_grid[np.random.choice(a=t0_grid.shape[0], size=sample_size, replace=True)]
