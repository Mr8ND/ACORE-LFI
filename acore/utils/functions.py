import numpy as np
import sys
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


neighbor_range = [1, 2, 3, 4, 5, 10, 15, 20, 25] + [50 * x for x in range(1, 21)] + [500 * x for x in range(3, 6)]


def clf_prob_value(clf, x_vec, theta_vec, d, d_obs):
    predict_mat = np.hstack((theta_vec.reshape(-1, d),
                            x_vec.reshape(-1, d_obs)))
    prob_mat = clf.predict_proba(predict_mat)
    return prob_mat[:, 1]


def train_clf(sample_size, gen_function, clf_model,
              d=1, p=0.5, clf_name='xgboost', cv_nn=5, marginal=False, nn_square_root=False):
    '''
    This function works for multiple dimensions of the theta_parameters and the generated sample.

    :param sample_size: value of the number of samples to be generated to train the classifier model
    :param gen_function: a function to generate samples from the problem at hand
    :param clf_model: classifier model (sklearn compatible)
    :param d: the dimensionality of the parameter theta
    :param p: probability of Algorithm 1: generate a point from G or F_theta
    :param clf_name: Name of the classifier used
    :param cv_nn: Number of folds to be used in CV for nearest neighbors
    :param marginal: Whether or not we should attempt a parametric approximation of the marginal
    :param nn_square_root: If true, the number of neighbors for NN is chosen with the square root of the data
    :return: Trained classifier model
    '''

    gen_sample = gen_function(sample_size=sample_size, p=p, marginal=marginal)

    # this line below assumes sample has form (theta, label, X), where both theta and X can be multidimensional
    #col_selected = [el for el in range(gen_sample.shape[1]) if el != d]
    #X, y = gen_sample[:, col_selected], gen_sample[:, d]
    # TODO: make this independent of the position of columns. Or at least error-proof
    X, y = gen_sample[:, 1:], gen_sample[:, 0]  # my code puts the label first

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


def choose_clf_settings_subroutine(b_train,
                                   clf_model,
                                   clf_name,
                                   gen_function,
                                   d,
                                   eval_X,
                                   eval_y,
                                   target_loss):

    clf = train_clf(sample_size=b_train, clf_model=clf_model,
                    gen_function=gen_function, d=d, clf_name=clf_name)

    est_prob_vec = clf.predict_proba(eval_X)[:, 1]
    loss_value = target_loss(y_true=eval_y, y_pred=est_prob_vec)

    return [clf_name, b_train, loss_value]


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


def compute_statistics_single_t0(clf, obs_sample, obs_sample_size, t0, grid_param_t1, d=1, d_obs=1):
    # Stack the data
    # First column: we repeat t0 for n times, and then repeat t1 for n times each
    # Second column: We duplicate the data n_t1 + 1 times
    n = obs_sample.reshape(obs_sample_size, d_obs).shape[0]
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


def _compute_statistics_single_t0(name,
                                  clf,
                                  obs_sample,
                                  obs_sample_size,  # size of observed sample from same theta
                                  t0, grid_param_t1,
                                  d=1, d_obs=1,
                                  n_samples=1):  # construct conf set for each (> 1 if conf band, i.e. all together)

    if obs_sample_size > 1:
        # need to check predict_mat, obs_sample should be 3dim if n_samples/obs_sample_size > 1
        # bff implementation for now is a special case where g==marginal, obs_sample_size==1
        raise NotImplementedError
    if d > 1:
        # still have to check the logic for this case (in theory it should similar to d=1 ...)
        raise NotImplementedError

    if name == 'bff':
        # in this special case bff reduces to simple odds at t0
        predict_mat = np.hstack((
            np.repeat(t0, n_samples).reshape(-1, d),
            obs_sample.reshape(-1, d_obs)
        ))
        prob_mat = clf.predict_proba(predict_mat)
        prob_mat[prob_mat == 0] = 1e-15
        assert prob_mat.shape == (n_samples, 2)

        # odds
        odds_t0 = prob_mat[:, 1] / prob_mat[:, 0]
        assert odds_t0.shape[0] == n_samples

        return odds_t0

    elif name == 'acore':  # TODO: thoroughly check this one. Not sure it's completely ok.
        # Stack the data
        # First column: we repeat t0 for n times, and then repeat t1 for n times each
        # Second column: We duplicate the data n_t1 + 1 times
        n_t1 = grid_param_t1.shape[0]

        predict_mat = np.hstack((
            np.vstack((
                np.repeat(t0, obs_sample_size*n_samples).reshape(-1, d),
                np.tile(np.repeat(grid_param_t1, obs_sample_size).reshape(-1, d),  # this is huge ...
                        (n_samples, 1))
            )),
            np.vstack((
                obs_sample.reshape(-1, d_obs),
                # TODO: check this to ensure obs_sample_size > 1 works
                np.repeat(obs_sample, n_t1, axis=0).reshape(-1, d_obs)
            ))
        ))
        assert predict_mat.shape == (n_samples * obs_sample_size * n_t1 + n_samples * obs_sample_size, d + d_obs)

        # Do the prediction step
        prob_mat = clf.predict_proba(predict_mat)
        prob_mat[prob_mat == 0] = 1e-15
        assert prob_mat.shape == (n_samples * obs_sample_size * n_t1 + n_samples * obs_sample_size, 2)

        # Calculate odds
        # We extract t0 values
        odds_t0 = np.log(prob_mat[0:obs_sample_size*n_samples, 1]) - np.log(prob_mat[0:obs_sample_size*n_samples, 0])
        assert odds_t0.shape[0] == obs_sample_size*n_samples

        # We then extract t1_values
        odds_t1 = np.log(prob_mat[obs_sample_size*n_samples:, 1]) - np.log(prob_mat[obs_sample_size*n_samples:, 0])
        # TODO: check this shape when obs_sample_size > 1
        assert odds_t1.shape[0] == n_samples * n_t1

        # Calculate sum of logs for each ...
        # ... of the samples from the same theta, for which size==observed_sample_size
        grouped_sum_t0 = odds_t0.reshape(-1, obs_sample_size).sum(axis=1)
        assert grouped_sum_t0.shape[0] == n_samples

        # ... and of the value of the t1 grid
        # TODO: should add 3rd dim and add sum over same_theta_sample if obs_sample_size > 1
        grouped_max_sum_t1 = odds_t1.reshape(-1, n_t1).max(axis=1)
        assert grouped_max_sum_t1.shape[0] == n_samples

        return grouped_sum_t0 - grouped_max_sum_t1

    else:
        raise NotImplementedError

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


def or_loss(clf, first_sample, second_sample):
    num1, den1, num2, den2 = clf.predict_proba(first_sample[:, (0, 1)]), clf.predict_proba(first_sample[:, (0, 2)]), \
                             clf.predict_proba(second_sample[:, (0, 1)]), clf.predict_proba(second_sample[:, (0, 2)])

    # Some of the classifiers might return odds which are infinity -- we filter those out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        odds_num1, odds_den1 = (num1[:, 1] / num1[:, 0]).reshape(-1, ), (den1[:, 1] / den1[:, 0]).reshape(-1, )
        odds_num2, odds_den2 = (num2[:, 1] / num2[:, 0]).reshape(-1, ), (den2[:, 1] / den2[:, 0]).reshape(-1, )

    first_term = np.mean([el / odds_den1[ii] for ii, el in enumerate(odds_num1) if el != np.inf
                          and odds_den1[ii] != np.inf and odds_den1[ii] > 0])
    second_term = np.mean([el / odds_den2[ii] for ii, el in enumerate(odds_num2) if el != np.inf
                          and odds_den2[ii] != np.inf and odds_den2[ii] > 0])

    return first_term - 2 * second_term
