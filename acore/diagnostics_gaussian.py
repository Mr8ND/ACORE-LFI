from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import argparse
import pickle
from tqdm.auto import tqdm
from datetime import datetime
from functools import partial
from xgboost import XGBClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from utils.functions import train_clf, compute_statistics_single_t0, compute_bayesfactor_single_t0, \
    sample_from_matrix, matrix_mesh, compute_bayesfactor_single_t0_nuisance
from utils.qr_functions import train_qr_algo
from qr_algorithms.complete_list import classifier_cde_dict
from models.toy_mvn_multid_isotropic import ToyMVNMultiDIsotropicLoader

model_dict = {
    'toy_iso': ToyMVNMultiDIsotropicLoader
}


def main(d_obs, run, b, b_prime, alpha, t0_val, sample_size_obs, classifier, classifier_cde, test_statistic,
          diagnostic_sample=2000, monte_carlo_samples=500, debug=False, seed=7, size_check=1000,
         size_marginal=1000, empirical_marginal=True, num_grid=11,
         verbose=False, marginal=False, uniform_grid_sample_size=1000, interval_limit=5):

    # Changing values if debugging
    b = b if not debug else 100
    b_prime = b_prime if not debug else 100

    # Assign the correct classifier in the INFERNO run
    classifier_dict = {'QDA': QuadraticDiscriminantAnalysis()}

    # We pass as inputs all arguments necessary for all classes, but some of them will not be picked up if they are
    # not necessary for a specific class
    model_obj = model_dict[run](
        d_obs=d_obs, marginal=marginal, size_marginal=size_marginal, empirical_marginal=empirical_marginal,
        true_param=t0_val, num_acore_grid=num_grid, num_pred_grid=num_grid, diagnostic_flag=True,
        uniform_grid_sample_size=uniform_grid_sample_size, low_int=-interval_limit, high_int=interval_limit
    )

    # Get the correct functions
    gen_obs_func = model_obj.sample_sim
    gen_sample_func = model_obj.generate_sample
    gen_param_fun = model_obj.sample_param_values
    t0_grid = model_obj.pred_grid
    grid_param = model_obj.acore_grid
    np.random.seed(seed)

    # Generate a series of parameter values for High-Dim gaussian
    param_mat = gen_param_fun(sample_size=diagnostic_sample)

    # Generate observation from those values
    sample_mat = np.apply_along_axis(
        arr=param_mat, axis=1, func1d=lambda row: gen_obs_func(sample_size=sample_size_obs, true_param=row))

    # Train the odds classifier
    clf_model = classifier_dict[classifier]
    clf_name = classifier
    clf_odds = train_clf(d=model_obj.d, sample_size=b, clf_model=clf_model, gen_function=gen_sample_func,
                         clf_name=clf_name, nn_square_root=True)

    # Compute the nuisance parameters if necessary
    theta0_mat = param_mat

    # Create the B' sample to check the diagnostic over
    theta_mat_bprime, sample_mat_bprime = model_obj.sample_msnh_algo5(b_prime=b_prime, sample_size=sample_size_obs)
    b_prime_sample_list = [(theta_mat_bprime, sample_mat_bprime)] * diagnostic_sample

    # Create the confidence sets now
    tau_obs_vec = np.zeros(diagnostic_sample)
    pred_quantile_vec = np.zeros(diagnostic_sample)
    pbar = tqdm(total=diagnostic_sample, desc='Compute Confidence sets')
    for idx in range(diagnostic_sample):

        # Compute the test statistic
        if test_statistic == 'acore':
            tau_obs = np.array([
                compute_statistics_single_t0(
                    clf=clf_odds, obs_sample=sample_mat[idx, :, :], t0=theta_0, grid_param_t1=grid_param,
                    d=model_obj.d, d_obs=model_obj.d_obs) for theta_0 in theta0_mat[idx, :].reshape(1, -1)])
        elif test_statistic == 'logavgacore':
            tau_obs = np.array([
                    compute_bayesfactor_single_t0(
                        clf=clf_odds, obs_sample=sample_mat[idx, :, :], t0=theta_0, gen_param_fun=gen_param_fun,
                        d=model_obj.d, d_obs=model_obj.d_obs, log_out=True)
                    for theta_0 in theta0_mat[idx, :].reshape(1, -1)])
        else:
            raise ValueError('The variable test_statistic needs to be either acore or logavgacore. '
                             'Currently %s' % test_statistic)
        tau_obs_vec[idx] = tau_obs

        # Compute the test statistics for the B' sample
        theta_mat_bprime, sample_mat_bprime = b_prime_sample_list[idx]
        if test_statistic == 'acore':
            stats_mat_bprime = np.array([compute_statistics_single_t0(
                clf=clf_odds, d=model_obj.d, d_obs=model_obj.d_obs, grid_param_t1=grid_param,
                t0=theta_0, obs_sample=sample_mat_bprime[kk, :, :]) for kk, theta_0 in enumerate(theta_mat_bprime)
            ])
        elif test_statistic == 'logavgacore':
            stats_mat_bprime = np.array([compute_bayesfactor_single_t0(
                clf=clf_odds, d=model_obj.d, d_obs=model_obj.d_obs, gen_param_fun=gen_param_fun,
                monte_carlo_samples=monte_carlo_samples, log_out=True,
                t0=theta_0, obs_sample=sample_mat_bprime[kk, :, :]) for kk, theta_0 in enumerate(theta_mat_bprime)
            ])
        else:
            raise ValueError('The variable test_statistic needs to be either acore or logavgacore. '
                             'Currently %s' % test_statistic)

        clf_params = classifier_cde_dict[classifier_cde]
        t0_pred_vec = train_qr_algo(model_obj=model_obj, theta_mat=theta_mat_bprime, stats_mat=stats_mat_bprime,
                                    algo_name=clf_params[0], learner_kwargs=clf_params[1],
                                    pytorch_kwargs=clf_params[2] if len(clf_params) > 2 else None,
                                    alpha=alpha, prediction_grid=theta0_mat[idx, :].reshape(-1, 1))
        pred_quantile_vec[idx] = t0_pred_vec.reshape(-1, )
        pbar.update(1)

    # Compute whether they are in the confidence interval
    in_confint = (tau_obs_vec >= pred_quantile_vec).astype(float)
    t0_vec_coverage_pred = theta0_mat
    predict_mat = matrix_mesh(np.meshgrid(
        np.linspace(start=model_obj.low_int, stop=model_obj.high_int, num=100),
        np.linspace(start=model_obj.low_int, stop=model_obj.high_int, num=100)
    ))

    # Now we can use some sort of logistic regression/XGboost to compute coverage across the parameter space
    # Calculate the mean with an XGB Classifier
    model = XGBClassifier(**{'max_depth': 3, 'n_estimators': 100})
    model.fit(t0_vec_coverage_pred, in_confint.reshape(-1, ))
    pred_cov_mean = model.predict_proba(predict_mat)[:, 1]
    percent_correct_coverage = np.average((pred_cov_mean > (1.0 - alpha)).astype(int))
    average_coverage = np.average(pred_cov_mean)

    out_dict = {
        'theta_mat': theta0_mat,
        'tau_obs_vec': tau_obs_vec,
        'pred_quantile_vec': pred_quantile_vec,
        'in_confint_vec': in_confint,
        'pred_cov_mean_xgb': pred_cov_mean,
        'percent_correct_coverage_xgb': percent_correct_coverage,
        'average_coverage_xgb': average_coverage,
        'b': b,
        'b_prime': b_prime,
        'classifier': classifier,
        'classifier_qr': classifier_cde,
        'sample_size_obs': sample_size_obs,
        'alpha': alpha,
        'diagnostic_sample': diagnostic_sample,
        'run': run,
        'nuisance': model_obj.nuisance_flag,
        'test_statistic': test_statistic
    }
    out_dir = 'sims/classifier_power_multid/'
    out_filename = 'toygaussianiso_diagnostic_%steststats_%sB_%sBprime_%sseed_alpha%s_sampleobs%s_t0val%s_%s_%s.pkl' % (
        test_statistic, b, b_prime, seed,
        str(alpha).replace('.', '-'), sample_size_obs,
        str(t0_val).replace('.', '-'), classifier_cde,
        datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    pickle.dump(out_dict, open(out_dir + out_filename, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', action="store", type=int, default=7,
                        help='Random State')
    parser.add_argument('--d_obs', action="store", type=int, default=2,
                        help='Dimensionality of the observed data (feature space)')
    parser.add_argument('--rep', action="store", type=int, default=10,
                        help='Number of Repetitions for calculating the Pinball loss')
    parser.add_argument('--b', action="store", type=int, default=5000,
                        help='Sample size to train the classifier for calculating odds')
    parser.add_argument('--b_prime', action="store", type=int, default=1000,
                        help='Sample size to train the quantile regression algorithm')
    parser.add_argument('--marginal', action='store_true', default=False,
                        help='Whether we are using a parametric approximation of the marginal or'
                             'the baseline reference G')
    parser.add_argument('--empirical_marginal', action='store_true', default=False,
                        help='Whether we are sampling directly from the empirical marginal for G')
    parser.add_argument('--alpha', action="store", type=float, default=0.1,
                        help='Statistical confidence level')
    parser.add_argument('--t0_val', action="store", type=float, default=5.0,
                        help='True param value')
    parser.add_argument('--run', action="store", type=str, default='toy_iso',
                        help='Problem to run')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If true, a very small value for the sample sizes is fit to make sure the'
                             'file can run quickly for debugging purposes')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='If true, logs are printed to the terminal')
    parser.add_argument('--sample_size_obs', action="store", type=int, default=10,
                        help='Sample size of the actual observed data.')
    parser.add_argument('--test_statistic', action="store", type=str, default='acore',
                        help='Type of ACORE test statistic to use.')
    parser.add_argument('--class_cde', action="store", type=str, default='xgb_d3_n100',
                        help='Classifier for quantile regression')
    parser.add_argument('--classifier', action="store", type=str, default='QDA',
                        help='Classifier for learning odds')
    parser.add_argument('--size_marginal', action="store", type=int, default=1000,
                        help='Sample size of the actual marginal distribution, if marginal is True.')
    parser.add_argument('--diagnostic_sample', action="store", type=int, default=2000,
                        help='Sample for diagnostics meaning.')
    parser.add_argument('--num_grid', action="store", type=int, default=11,
                        help='Number of points in the parameter grid.')
    parser.add_argument('--monte_carlo_samples', action="store", type=int, default=1000,
                        help='Sample size for the calculation of integral in the Logavgacore.')
    parser.add_argument('--unif_grid', action="store", type=int, default=1000,
                        help='Number of grid points to evaluate for ACORE maximimzation.')
    parser.add_argument('--int_limit', action="store", type=int, default=5.0,
                        help='Limit over which the integration/maximization is done for the mean value.')
    argument_parsed = parser.parse_args()

    main(
        d_obs=argument_parsed.d_obs,
        run=argument_parsed.run,
        marginal=argument_parsed.marginal,
        empirical_marginal=argument_parsed.empirical_marginal,
        b=argument_parsed.b,  # b_val,
        b_prime=argument_parsed.b_prime,
        alpha=argument_parsed.alpha,
        t0_val=argument_parsed.t0_val,
        debug=argument_parsed.debug,
        sample_size_obs=argument_parsed.sample_size_obs,
        seed=argument_parsed.seed,
        verbose=argument_parsed.verbose,
        test_statistic=argument_parsed.test_statistic,
        classifier_cde=argument_parsed.class_cde,
        size_marginal=argument_parsed.size_marginal,
        monte_carlo_samples=argument_parsed.monte_carlo_samples,
        classifier=argument_parsed.classifier,
        diagnostic_sample=argument_parsed.diagnostic_sample,
        num_grid=argument_parsed.num_grid
    )
