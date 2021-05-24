from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import argparse
import pickle
from tqdm.auto import tqdm
from datetime import datetime
from functools import partial
from xgboost import XGBClassifier
from scipy.stats import chi2

from utils.functions import train_clf, compute_statistics_single_t0, compute_bayesfactor_single_t0, \
    sample_from_matrix, matrix_mesh, compute_bayesfactor_single_t0_nuisance
from models.hep_counting import HepCountingNuisanceLoader
from utils.qr_functions import train_qr_algo
from or_classifiers.toy_example_list import classifier_dict_multid_hep
from qr_algorithms.complete_list import classifier_cde_dict

model_dict = {
    'hep_counting': HepCountingNuisanceLoader
}


def main(d_obs, run, b, b_prime, alpha, t0_val, sample_size_obs, classifier, classifier_cde, test_statistic,
         alternative_norm, diagnostic_sample=2000, monte_carlo_samples=500, debug=False, seed=7,
         size_marginal=1000, empirical_marginal=True, benchmark=1, nuisance_parameters=False, num_grid=11,
         marginal=False, num_acore_grid=101):

    # Changing values if debugging
    b = b if not debug else 100
    b_prime = b_prime if not debug else 100

    if 'hep_counting' in run:
        classifier_dict = classifier_dict_multid_hep
    else:
        raise NotImplementedError('OR Classification is only specific for hep_counting in this file.')

    # We pass as inputs all arguments necessary for all classes, but some of them will not be picked up if they are
    # not necessary for a specific class
    model_obj = model_dict[run](
        d_obs=d_obs, marginal=marginal, size_marginal=size_marginal, empirical_marginal=empirical_marginal,
        true_param=t0_val, alt_mu_norm=alternative_norm, nuisance_parameters=nuisance_parameters,
        benchmark=benchmark, num_acore_grid=num_acore_grid, num_pred_grid=num_grid
    )

    # Get the correct functions
    gen_obs_func = model_obj.sample_sim
    gen_sample_func = model_obj.generate_sample
    gen_param_fun = model_obj.sample_param_values
    grid_param = model_obj.acore_grid
    np.random.seed(seed)

    # Generate a series of parameter values for INFERNO
    param_mat = gen_param_fun(sample_size=diagnostic_sample)
    param_mat[:, 1] = 100.0
    param_mat[:, 2] = 0.75

    # Generate observation from those values
    sample_mat = np.apply_along_axis(
        arr=param_mat, axis=1, func1d=lambda row: gen_obs_func(sample_size=sample_size_obs, true_param=row))

    # Train the odds classifier
    clf_model = classifier_dict[classifier]
    clf_name = classifier
    clf_odds = train_clf(d=model_obj.d, sample_size=b, clf_model=clf_model, gen_function=gen_sample_func,
                         clf_name=clf_name, nn_square_root=True)

    # Compute the nuisance parameters
    theta0_nuisance_mat = np.zeros((diagnostic_sample, 3))
    for kk in range(diagnostic_sample):
        theta0_nuisance_mat[kk, 0] = param_mat[kk, 0]
        theta0_nuisance_mat[kk, 1:] = model_obj.nuisance_parameter_minimization(
            x_obs=sample_mat[kk, :, :], target_params=param_mat[kk, 0].reshape(-1, ), clf_odds=clf_odds)[:2]

    # Then we compute the various grids over which we minimize ACORE or we sample BFF (logavgacore)
    _, acore_grid_out = model_obj.calculate_nuisance_parameters_over_grid(
        t0_grid=grid_param, clf_odds=clf_odds, x_obs=sample_mat[0, :, :])

    # Create the confidence sets now
    tau_obs_vec = np.zeros(diagnostic_sample)
    pbar = tqdm(total=diagnostic_sample, desc='Compute Confidence sets')
    for idx in range(diagnostic_sample):

        # Compute the test statistic
        if test_statistic == 'acore':
            tau_obs = np.array([
                compute_statistics_single_t0(
                    clf=clf_odds, obs_sample=sample_mat[idx, :, :], t0=theta_0, grid_param_t1=acore_grid_out,
                    d=model_obj.d, d_obs=model_obj.d_obs) for theta_0 in theta0_nuisance_mat[idx, :].reshape(1, -1)])
        elif test_statistic == 'logavgacore':
            tau_obs = np.array([
                compute_bayesfactor_single_t0_nuisance(
                    clf=clf_odds, obs_sample=sample_mat[idx, :, :],
                    t0=theta_0[:len(model_obj.target_params_cols)],
                    gen_param_fun=gen_param_fun,
                    d=model_obj.d, d_obs=model_obj.d_obs, log_out=True,
                    d_param_interest=len(model_obj.target_params_cols),
                    monte_carlo_samples=monte_carlo_samples)
                for theta_0 in theta0_nuisance_mat[idx, :].reshape(1, -1)])
        else:
            raise ValueError('The variable test_statistic needs to be either acore or logavgacore. '
                             'Currently %s' % test_statistic)
        tau_obs_vec[idx] = tau_obs
        pbar.update(1)

    # Create the B' sample to check the diagnostic over
    theta_mat_bprime, sample_mat_bprime = model_obj.sample_msnh_algo5(b_prime=b_prime, sample_size=sample_size_obs)
    if test_statistic == 'acore':
        stats_mat_bprime = np.array([compute_statistics_single_t0(
            clf=clf_odds, d=model_obj.d, d_obs=model_obj.d_obs, grid_param_t1=acore_grid_out,
            t0=theta_0, obs_sample=sample_mat_bprime[kk, :, :]) for kk, theta_0 in enumerate(theta_mat_bprime)
        ])
    elif test_statistic == 'logavgacore':
        stats_mat_bprime = np.array([compute_bayesfactor_single_t0_nuisance(
            clf=clf_odds, d=model_obj.d, d_obs=model_obj.d_obs, gen_param_fun=gen_param_fun,
            monte_carlo_samples=monte_carlo_samples, log_out=True,
            t0=theta_0, obs_sample=sample_mat_bprime[kk, :, :],
            d_param_interest=len(model_obj.target_params_cols))
            for kk, theta_0 in enumerate(theta_mat_bprime[:, :len(model_obj.target_params_cols)])
        ])
    else:
        raise ValueError('The variable test_statistic needs to be either acore or logavgacore. '
                         'Currently %s' % test_statistic)

    clf_params = classifier_cde_dict[classifier_cde]
    t0_pred_vec = train_qr_algo(model_obj=model_obj, theta_mat=theta_mat_bprime, stats_mat=stats_mat_bprime,
                                algo_name=clf_params[0], learner_kwargs=clf_params[1],
                                pytorch_kwargs=clf_params[2] if len(clf_params) > 2 else None,
                                alpha=alpha, prediction_grid=theta0_nuisance_mat).reshape(-1, )

    # Compute whether they are in the confidence interval
    in_confint = (tau_obs_vec >= t0_pred_vec).astype(float)
    in_confint_asymptotic = None
    if test_statistic == 'acore':
        chisquare_cutoff = chi2.ppf(q=1.0 - alpha, df=1)
        cutoff_val_asymp = np.array([-0.5 * chisquare_cutoff] * tau_obs_vec.shape[0])
        in_confint_asymptotic = (tau_obs_vec >= cutoff_val_asymp).astype(float)
    t0_vec_coverage_pred = theta0_nuisance_mat[:, 0].reshape(-1, 1)
    predict_mat = np.linspace(start=model_obj.low_int_signal, stop=model_obj.high_int_signal, num=100).reshape(-1, 1)

    # Now we can use some sort of logistic regression/XGboost to compute coverage across the parameter space
    # Calculate the mean with an XGB Classifier
    model = XGBClassifier(**{'max_depth': 3, 'n_estimators': 100})
    model.fit(t0_vec_coverage_pred, in_confint.reshape(-1, ))
    pred_cov_mean = model.predict_proba(predict_mat)[:, 1]
    percent_correct_coverage = np.average((pred_cov_mean > (1.0 - alpha)).astype(int))
    average_coverage = np.average(pred_cov_mean)

    out_dict = {
        'theta_mat': theta0_nuisance_mat,
        'tau_obs_vec': tau_obs_vec,
        'pred_quantile_vec': t0_pred_vec,
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
        'benchmark': benchmark,
        'nuisance': model_obj.nuisance_flag,
        'test_statistic': test_statistic,
        'in_confint_asymptotic': in_confint_asymptotic
    }
    out_dir = 'sims/classifier_power_multid/'
    out_filename = 'hepcounting_b%s_diagnostic_%steststats_%sB_%sBprime_%sseed_alpha%s_sampleobs%s_t0val%s_%s_%s.pkl' % (
        benchmark, test_statistic, b, b_prime, seed,
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
    parser.add_argument('--run', action="store", type=str, default='hep_counting',
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
    parser.add_argument('--class_cde', action="store", type=str, default='pytorch',
                        help='Classifier for quantile regression')
    parser.add_argument('--classifier', action="store", type=str, default='QDA',
                        help='Classifier for learning odds')
    parser.add_argument('--size_marginal', action="store", type=int, default=1000,
                        help='Sample size of the actual marginal distribution, if marginal is True.')
    parser.add_argument('--monte_carlo_samples', action="store", type=int, default=1000,
                        help='Sample size for the calculation of the OR loss.')
    parser.add_argument('--alt_norm', action="store", type=float, default=5,
                        help='Norm of the mean under the alternative -- to be used for toy_mvn_multid_simplehyp only.')
    parser.add_argument('--benchmark', action="store", type=int, default=1,
                        help='Benchmark to use for the INFERNO class.')
    parser.add_argument('--nuisance', action='store_true', default=False,
                        help='If true, uses nuisance parameters if available.')
    parser.add_argument('--diagnostic_sample', action="store", type=int, default=2000,
                        help='Sample for diagnostics meaning.')
    parser.add_argument('--num_grid', action="store", type=int, default=11,
                        help='Number of points in the parameter grid.')
    parser.add_argument('--num_acore_grid', action="store", type=int, default=1000,
                        help='Number of grid points for the grid over which to evaluate ACORE maximization grid.')
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
        test_statistic=argument_parsed.test_statistic,
        classifier_cde=argument_parsed.class_cde,
        size_marginal=argument_parsed.size_marginal,
        monte_carlo_samples=argument_parsed.monte_carlo_samples,
        alternative_norm=argument_parsed.alt_norm,
        benchmark=argument_parsed.benchmark,
        nuisance_parameters=argument_parsed.nuisance,
        classifier=argument_parsed.classifier,
        diagnostic_sample=argument_parsed.diagnostic_sample,
        num_grid=argument_parsed.num_grid,
        num_acore_grid=argument_parsed.num_acore_grid
    )
