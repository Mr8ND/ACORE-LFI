from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import argparse
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.metrics import log_loss
from functools import partial

from utils.functions import train_clf, compute_statistics_single_t0, clf_prob_value, compute_bayesfactor_single_t0, \
    compute_averageodds_single_t0, odds_ratio_loss, sample_from_matrix
from models.toy_gmm_multid import ToyGMMMultiDLoader
from models.toy_mvn import ToyMVNLoader
from models.toy_mvn_simplehyp import ToyMVNSimpleHypLoader
from models.toy_mvn_multid import ToyMVNMultiDLoader
from models.toy_mvn_multid_simplehyp import ToyMVNMultiDSimpleHypLoader
from models.inferno import InfernoToyLoader
from utils.qr_functions import train_qr_algo
from or_classifiers.toy_example_list import classifier_dict_multid, classifier_inferno_dict, classifier_dict_multid_power
from qr_algorithms.complete_list import classifier_cde_dict

model_dict = {
    # 'gmm': ToyGMMMultiDLoader,
    'mvn': ToyMVNLoader,
    'mvn_simplehyp': ToyMVNSimpleHypLoader,
    'mvn_multid': ToyMVNMultiDLoader,
    'mvn_multid_simplehyp': ToyMVNMultiDSimpleHypLoader,
    'inferno': InfernoToyLoader
}


def main(d_obs, run, rep, b, b_prime, alpha, t0_val, sample_size_obs, classifier_cde, test_statistic, alternative_norm,
         monte_carlo_samples=500, debug=False, seed=7, size_check=1000, verbose=False, marginal=False,
         size_marginal=1000, empirical_marginal=True, benchmark=1, nuisance_parameters=False, nuisance_confint=False,
         guided_sim=False, guided_sample=1000, exactlr_mc=1000):

    # Changing values if debugging
    b = b if not debug else 100
    b_prime = b_prime if not debug else 100
    size_check = size_check if not debug else 100
    rep = rep if not debug else 2
    #classifier_dict = classifier_dict_multid if 'inferno' not in run else classifier_inferno_dict
    classifier_dict = classifier_dict_multid_power if 'inferno' not in run else classifier_inferno_dict

    # We pass as inputs all arguments necessary for all classes, but some of them will not be picked up if they are
    # not necessary for a specific class
    model_obj = model_dict[run](
        d_obs=d_obs, marginal=marginal, size_marginal=size_marginal, empirical_marginal=empirical_marginal,
        true_param=t0_val, alt_mu_norm=alternative_norm, nuisance_parameters=nuisance_parameters,
        benchmark=benchmark
    )

    # Get the correct functions
    msnh_sampling_func = model_obj.sample_msnh_algo5
    gen_obs_func = model_obj.sample_sim
    gen_sample_func = model_obj.generate_sample
    gen_param_fun = model_obj.sample_param_values
    t0_grid = model_obj.pred_grid
    grid_param = model_obj.acore_grid
    tp_func = model_obj.compute_exact_prob
    t0_param_val = model_obj.true_param
    true_param_row_idx = model_obj.idx_row_true_param
    if run == 'mvn_multid' and nuisance_parameters:
        compute_exactodds_nuisance_single_t0 = model_obj.compute_exactodds_nuisance_single_t0

    # Specific case for INFERNO with exact LR and nuisance parameters
    if run == 'inferno' and test_statistic == 'exactlr':
        compute_exactlr_single_t0 = model_obj.compute_exactlr_single_t0
        compute_exactlr_msnh_t0 = model_obj.compute_exactlr_msnh_t0
        compute_exactlr_distribution_t0 = model_obj.compute_exactlr_distribution_t0

    # Creating sample to check entropy about
    np.random.seed(seed)
    sample_check = gen_sample_func(sample_size=size_check)
    theta_vec = sample_check[:, :model_obj.d]
    x_vec = sample_check[:, (model_obj.d + 1):]
    bern_vec = sample_check[:, model_obj.d]

    true_prob_vec = tp_func(theta_vec=theta_vec, x_vec=x_vec)
    entropy_est = -np.average([np.log(true_prob_vec[kk]) if el == 1
                               else np.log(1 - true_prob_vec[kk])
                               for kk, el in enumerate(bern_vec)])

    # Loop over repetitions and classifiers
    # Each time we train the different classifiers, we build the intervals and we record
    # whether the point is in or not.
    out_val = []
    out_cols = ['d_obs', 'test_statistic', 'b_prime', 'b', 'classifier', 'classifier_cde', 'run', 'rep', 'sample_size_obs',
                'cross_entropy_loss', 't0_true_val', 'coverage', 'power', 'size_CI', 'true_entropy', 'or_loss_value',
                'monte_carlo_samples', 'benchmark', 'nuisance_parameters', 'alternative_mu_norm', 'guided_sim',
                'guided_sample']
    pbar = tqdm(total=rep, desc='Toy Example for Simulations, n=%s, b=%s' % (sample_size_obs, b))
    for jj in range(rep):

        # Generates samples for each t0 values, so to be able to check both coverage and power
        x_obs = gen_obs_func(sample_size=sample_size_obs, true_param=t0_param_val)

        # Train the classifier for the odds
        clf_odds_fitted = {}
        clf_cde_fitted = {}
        for clf_name, clf_model in sorted(classifier_dict.items(), key=lambda x: x[0]):
            clf_odds = train_clf(d=model_obj.d, sample_size=b, clf_model=clf_model, gen_function=gen_sample_func,
                                 clf_name=clf_name, nn_square_root=True)
            if verbose:
                print('----- %s Trained' % clf_name)

            if model_obj.nuisance_flag:
                t0_grid, acore_grid = model_obj.calculate_nuisance_parameters_over_grid(
                    t0_grid=model_obj.pred_grid, clf_odds=clf_odds, x_obs=x_obs)
                gen_param_fun = partial(sample_from_matrix, t0_grid=t0_grid)
                grid_param = acore_grid

            if test_statistic == 'acore':
                tau_obs = np.array([
                    compute_statistics_single_t0(
                        clf=clf_odds, obs_sample=x_obs, t0=theta_0, grid_param_t1=grid_param,
                        d=model_obj.d, d_obs=model_obj.d_obs) for theta_0 in t0_grid])
            elif test_statistic == 'avgacore':
                tau_obs = np.array([
                    compute_bayesfactor_single_t0(
                        clf=clf_odds, obs_sample=x_obs, t0=theta_0, gen_param_fun=gen_param_fun,
                        d=model_obj.d, d_obs=model_obj.d_obs) for theta_0 in t0_grid])
            elif test_statistic == 'logavgacore':
                tau_obs = np.array([
                    compute_bayesfactor_single_t0(
                        clf=clf_odds, obs_sample=x_obs, t0=theta_0, gen_param_fun=gen_param_fun,
                        d=model_obj.d, d_obs=model_obj.d_obs, log_out=True) for theta_0 in t0_grid])
            elif test_statistic == 'averageodds':
                tau_obs = np.array([
                    compute_averageodds_single_t0(
                        clf=clf_odds, obs_sample=x_obs, t0=theta_0, d=model_obj.d,
                        d_obs=model_obj.d_obs) for theta_0 in t0_grid])
            elif test_statistic == 'exactodds_nuisance':
                tau_obs = np.array([
                    compute_exactodds_nuisance_single_t0(
                        obs_sample=x_obs, t0=theta_0) for theta_0 in t0_grid])
            elif test_statistic == 'exactlr':
                tau_obs = compute_exactlr_single_t0(
                        obs_sample=x_obs, t0_grid=t0_grid, grid_param=grid_param)
            else:
                raise ValueError('The variable test_statistic needs to be either acore, avgacore, logavgacore, '
                                 'exactodds_nuisance, exactlr. Currently %s' % test_statistic)

            # Calculating cross-entropy
            est_prob_vec = clf_prob_value(clf=clf_odds, x_vec=x_vec, theta_vec=theta_vec, d=model_obj.d,
                                          d_obs=model_obj.d_obs)
            loss_value = log_loss(y_true=bern_vec, y_pred=est_prob_vec)

            # Calculating or loss
            or_loss_value = odds_ratio_loss(clf=clf_odds, x_vec=x_vec, theta_vec=theta_vec,
                                            bern_vec=bern_vec, d=model_obj.d, d_obs=model_obj.d_obs)
            clf_odds_fitted[clf_name] = (tau_obs, loss_value, or_loss_value)

            # Train the quantile regression algorithm for confidence levels
            if guided_sim:
                # We now sample a set of thetas from the parameter (set to be 25% of the b_prime)
                # budget, then resample them according to the odds values, fit a gaussian and then sample the
                # datasets from that.
                theta_mat_sample = gen_param_fun(sample_size=guided_sample)

                if test_statistic == 'acore':
                    stats_sample = np.apply_along_axis(arr=theta_mat_sample.reshape(-1, model_obj.d), axis=1,
                                                       func1d=lambda row: compute_statistics_single_t0(
                                                           clf=clf_odds,
                                                           obs_sample=x_obs,
                                                           t0=row,
                                                           grid_param_t1=grid_param,
                                                           d=model_obj.d,
                                                           d_obs=model_obj.d_obs
                                                       ))
                elif test_statistic == 'avgacore':
                    stats_sample = np.apply_along_axis(arr=theta_mat_sample.reshape(-1, model_obj.d), axis=1,
                                                       func1d=lambda row: compute_bayesfactor_single_t0(
                                                           clf=clf_odds,
                                                           obs_sample=x_obs,
                                                           t0=row,
                                                           gen_param_fun=gen_param_fun,
                                                           d=model_obj.d,
                                                           d_obs=model_obj.d_obs,
                                                           monte_carlo_samples=monte_carlo_samples
                                                       ))
                elif test_statistic == 'logavgacore':
                    stats_sample = np.apply_along_axis(arr=theta_mat_sample.reshape(-1, model_obj.d), axis=1,
                                                       func1d=lambda row: compute_bayesfactor_single_t0(
                                                           clf=clf_odds,
                                                           obs_sample=x_obs,
                                                           t0=row,
                                                           gen_param_fun=gen_param_fun,
                                                           d=model_obj.d,
                                                           d_obs=model_obj.d_obs,
                                                           monte_carlo_samples=monte_carlo_samples,
                                                           log_out=True
                                                       ))
                elif test_statistic == 'averageodds':
                    stats_sample = np.apply_along_axis(arr=theta_mat_sample.reshape(-1, model_obj.d), axis=1,
                                                       func1d=lambda row: compute_averageodds_single_t0(
                                                           clf=clf_odds,
                                                           obs_sample=x_obs,
                                                           t0=row,
                                                           d=model_obj.d,
                                                           d_obs=model_obj.d_obs
                                                       ))
                elif test_statistic == 'exactodds_nuisance':
                    stats_sample = np.apply_along_axis(arr=theta_mat_sample.reshape(-1, model_obj.d), axis=1,
                                                       func1d=lambda row: compute_exactodds_nuisance_single_t0(
                                                           obs_sample=x_obs,
                                                           t0=row
                                                       ))
                elif test_statistic == 'exactlr':
                    # stats_sample = compute_exactlr_single_t0(
                    #     obs_sample=x_obs, t0_grid=theta_mat_sample.reshape(-1, model_obj.d), grid_param=grid_param)
                    t0_pred_vec = compute_exactlr_distribution_t0(
                        prediction_grid=t0_grid, monte_carlo_samples=exactlr_mc, sample_size_obs=sample_size_obs, alpha=alpha)
                else:
                    raise ValueError('The variable test_statistic needs to be either acore, avgacore,'
                                     ' logavgacore, exactodds_nuisance or exactlr. '
                                     'Currently %s' % test_statistic)

                # If there are log-odds, then some of the values might be negative, so we need to exponentiate them
                # so to make sure that the large negative numbers are counted correctly (i.e. as very low probability,
                # not probabilities with large magnitudes).
                if test_statistic in ['acore', 'logavgacore']:
                    stats_sample = np.exp(stats_sample)
                stats_sample = stats_sample / np.sum(stats_sample)
                theta_mat_gaussian_fit_idx = np.random.choice(a=theta_mat_sample.shape[0], p=stats_sample.reshape(-1, ),
                                                              size=guided_sample)
                theta_mat_gaussian_fit = theta_mat_sample[theta_mat_gaussian_fit_idx, :]
                mean_gaussian_fit = np.mean(theta_mat_gaussian_fit, axis=0)
                if run in ['mvn', 'mvn_simplehyp']:
                    std_gaussian_fit = np.std(theta_mat_gaussian_fit)
                    theta_mat = np.random.normal(
                        size=b_prime, loc=mean_gaussian_fit, scale=std_gaussian_fit).clip(
                        min=model_obj.low_int, max=model_obj.high_int)
                else:
                    cov_gaussian_fit = np.cov(theta_mat_gaussian_fit, rowvar=False)
                    theta_mat = np.random.multivariate_normal(
                        size=b_prime, mean=mean_gaussian_fit, cov=cov_gaussian_fit, ).clip(
                        min=model_obj.low_int, max=model_obj.high_int)
                sample_mat = np.apply_along_axis(arr=theta_mat.reshape(-1, model_obj.d), axis=1,
                                                 func1d=lambda row: gen_obs_func(sample_size=sample_size_obs,
                                                                                 true_param=row))
            else:
                theta_mat, sample_mat = msnh_sampling_func(b_prime=b_prime, sample_size=sample_size_obs)
            
            if test_statistic == 'acore':
                stats_mat = np.array([compute_statistics_single_t0(
                    clf=clf_odds, d=model_obj.d, d_obs=model_obj.d_obs, grid_param_t1=grid_param,
                    t0=theta_0, obs_sample=sample_mat[kk, :, :]) for kk, theta_0 in enumerate(theta_mat)
                ])
            elif test_statistic == 'avgacore':
                stats_mat = np.array([compute_bayesfactor_single_t0(
                                      clf=clf_odds, d=model_obj.d, d_obs=model_obj.d_obs, gen_param_fun=gen_param_fun,
                                      monte_carlo_samples=monte_carlo_samples,
                                      t0=theta_0, obs_sample=sample_mat[kk, :, :]) for kk, theta_0 in enumerate(theta_mat)
                                     ])
            elif test_statistic == 'logavgacore':
                stats_mat = np.array([compute_bayesfactor_single_t0(
                                      clf=clf_odds, d=model_obj.d, d_obs=model_obj.d_obs, gen_param_fun=gen_param_fun,
                                      monte_carlo_samples=monte_carlo_samples, log_out=True,
                                      t0=theta_0, obs_sample=sample_mat[kk, :, :]) for kk, theta_0 in enumerate(theta_mat)
                                     ])
            elif test_statistic == 'averageodds':
                stats_mat = np.array([compute_averageodds_single_t0(clf=clf_odds, d=model_obj.d, d_obs=model_obj.d_obs,
                                      t0=theta_0, obs_sample=sample_mat[kk, :, :], ) for kk, theta_0 in enumerate(theta_mat)
                                    ])
            elif test_statistic == 'exactodds_nuisance':
                stats_mat = np.array([compute_exactodds_nuisance_single_t0(t0=theta_0,
                                      obs_sample=sample_mat[kk, :, :], ) for kk, theta_0 in enumerate(theta_mat)
                                    ])
            elif test_statistic == 'exactlr':
                # stats_mat = compute_exactlr_msnh_t0(
                #     t0_grid=theta_mat, sample_mat=sample_mat, grid_param=grid_param)
                t0_pred_vec = compute_exactlr_distribution_t0(
                    prediction_grid=t0_grid, monte_carlo_samples=exactlr_mc, sample_size_obs=sample_size_obs, alpha=alpha)
            else:
                raise ValueError('The variable test_statistic needs to be either acore, avgacore, logavgacore, '
                                 'exactodds_nuisance or exactlr. Currently %s' % test_statistic)

            clf_cde_fitted[clf_name] = {}
            clf_name_qr = classifier_cde
            clf_params = classifier_cde_dict[classifier_cde]
            if test_statistic != 'exactlr':
                t0_pred_vec = train_qr_algo(model_obj=model_obj, theta_mat=theta_mat, stats_mat=stats_mat,
                                            algo_name=clf_params[0], learner_kwargs=clf_params[1],
                                            pytorch_kwargs=clf_params[2] if len(clf_params) > 2 else None,
                                            alpha=alpha, prediction_grid=t0_grid)
            clf_cde_fitted[clf_name][clf_name_qr] = t0_pred_vec

        # At this point all it's left is to record
        for clf_name, (tau_obs_val, cross_ent_loss, or_loss_value) in clf_odds_fitted.items():
            for clf_name_qr, cutoff_val in clf_cde_fitted[clf_name].items():
                in_confint = (np.delete(tau_obs_val, [true_param_row_idx]) >=
                              np.delete(cutoff_val, [true_param_row_idx])).astype(float)
                size_temp = np.mean(in_confint)
                coverage = int(tau_obs_val[true_param_row_idx] >= cutoff_val[true_param_row_idx])
                power = 1 - in_confint if isinstance(in_confint, float) else (in_confint.shape[0] - 
                                                                  np.sum(in_confint)) / in_confint.shape[0]
                out_val.append([
                    d_obs, test_statistic, b_prime, b, clf_name, clf_name_qr, run, jj, sample_size_obs,
                    cross_ent_loss, t0_val, coverage, power,
                    size_temp, entropy_est, or_loss_value, monte_carlo_samples, benchmark, int(nuisance_parameters),
                    alternative_norm, int(guided_sim), guided_sample
                ])
                
                #### RECORD for each tau_obs_val point: its location (d1, d2), tau value, cutoff value, decision (1 or 0)
                if nuisance_confint:
                    plot_df = pd.DataFrame.from_dict({
                        'theta1': t0_grid[:, 0],
                        'theta2': t0_grid[:, 1],
                        'tau_statistic': tau_obs_val.reshape(-1, ),
                        'cutoff': cutoff_val.reshape(-1, ),
                        'decision': (tau_obs_val.reshape(-1, ) >= cutoff_val.reshape(-1, )).astype(float)
                        })

                    pkl_dir = 'sims/classifier_power_multid/plot_df/'
                    pkl_filename = 'nuisance_%s_d%s_%s_%sB_%sBprime_%s.pkl' % (
                        run, d_obs, test_statistic, b, b_prime,
                        datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
                    )
                    plot_df.to_pickle(pkl_dir + pkl_filename)
                
        pbar.update(1)

    # Saving the results
    out_df = pd.DataFrame.from_records(data=out_val, index=range(len(out_val)), columns=out_cols)
    out_dir = 'sims/classifier_power_multid/'
    out_filename = 'd%s_%steststats_%sB_%sBprime_%s_%srep_alpha%s_sampleobs%s_t0val%s_%s_%s.csv' % (
        d_obs, test_statistic, b, b_prime, run, rep,
        str(alpha).replace('.', '-'), sample_size_obs,
        str(t0_val).replace('.', '-'), classifier_cde,
        datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    out_df.to_csv(out_dir + out_filename)


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
    parser.add_argument('--run', action="store", type=str, default='mvn_multid',
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
    parser.add_argument('--nuisance_confint', action='store_true', default=False,
                        help='If true, stores plot_df to visualize confidence interval.')
    parser.add_argument('--guided_sim', action='store_true', default=False,
                        help='If true, we guided the sampling for the B prime in order to get meaningful results.')
    parser.add_argument('--guided_sample', action="store", type=int, default=2500,
                        help='The sample size to be used for the guided simulation. Only used if guided_sim is True.')
    parser.add_argument('--exactlr_mc', action="store", type=int, default=1000,
                        help='Monte Carlo samples for the exact likelihood ratio calculations.')
    argument_parsed = parser.parse_args()

    
    main(
        d_obs=argument_parsed.d_obs,
        run=argument_parsed.run,
        rep=argument_parsed.rep,
        marginal=argument_parsed.marginal,
        empirical_marginal=argument_parsed.empirical_marginal,
        b=argument_parsed.b,      # b_val,
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
        alternative_norm=argument_parsed.alt_norm,
        benchmark=argument_parsed.benchmark,
        nuisance_parameters=argument_parsed.nuisance,
        nuisance_confint=argument_parsed.nuisance_confint,
        guided_sim=argument_parsed.guided_sim,
        guided_sample=argument_parsed.guided_sample,
        exactlr_mc=argument_parsed.exactlr_mc
    )
