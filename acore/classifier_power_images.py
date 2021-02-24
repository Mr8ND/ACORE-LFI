from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import argparse
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

from utils.functions import compute_statistics_single_t0, compute_averageodds_single_t0, compute_bayesfactor_single_t0
from models.galsim import GalSimLoader
from utils.qr_functions import train_qr_algo
from qr_algorithms.complete_list import classifier_cde_dict

model_dict = {
    'galsim': GalSimLoader
}


def main(model_name, d_obs, run, rep, b_prime, alpha,  sample_size_obs, classifier_cde, test_statistic,
         debug=False, seed=7, monte_carlo_samples=500, cuda_flag=False, verbose=False):

    # Changing values if debugging
    b_prime = b_prime if not debug else 100
    rep = rep if not debug else 2

    # We pass as inputs all arguments necessary for all classes, but some of them will not be picked up if they are
    # not necessary for a specific class
    model_obj = model_dict[run](seed=seed, model_name=model_name, cuda_flag=cuda_flag)

    # Get the correct functions
    msnh_sampling_func = model_obj.sample_msnh_algo5
    gen_obs_func = model_obj.sample_sim_true_param
    t0_grid = model_obj.pred_grid
    grid_param = model_obj.acore_grid
    t0_param_val = model_obj.true_param
    true_param_row_idx = model_obj.idx_row_true_param
    clf_odds = model_obj.clf_obj
    gen_param_fun = model_obj.sample_param_values

    # Loop over repetitions and classifiers
    # Each time we train the different QR classifiers (the OR classifier is already trained and loaded),
    # we build the intervals and we record whether the point is in or not.
    out_val = []
    out_cols = ['d_obs', 'test_statistic', 'b_prime', 'classifier_cde', 'run', 'rep',
                'sample_size_obs', 'coverage', 'power', 'size_CI', 'model_name', 'monte_carlo_samples']
    pbar = tqdm(total=rep, desc="Toy Example for Simulations, n=%s, B'=%s" % (sample_size_obs, b_prime))
    for jj in range(rep):

        # Generates samples for each t0 values, so to be able to check both coverage and power
        x_obs = gen_obs_func(sample_size=sample_size_obs)

        if test_statistic == 'acore':
            tau_obs = np.array([
                compute_statistics_single_t0(
                    clf=clf_odds, obs_sample=x_obs, t0=theta_0, grid_param_t1=grid_param,
                    d=model_obj.d, d_obs=model_obj.d_obs) for theta_0 in t0_grid])
        elif test_statistic == 'averageodds':
            tau_obs = np.array([
                compute_averageodds_single_t0(
                    clf=clf_odds, obs_sample=x_obs, t0=theta_0, d=model_obj.d,
                    d_obs=model_obj.d_obs) for theta_0 in t0_grid])
        elif test_statistic == 'logavgacore':
            tau_obs = np.array([
                compute_bayesfactor_single_t0(
                    clf=clf_odds, obs_sample=x_obs, t0=theta_0, gen_param_fun=gen_param_fun,
                    d=model_obj.d, d_obs=model_obj.d_obs, log_out=True) for theta_0 in t0_grid])
        else:
            raise ValueError('The variable test_statistic needs to be either acore, logavgacore, averageodds.'
                             ' Currently %s' % test_statistic)

        # Train the quantile regression algorithm for confidence levels
        theta_mat, sample_mat = msnh_sampling_func(b_prime=b_prime, sample_size=sample_size_obs)

        if test_statistic == 'acore':
            stats_mat = np.array([compute_statistics_single_t0(
                clf=clf_odds, d=model_obj.d, d_obs=model_obj.d_obs, grid_param_t1=grid_param,
                t0=theta_0, obs_sample=sample_mat[kk, :, :]) for kk, theta_0 in enumerate(theta_mat)
            ])
        elif test_statistic == 'averageodds':
            stats_mat = np.array([compute_averageodds_single_t0(clf=clf_odds, d=model_obj.d, d_obs=model_obj.d_obs,
                                                                t0=theta_0, obs_sample=sample_mat[kk, :, :], ) for
                                  kk, theta_0 in enumerate(theta_mat)
                                  ])
        elif test_statistic == 'logavgacore':
            stats_mat = np.array([compute_bayesfactor_single_t0(
                clf=clf_odds, d=model_obj.d, d_obs=model_obj.d_obs, gen_param_fun=gen_param_fun,
                monte_carlo_samples=monte_carlo_samples, log_out=True,
                t0=theta_0, obs_sample=sample_mat[kk, :, :]) for kk, theta_0 in enumerate(theta_mat)
            ])
        else:
            raise ValueError('The variable test_statistic needs to be either acore or averageodds. '
                             'Currently %s' % test_statistic)

        # Train the QR algorithms
        clf_name_qr = classifier_cde
        clf_params = classifier_cde_dict[classifier_cde]
        t0_pred_vec = train_qr_algo(model_obj=model_obj, theta_mat=theta_mat, stats_mat=stats_mat,
                                    algo_name=clf_params[0], learner_kwargs=clf_params[1],
                                    pytorch_kwargs=clf_params[2] if len(clf_params) > 2 else None,
                                    alpha=alpha, prediction_grid=t0_grid)

        # Record all information
        in_confint = (np.delete(tau_obs, [true_param_row_idx]) >=
                      np.delete(t0_pred_vec, [true_param_row_idx])).astype(float)
        size_temp = np.mean(in_confint)
        coverage = int(tau_obs[true_param_row_idx] >= t0_pred_vec[true_param_row_idx])
        power = 1 - in_confint if isinstance(in_confint, float) else (in_confint.shape[0] -
                                                                      np.sum(in_confint)) / in_confint.shape[0]
        out_val.append([
            d_obs, test_statistic, b_prime, clf_name_qr, run, jj, sample_size_obs,
            coverage, power, size_temp, model_name, monte_carlo_samples
        ])

        pbar.update(1)

    # Saving the results
    out_df = pd.DataFrame.from_records(data=out_val, index=range(len(out_val)), columns=out_cols)
    out_dir = 'sims/classifier_power_images/'
    out_filename = 'd%s_%steststats_%sBprime_%s_%s_%srep_alpha%s_sampleobs%s_%s_%s.csv' % (
        d_obs, test_statistic, b_prime, run, model_name, rep,
        str(alpha).replace('.', '-'), sample_size_obs, classifier_cde,
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
    parser.add_argument('--b_prime', action="store", type=int, default=1000,
                        help='Sample size to train the quantile regression algorithm')
    parser.add_argument('--alpha', action="store", type=float, default=0.1,
                        help='Statistical confidence level')
    parser.add_argument('--run', action="store", type=str, default='galsim',
                        help='Problem to run.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If true, a very small value for the sample sizes is fit to make sure the'
                             'file can run quickly for debugging purposes')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='If true, logs are printed to the terminal')
    parser.add_argument('--sample_size_obs', action="store", type=int, default=1,
                        help='Sample size of the actual observed data.')
    parser.add_argument('--test_statistic', action="store", type=str, default='acore',
                        help='Type of ACORE test statistic to use.')
    parser.add_argument('--class_cde', action="store", type=str, default='pytorch',
                        help='Classifier for quantile regression')
    parser.add_argument('--model_name', action="store", type=str, default='alexnet',
                        help='Name for the pre-trained CNN classifier to be used.')
    parser.add_argument('--monte_carlo_samples', action="store", type=int, default=1000,
                        help='Sample size for the calculation of the OR loss.')
    parser.add_argument('--cuda_flag', action='store_true', default=False,
                        help='If true, uses a GPU if available.')
    argument_parsed = parser.parse_args()

    main(
        d_obs=argument_parsed.d_obs,
        run=argument_parsed.run,
        rep=argument_parsed.rep,
        b_prime=argument_parsed.b_prime,
        alpha=argument_parsed.alpha,
        debug=argument_parsed.debug,
        sample_size_obs=argument_parsed.sample_size_obs,
        seed=argument_parsed.seed,
        verbose=argument_parsed.verbose,
        test_statistic=argument_parsed.test_statistic,
        classifier_cde=argument_parsed.class_cde,
        model_name=argument_parsed.model_name,
        monte_carlo_samples=argument_parsed.monte_carlo_samples,
        cuda_flag=argument_parsed.cuda_flag
    )
