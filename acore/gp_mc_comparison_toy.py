from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import argparse
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from utils.functions import train_clf, compute_statistics_single_t0
from models.toy_poisson import ToyPoissonLoader
from models.toy_gmm import ToyGMMLoader
from models.toy_gamma import ToyGammaLoader
from utils.qr_functions import train_qr_algo
from or_classifiers.toy_example_list import classifier_dict
from qr_algorithms.complete_list import classifier_cde_dict
from utils.gp_functions import train_gp, compute_statistics_single_t0_gp
from scipy.stats import chi2

model_dict = {
    'poisson': ToyPoissonLoader,
    'gmm': ToyGMMLoader,
    'gamma': ToyGammaLoader
}


def main(run, rep, b, b_prime, alpha, sample_size_obs, classifier_cde, sample_type='MC', cutoff='qr',
         debug=False, seed=7, size_check=1000, verbose=False, marginal=False, size_marginal=1000):

    # Changing values if debugging
    b = b if not debug else 100
    b_prime = b_prime if not debug else 100
    size_check = size_check if not debug else 100
    rep = rep if not debug else 2
    model_obj = model_dict[run](marginal=marginal, size_marginal=size_marginal)

    # Get the correct functions
    msnh_sampling_func = model_obj.sample_msnh_algo5
    grid_param = model_obj.grid
    gen_obs_func = model_obj.sample_sim
    gen_sample_func = model_obj.generate_sample
    t0_grid = model_obj.pred_grid
    t0_val = model_obj.true_param
    lik_func = model_obj.compute_exact_likelihood
    np.random.seed(seed)

    # Adding Gaussian Process as an option in the classifier toy example
    anchor_points_vec = [5, 10, 25]
    for anchor_points in anchor_points_vec:
        classifier_dict['gaussian_process_' + str(anchor_points)] = anchor_points

    # Loop over repetitions and classifiers
    # Each time we train the different classifiers, we build the intervals and we record
    # whether the point is in or not.
    out_val = []
    out_cols = ['b_prime', 'b', 'classifier', 'classifier_cde', 'run', 'rep', 'sample_size_obs',
                't0_true_val', 'theta_0_current', 'on_true_t0',
                'estimated_tau', 'estimated_cutoff', 'in_confint', 'out_confint', 'size_CI', 'mse_loss',
                'training_time', 'pred_time', 'bprime_time', 'cutoff_time', 'total_time']
    pbar = tqdm(total=rep, desc='Toy Example for Simulations, n=%s, b=%s' % (sample_size_obs, b))
    for jj in range(rep):

        # Generates samples for each t0 values, so to be able to check both coverage and power
        x_obs = gen_obs_func(sample_size=sample_size_obs, true_param=t0_val)

        # Calculate the true likelihood ratio
        lik_theta0 = np.array([np.sum(np.log(lik_func(x_obs=x_obs, true_param=theta_0))) for theta_0 in t0_grid])
        max_across_grid = np.max(np.array([np.sum(np.log(lik_func(x_obs=x_obs, true_param=t1))) for t1 in grid_param]))
        true_tau_obs = lik_theta0.reshape(-1, ) - max_across_grid.reshape(1)
        # print('TRUE', true_tau_obs)

        # Train the classifier for the odds
        clf_odds_fitted = {}
        clf_cde_fitted = {}
        for clf_name, clf_model in sorted(classifier_dict.items(), key=lambda x: x[0]):
            start_time = datetime.now()

            if 'gaussian_process' in clf_name:

                # Train Gaussian Process
                gp_model = train_gp(sample_size=b, n_anchor_points=clf_model, model_obj=model_obj, t0_grid=t0_grid,
                                    sample_type=sample_type)
                training_time = datetime.now()

                # Calculate LR given a Gaussian Process
                tau_obs = np.array([
                    compute_statistics_single_t0_gp(
                        gp_model=gp_model, obs_sample=x_obs, t0=theta_0, grid_param_t1=grid_param,
                        d=model_obj.d, d_obs=model_obj.d_obs) for theta_0 in t0_grid])
                clf_odds_fitted[clf_name] = (tau_obs, np.mean((tau_obs - true_tau_obs)**2))
                # print(clf_name, clf_odds_fitted[clf_name])
                pred_time = datetime.now()

                # Calculate the LR statistics given a sample
                if cutoff == 'qr':
                    theta_mat, sample_mat = msnh_sampling_func(b_prime=b_prime, sample_size=sample_size_obs)
                    full_mat = np.hstack((theta_mat, sample_mat))
                    stats_mat = np.apply_along_axis(arr=full_mat, axis=1,
                                                    func1d=lambda row: compute_statistics_single_t0_gp(
                                                        gp_model=gp_model,
                                                        obs_sample=row[model_obj.d:],
                                                        t0=row[:model_obj.d],
                                                        grid_param_t1=grid_param,
                                                        d=model_obj.d,
                                                        d_obs=model_obj.d_obs
                                                    ))
                    bprime_time = datetime.now()

                    clf_cde_fitted[clf_name] = {}
                    # for clf_name_qr, clf_params in sorted(classifier_cde_dict.items(), key=lambda x: x[0]):
                    clf_name_qr = classifier_cde
                    clf_params = classifier_cde_dict[classifier_cde]
                    t0_pred_vec = train_qr_algo(model_obj=model_obj, theta_mat=theta_mat, stats_mat=stats_mat,
                                                algo_name=clf_params[0], learner_kwargs=clf_params[1],
                                                pytorch_kwargs=clf_params[2] if len(clf_params) > 2 else None,
                                                alpha=alpha, prediction_grid=t0_grid)
                elif cutoff == 'chisquare':
                    chisquare_cutoff = chi2.ppf(q=1.0-alpha, df=1)
                    t0_pred_vec = np.array([-0.5 * chisquare_cutoff] * tau_obs.shape[0])

                    bprime_time = datetime.now()
                    clf_name_qr = classifier_cde
                    clf_cde_fitted[clf_name] = {}
                else:
                    raise ValueError('Cutoff %s not recognized. Either "qr" or "chisquare" are accepted' % cutoff)

            else:
                clf_odds = train_clf(sample_size=b, clf_model=clf_model, gen_function=gen_sample_func,
                                     clf_name=clf_name, marginal=marginal, nn_square_root=True)
                training_time = datetime.now()

                if verbose:
                    print('----- %s Trained' % clf_name)
                tau_obs = np.array([
                    compute_statistics_single_t0(
                        clf=clf_odds, obs_sample=x_obs, t0=theta_0, grid_param_t1=grid_param,
                        d=model_obj.d, d_obs=model_obj.d_obs) for theta_0 in t0_grid])
                clf_odds_fitted[clf_name] = (tau_obs, np.mean((tau_obs - true_tau_obs)**2))
                # print(clf_name, clf_odds_fitted[clf_name])
                pred_time = datetime.now()

                # Train the quantile regression algorithm for confidence levels
                theta_mat, sample_mat = msnh_sampling_func(b_prime=b_prime, sample_size=sample_size_obs)
                full_mat = np.hstack((theta_mat, sample_mat))
                stats_mat = np.apply_along_axis(arr=full_mat, axis=1,
                                                func1d=lambda row: compute_statistics_single_t0(
                                                    clf=clf_odds,
                                                    obs_sample=row[model_obj.d:],
                                                    t0=row[:model_obj.d],
                                                    grid_param_t1=grid_param,
                                                    d=model_obj.d,
                                                    d_obs=model_obj.d_obs
                                                ))
                bprime_time = datetime.now()

                clf_cde_fitted[clf_name] = {}
                # for clf_name_qr, clf_params in sorted(classifier_cde_dict.items(), key=lambda x: x[0]):
                clf_name_qr = classifier_cde
                clf_params = classifier_cde_dict[classifier_cde]
                t0_pred_vec = train_qr_algo(model_obj=model_obj, theta_mat=theta_mat, stats_mat=stats_mat,
                                            algo_name=clf_params[0], learner_kwargs=clf_params[1],
                                            pytorch_kwargs=clf_params[2] if len(clf_params) > 2 else None,
                                            alpha=alpha, prediction_grid=t0_grid)

            cutoff_time = datetime.now()
            clf_cde_fitted[clf_name][clf_name_qr] = (
                t0_pred_vec, ((training_time - start_time).total_seconds() * 100,
                              (pred_time - training_time).total_seconds() * 100,
                              (bprime_time - pred_time).total_seconds() * 100,
                              (cutoff_time - bprime_time).total_seconds() * 100))

        # At this point all it's left is to record
        for clf_name, (tau_obs_val, mse_val) in clf_odds_fitted.items():
            for clf_name_qr, (cutoff_val, time_vec) in clf_cde_fitted[clf_name].items():
                size_temp = np.sum((tau_obs_val >= cutoff_val).astype(int))/t0_grid.shape[0]
                for kk, theta_0_current in enumerate(t0_grid):
                    out_val.append([
                        b_prime, b, clf_name, clf_name_qr, run, jj, sample_size_obs,
                        t0_val, theta_0_current, int(t0_val == theta_0_current),
                        tau_obs_val[kk], cutoff_val[kk], int(tau_obs_val[kk] > cutoff_val[kk]),
                        int(tau_obs_val[kk] <= cutoff_val[kk]), size_temp, mse_val,
                        time_vec[0], time_vec[1], time_vec[2], time_vec[3], sum(time_vec)
                    ])
        pbar.update(1)

    # Saving the results
    out_df = pd.DataFrame.from_records(data=out_val, index=range(len(out_val)), columns=out_cols)
    out_dir = 'sims/gp_mc_comparison/'
    out_filename = 'classifier_reps_gp_mc_comparison_%sB_%sBprime_%s_%srep_alpha%s_sampleobs%s_t0val%s_%s_%s.csv' % (
        b, b_prime, run, rep, str(alpha).replace('.', '-'), sample_size_obs,
        str(t0_val).replace('.', '-'), classifier_cde,
        datetime.strftime(datetime.today(), '%Y-%m-%d')
    )
    out_df.to_csv(out_dir + out_filename)

    # Print results
    cov_df = out_df[out_df['on_true_t0'] == 1][['classifier', 'classifier_cde',
                                                'in_confint', 'mse_loss', 'size_CI',
                                                'training_time', 'pred_time', 'bprime_time', 'cutoff_time',
                                                'total_time']]
    print(cov_df.groupby(['classifier', 'classifier_cde']).agg({'in_confint': [np.average],
                                                                'size_CI': [np.average, np.std],
                                                                'mse_loss': [np.average, np.std],
                                                                'training_time': [np.average, np.std],
                                                                'pred_time': [np.average, np.std],
                                                                'bprime_time': [np.average, np.std],
                                                                'cutoff_time': [np.average, np.std],
                                                                'total_time': [np.average, np.std]}))

    # Power plots
    out_df['class_combo'] = out_df[['classifier', 'classifier_cde']].apply(lambda x: x[0] + '---' + x[1], axis = 1)
    plot_df = out_df[['class_combo', 'theta_0_current', 'out_confint']].groupby(
        ['class_combo', 'theta_0_current']).mean().reset_index()
    fig = plt.figure(figsize=(20, 10))
    sns.lineplot(x='theta_0_current', y='out_confint', hue='class_combo', data=plot_df, palette='cubehelix')
    plt.legend(loc='best', fontsize=25)
    plt.xlabel(r'$\theta$', fontsize=25)
    plt.ylabel('Power', fontsize=25)
    plt.title("Power of Hypothesis Test, B=%s, B'=%s, n=%s, %s" % (
        b, b_prime, sample_size_obs, run.title()), fontsize=25)
    out_dir = 'images/gp_mc_comparison/'
    outfile_name = 'power_gp_mc_comparison_reps_%sB_%sBprime_%s_%srep_alpha%s_sampleobs%s_t0val%s_%s_%s.pdf' % (
        b, b_prime, run, rep, str(alpha).replace('.', '-'), sample_size_obs,
        str(t0_val).replace('.', '-'), classifier_cde,
        datetime.strftime(datetime.today(), '%Y-%m-%d')
    )
    plt.tight_layout()
    plt.savefig(out_dir + outfile_name)
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', action="store", type=int, default=7,
                        help='Random State')
    parser.add_argument('--rep', action="store", type=int, default=10,
                        help='Number of Repetitions for calculating the Pinball loss')
    parser.add_argument('--b', action="store", type=int, default=5000,
                        help='Sample size to train the classifier for calculating odds')
    parser.add_argument('--b_prime', action="store", type=int, default=1000,
                        help='Sample size to train the quantile regression algorithm')
    parser.add_argument('--marginal', action='store_true', default=False,
                        help='Whether we are using a parametric approximation of the marginal or'
                             'the baseline reference G')
    parser.add_argument('--alpha', action="store", type=float, default=0.1,
                        help='Statistical confidence level')
    parser.add_argument('--run', action="store", type=str, default='poisson',
                        help='Problem to run')
    parser.add_argument('--sample_type', action="store", type=str, default='MC',
                        help='Sampling type for the Gaussian Process. MC means Monte Carlo, so it selects a number'
                             ' of anchor points and samples sample_size/anchor points samples there, otherwise it'
                             ' samples points uniformly.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If true, a very small value for the sample sizes is fit to make sure the'
                             'file can run quickly for debugging purposes')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='If true, logs are printed to the terminal')
    parser.add_argument('--sample_size_obs', action="store", type=int, default=10,
                        help='Sample size of the actual observed data.')
    parser.add_argument('--t0_val', action="store", type=float, default=10.0,
                        help='True parameter which generates the observed dataset')
    parser.add_argument('--class_cde', action="store", type=str, default='xgb_d3_n100',
                        help='Classifier for quantile regression')
    parser.add_argument('--size_marginal', action="store", type=int, default=1000,
                        help='Sample size of the actual marginal distribution, if marginal is True.')
    parser.add_argument('--or_loss_samples', action="store", type=int, default=1000,
                        help='Sample size for the calculation of the OR loss.')
    parser.add_argument('--cutoff', action="store", type=str, default='qr',
                        help='How to obtain the cutoff approximation: either qr (quantile regression estimation)'
                             ' or chisquare (chisquare approximation via Wilks theorem)')
    # parser.add_argument('--n_anchor', action="store", type=int, default=5,
    #                     help='Number of Gaussian Process anchor points in the theta space.')
    argument_parsed = parser.parse_args()

    b_vec = [100, 500, 1000]
    for b_val in b_vec:
        main(
            run=argument_parsed.run,
            rep=argument_parsed.rep,
            marginal=argument_parsed.marginal,
            b=b_val,      # argument_parsed.b,
            b_prime=argument_parsed.b_prime,
            alpha=argument_parsed.alpha,
            debug=argument_parsed.debug,
            sample_size_obs=argument_parsed.sample_size_obs,
            seed=argument_parsed.seed,
            verbose=argument_parsed.verbose,
            classifier_cde=argument_parsed.class_cde,
            size_marginal=argument_parsed.size_marginal,
            sample_type=argument_parsed.sample_type,
            cutoff=argument_parsed.cutoff
        )
