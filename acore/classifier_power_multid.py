from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import argparse
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.metrics import log_loss
import seaborn as sns
import matplotlib.pyplot as plt

from utils.functions import train_clf, compute_statistics_single_t0, clf_prob_value, or_loss
from models.toy_gmm_multid import ToyGMMMultiDLoader
from utils.qr_functions import train_qr_algo
from or_classifiers.toy_example_list import classifier_dict_multid as classifier_dict
from qr_algorithms.complete_list import classifier_cde_dict

model_dict = {
    'gmm': ToyGMMMultiDLoader
}


def main(d_obs, run, rep, b, b_prime, alpha, sample_size_obs, classifier_cde, or_loss_samples=1000,
         debug=False, seed=7, size_check=1000, verbose=False, marginal=False, size_marginal=1000):

    # Changing values if debugging
    b = b if not debug else 100
    b_prime = b_prime if not debug else 100
    size_check = size_check if not debug else 100
    rep = rep if not debug else 2
    model_obj = model_dict[run](d_obs=d_obs, marginal=marginal, size_marginal=size_marginal)

    # Get the correct functions
    msnh_sampling_func = model_obj.sample_msnh_algo5
    grid_param = model_obj.grid
    gen_obs_func = model_obj.sample_sim
    gen_sample_func = model_obj.generate_sample
    t0_grid = model_obj.pred_grid
    tp_func = model_obj.compute_exact_prob
    t0_val = model_obj.true_param
    # or_loss_sample_func = model_obj.create_samples_for_or_loss

    # Creating sample to check entropy about
    np.random.seed(seed)
    sample_check = gen_sample_func(sample_size=size_check, marginal=marginal)
    theta_vec = sample_check[:, :model_obj.d]
    x_vec = sample_check[:, (model_obj.d + 1):]
    bern_vec = sample_check[:, model_obj.d]

    true_prob_vec = tp_func(theta_vec=theta_vec, x_vec=x_vec)
    entropy_est = -np.average([np.log(true_prob_vec[kk]) if el == 1
                               else np.log(1 - true_prob_vec[kk])
                               for kk, el in enumerate(bern_vec)])

    # Creating sample to calculate the OR loss
    # first_term_sample, second_term_sample = or_loss_sample_func(or_loss_samples=or_loss_samples)

    # Loop over repetitions and classifiers
    # Each time we train the different classifiers, we build the intervals and we record
    # whether the point is in or not.
    out_val = []
    out_cols = ['d_obs', 'b_prime', 'b', 'classifier', 'classifier_cde', 'run', 'rep', 'sample_size_obs',
                'cross_entropy_loss', 't0_true_val', 'theta_0_current', 'on_true_t0',
                'estimated_tau', 'estimated_cutoff', 'in_confint', 'out_confint', 'size_CI', 'true_entropy',
                'or_loss']
    pbar = tqdm(total=rep, desc='Toy Example for Simulations, n=%s, b=%s' % (sample_size_obs, b))
    for jj in range(rep):

        # Generates samples for each t0 values, so to be able to check both coverage and power
        x_obs = gen_obs_func(sample_size=sample_size_obs, true_param=t0_val)

        # Train the classifier for the odds
        clf_odds_fitted = {}
        clf_cde_fitted = {}
        for clf_name, clf_model in sorted(classifier_dict.items(), key=lambda x: x[0]):
            clf_odds = train_clf(sample_size=b, clf_model=clf_model, gen_function=gen_sample_func,
                                 clf_name=clf_name, marginal=marginal, nn_square_root=True)
            if verbose:
                print('----- %s Trained' % clf_name)
            tau_obs = np.array([
                compute_statistics_single_t0(
                    clf=clf_odds, obs_sample=x_obs, t0=theta_0, grid_param_t1=grid_param,
                    d=model_obj.d, d_obs=model_obj.d_obs) for theta_0 in t0_grid])

            # Calculating cross-entropy
            est_prob_vec = clf_prob_value(clf=clf_odds, x_vec=x_vec, theta_vec=theta_vec, d=model_obj.d,
                                          d_obs=model_obj.d_obs)
            loss_value = log_loss(y_true=bern_vec, y_pred=est_prob_vec)

            # Calculating or loss
            or_loss_value = 0
            # or_loss_value = or_loss(clf=clf_odds, first_sample=first_term_sample, second_sample=second_term_sample)
            clf_odds_fitted[clf_name] = (tau_obs, loss_value, or_loss_value)

            # Train the quantile regression algorithm for confidence levels
            theta_mat, sample_mat = msnh_sampling_func(b_prime=b_prime, sample_size=sample_size_obs)
            stats_mat = np.array([compute_statistics_single_t0(
                clf=clf_odds, d=model_obj.d, d_obs=model_obj.d_obs, grid_param_t1=grid_param,
                t0=theta_0, obs_sample=sample_mat[kk, :, :]) for kk, theta_0 in enumerate(theta_mat)])

            clf_cde_fitted[clf_name] = {}
            # for clf_name_qr, clf_params in sorted(classifier_cde_dict.items(), key=lambda x: x[0]):
            clf_name_qr = classifier_cde
            clf_params = classifier_cde_dict[classifier_cde]
            t0_pred_vec = train_qr_algo(model_obj=model_obj, theta_mat=theta_mat, stats_mat=stats_mat,
                                        algo_name=clf_params[0], learner_kwargs=clf_params[1],
                                        pytorch_kwargs=clf_params[2] if len(clf_params) > 2 else None,
                                        alpha=alpha, prediction_grid=t0_grid)
            clf_cde_fitted[clf_name][clf_name_qr] = t0_pred_vec

        # At this point all it's left is to record
        for clf_name, (tau_obs_val, cross_ent_loss, or_loss_value) in clf_odds_fitted.items():
            for clf_name_qr, cutoff_val in clf_cde_fitted[clf_name].items():
                size_temp = np.sum((tau_obs_val >= cutoff_val).astype(int))/t0_grid.shape[0]
                for kk, theta_0_current in enumerate(t0_grid):
                    out_val.append([
                        d_obs, b_prime, b, clf_name, clf_name_qr, run, jj, sample_size_obs, cross_ent_loss,
                        t0_val, theta_0_current, int(t0_val == theta_0_current),
                        tau_obs_val[kk], cutoff_val[kk], int(tau_obs_val[kk] > cutoff_val[kk]),
                        int(tau_obs_val[kk] <= cutoff_val[kk]), size_temp, entropy_est, or_loss_value
                    ])
        pbar.update(1)

    # Saving the results
    out_df = pd.DataFrame.from_records(data=out_val, index=range(len(out_val)), columns=out_cols)
    out_dir = 'sims/classifier_power_multid/'
    out_filename = 'classifier_power_multid%s_%sB_%sBprime_%s_%srep_alpha%s_sampleobs%s_t0val%s_%s_%s.csv' % (
        d_obs, b, b_prime, run, rep, str(alpha).replace('.', '-'), sample_size_obs,
        str(t0_val).replace('.', '-'), classifier_cde,
        datetime.strftime(datetime.today(), '%Y-%m-%d')
    )
    out_df.to_csv(out_dir + out_filename)

    # Print results
    cov_df = out_df[out_df['on_true_t0'] == 1][['classifier', 'classifier_cde',
                                                'in_confint', 'cross_entropy_loss', 'size_CI']]
    print(cov_df.groupby(['classifier', 'classifier_cde']).agg({'in_confint': [np.average],
                                                                'size_CI': [np.average, np.std],
                                                                'cross_entropy_loss': [np.average, np.std]}))

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
    out_dir = 'images/classifier_power_multid/'
    outfile_name = 'power_multid%s_classifier_reps_%sB_%sBprime_%s_%srep_alpha%s_sampleobs%s_t0val%s_%s_%s.pdf' % (
        d_obs, b, b_prime, run, rep, str(alpha).replace('.', '-'), sample_size_obs,
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
    parser.add_argument('--alpha', action="store", type=float, default=0.1,
                        help='Statistical confidence level')
    parser.add_argument('--run', action="store", type=str, default='gmm',
                        help='Problem to run')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If true, a very small value for the sample sizes is fit to make sure the'
                             'file can run quickly for debugging purposes')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='If true, logs are printed to the terminal')
    parser.add_argument('--sample_size_obs', action="store", type=int, default=10,
                        help='Sample size of the actual observed data.')
    parser.add_argument('--class_cde', action="store", type=str, default='xgb_d3_n100',
                        help='Classifier for quantile regression')
    parser.add_argument('--size_marginal', action="store", type=int, default=1000,
                        help='Sample size of the actual marginal distribution, if marginal is True.')
    parser.add_argument('--or_loss_samples', action="store", type=int, default=1000,
                        help='Sample size for the calculation of the OR loss.')
    argument_parsed = parser.parse_args()

    b_vec = [100, 500, 1000]
    for b_val in b_vec:
        main(
            d_obs=argument_parsed.d_obs,
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
            or_loss_samples=argument_parsed.or_loss_samples
        )
