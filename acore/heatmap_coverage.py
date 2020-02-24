from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import seaborn as sns

from utils.functions import compute_statistics_single_t0, train_clf
from or_classifiers.complete_list import classifier_dict, classifier_conv_dict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm
from datetime import datetime
from models.sen_poisson import SenPoissonLoader
from models.camelus_wl import CamelusSimLoader


model_dict = {
    'camelus': CamelusSimLoader,
    'poisson': SenPoissonLoader
}


def main(b, b_prime, alpha, classifier, sample_size_obs, run, rep,
         debug=False, seed=7, verbose=False, size_reference=1000):

    # Setup the variables, also to account for debug runs
    np.random.seed(seed)
    b = b if not debug else 100
    b_prime = b_prime if not debug else 100
    sample_size_obs = sample_size_obs if not debug else 1
    rep = rep if not debug else 1

    # Create the loader object, which drives most
    print('----- Loading Simulations In')
    model_obj = model_dict[run]()

    # Also, calculate the reference distribution
    model_obj.set_reference_g(size_reference=size_reference)

    # Get the correct functions
    msnh_sampling_func = model_obj.sample_msnh_algo5
    grid_param = model_obj.grid
    clf_model = classifier_dict[classifier]
    gen_sample_func = model_obj.generate_sample
    t0_grid = model_obj.grid
    gen_obs_func = model_obj.sample_sim
    classifier = classifier.replace('\n', '').replace(' ', '-')

    # Start the loop
    out_val = []
    out_cols = ['b_prime', 'b', 'classifier', 'classifier_cde', 'run', 'rep', 'sample_size_obs',
                't0_true_ax0', 't0_true_ax1', 'theta_0_current_ax0', 'theta_0_current_ax1', 'on_true_theta',
                'estimated_tau', 'estimated_cutoff', 'in_confint', 'out_confint', 'size_CI']
    pbar = tqdm(total=rep, desc='Toy Example for Simulations, n=%s' % sample_size_obs)
    for jj in range(rep):

        # Calculate Odds
        if verbose:
            print('----- Calculating Odds')
        # Compute Odds via classifier
        clf = train_clf(sample_size=b, clf_model=clf_model, gen_function=gen_sample_func, d=model_obj.d,
                        clf_name=classifier)
        if verbose:
            print('----- %s Trained' % classifier)

        # Train Quantile Regression
        if verbose:
            print('----- Training Quantile Regression Algorithm')
        theta_mat, sample_mat = msnh_sampling_func(b_prime=b_prime, sample_size=sample_size_obs)

        # Compute the tau values for QR training
        stats_mat = np.array([compute_statistics_single_t0(
            clf=clf, d=model_obj.d, d_obs=model_obj.d_obs, grid_param_t1=grid_param,
            t0=theta_0, obs_sample=sample_mat[kk, :, :]) for kk, theta_0 in enumerate(theta_mat)])

        # Fit the QR model
        model = GradientBoostingRegressor(loss='quantile', alpha=alpha, **{'max_depth': 5, 'n_estimators': 1000})
        model.fit(theta_mat.reshape(-1, 2), stats_mat.reshape(-1, ))
        t0_pred_vec = model.predict(t0_grid.reshape(-1, 2))
        if verbose:
            print('----- Quantile Regression Algorithm Trained')
            pbar2 = tqdm(total=len(t0_grid), desc='Toy Example for Simulations, n=%s' % sample_size_obs)

        for t0_val in t0_grid:

            # Create a sample of observed data that are going to be used later
            # and compute statistics tau value for each t0
            x_obs = gen_obs_func(sample_size=sample_size_obs, true_param=t0_val)
            tau_obs = np.array([
                compute_statistics_single_t0(
                    clf=clf, obs_sample=x_obs, t0=theta_0,
                    d=model_obj.d, d_obs=model_obj.d_obs, grid_param_t1=grid_param) for theta_0 in t0_grid
            ])

            size_temp = np.sum((tau_obs > t0_pred_vec).astype(int))/tau_obs.shape[0]

            # At this point all it's left is to record
            for kk, theta_0_current in enumerate(t0_grid):
                out_val.append([
                    b_prime, b, classifier, 'XGBoost -- (d5, n1000)', run, jj, sample_size_obs,
                    t0_val[0], t0_val[1], theta_0_current[0], theta_0_current[1],
                    1 if np.sum((t0_val == theta_0_current).astype(int)) == 2 else 0,
                    tau_obs[kk], t0_pred_vec[kk], int(tau_obs[kk] > t0_pred_vec[kk]),
                    int(tau_obs[kk] <= t0_pred_vec[kk]), size_temp
                ])
            if verbose:
                pbar2.update(1)
        pbar.update(1)

    # Saving the results
    out_df = pd.DataFrame.from_records(data=out_val, index=range(len(out_val)), columns=out_cols)
    out_dir = 'sims/sen_poisson_2d/'
    out_filename = '2d_sen_poisson_heatmap_%sB_%sBprime_%s_%srep_alpha%s_sampleobs%s_std15_%s.csv' % (
        b, b_prime, run, rep, str(alpha).replace('.', '-'), sample_size_obs,
        datetime.strftime(datetime.today(), '%Y-%m-%d')
    )
    out_df.to_csv(out_dir + out_filename)

    # Generating Heatmap -- Observed Values
    plot_df = out_df[out_df['on_true_theta'] == 1][['t0_true_ax0', 't0_true_ax1', 'in_confint']]
    plot_df = plot_df.groupby(['t0_true_ax0', 't0_true_ax1']).mean().reset_index()

    plt.figure(figsize=(15, 7.5))
    plot_df_heatmap = plot_df.pivot('t0_true_ax1', 't0_true_ax0', 'in_confint')
    ax = sns.heatmap(plot_df_heatmap, cmap='RdYlGn',
                     vmax=plot_df['in_confint'].max(), vmin=plot_df['in_confint'].min())
    ax.invert_yaxis()
    plt.title("Observed Coverage Across %sD %s Param Space, B=%s, B'=%s, n=%s" % (
        model_obj.d, run.title(), b, b_prime, sample_size_obs
    ), fontsize=25)
    plt.xlabel('Background', fontsize=25)
    plt.ylabel('Signal', fontsize=25)
    plt.tight_layout()
    image_name = 'heatmap_observed_coverage_%sD_%s_b_%s_bprime_%s_n%s_%s.pdf' % (
        model_obj.d, run, b, b_prime, sample_size_obs,
        datetime.strftime(datetime.today(), '%Y-%m-%d'))
    plt.savefig('images/%s' % model_obj.out_directory + image_name)

    # Generating Heatmap -- Estimated Coverage
    print('----- Estimating Coverage')
    X_cov = out_df[out_df['on_true_theta'] == 1][['t0_true_ax0', 't0_true_ax1']].values
    y_cov = out_df[out_df['on_true_theta'] == 1]['in_confint'].values

    model = LogisticRegression(penalty='none', solver='saga', max_iter=10000)
    model.fit(X_cov, y_cov)
    pred_grid = model_obj.make_grid_over_param_space(50)
    pred_cov = model.predict_proba(pred_grid)

    plot_df_cov = pd.DataFrame.from_dict({
        't0_true_ax0': np.round(pred_grid[:, 0], 1),
        't0_true_ax1': np.round(pred_grid[:, 1], 1),
        'in_confint': pred_cov[:, 1]
    })
    plot_df_heatmap = plot_df_cov.pivot('t0_true_ax1', 't0_true_ax0', 'in_confint')

    plt.figure(figsize=(15, 7.5))
    ax = sns.heatmap(plot_df_heatmap, cmap='RdYlGn',
                     vmax=plot_df_cov['in_confint'].max(), vmin=plot_df_cov['in_confint'].min())
    ax.invert_yaxis()
    plt.title("Estimated Coverage Across %sD %s Space, B=%s, B'=%s, n=%s" % (
        model_obj.d, run.title(), b, b_prime, sample_size_obs
    ), fontsize=25)
    plt.xlabel('Background', fontsize=25)
    plt.ylabel('Signal', fontsize=25)
    plt.tight_layout()
    image_name = 'heatmap_estimated_coverage_%sD_%s_b_%s_bprime_%s_n%s_%s.pdf' % (
        model_obj.d, run, b, b_prime, sample_size_obs,
        datetime.strftime(datetime.today(), '%Y-%m-%d'))
    plt.savefig('images/%s' % model_obj.out_directory + image_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', action="store", type=int, default=7,
                        help='Random State')
    parser.add_argument('--b', action="store", type=int, default=50000,
                        help='Sample size to train the classifier for calculating odds')
    parser.add_argument('--b_prime', action="store", type=int, default=5000,
                        help='Sample size to train the quantile regression algorithm')
    parser.add_argument('--alpha', action="store", type=float, default=0.1,
                        help='Statistical confidence level')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If true, a very small value for the sample sizes is fit to make sure the'
                             'file can run quickly for debugging purposes')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='If true, logs are printed to the terminal')
    parser.add_argument('--sample_size_obs', action="store", type=int, default=100,
                        help='Sample size of the actual observed data.')
    parser.add_argument('--size_reference', action="store", type=int, default=100,
                        help='Sample size for calculating the reference distribution.')
    parser.add_argument('--run', action="store", type=str, default='poisson',
                        help='Problem to run')
    parser.add_argument('--classifier', action="store", type=str, default='qda',
                        help='Classifier to run for learning the odds'),
    parser.add_argument('--rep', action="store", type=int, default=10,
                        help='Number of Repetitions for calculating coverage')
    argument_parsed = parser.parse_args()

    for (b_val, b_prime_val) in [(10000, 10000)]:
        main(
            b=b_val,   # argument_parsed.b,
            b_prime=b_prime_val,  # argument_parsed.b_prime,
            alpha=argument_parsed.alpha,
            debug=argument_parsed.debug,
            sample_size_obs=argument_parsed.sample_size_obs,
            seed=argument_parsed.seed,
            verbose=argument_parsed.verbose,
            classifier=classifier_conv_dict[argument_parsed.classifier],
            run=argument_parsed.run,
            rep=argument_parsed.rep,
            size_reference=argument_parsed.size_reference
        )
