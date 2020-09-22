from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import pickle
import sys
sys.path.append("..")

from tqdm.auto import tqdm
from datetime import datetime
from or_classifiers.complete_list import classifier_dict, classifier_conv_dict
from qr_algorithms.complete_list import classifier_cde_dict
from models.sen_poisson import SenPoissonLoader
from models.camelus_wl import CamelusSimLoader
from utils.qr_functions import train_qr_algo
from utils.functions import train_clf, compute_statistics_single_t0, compute_bayesfactor_single_t0


model_dict = {
    'camelus': CamelusSimLoader,
    'poisson': SenPoissonLoader
}


def main(b, b_prime, alpha, classifier, class_cde, sample_size_obs, run, t_star, c_star, test_statistic,
         debug=False, seed=7, n_sampled=500, size_reference=1000):

    # Setup the variables, also to account for debug runs
    np.random.seed(seed)
    b = b if not debug else 100
    b_prime = b_prime if not debug else 100
    sample_size_obs = sample_size_obs if not debug else 1
    n_sampled = n_sampled if not debug else 10

    # Create the loader object, which drives most
    print('----- Loading Simulations In')
    model_obj = model_dict[run]() if not debug else model_dict[run](num_grid=21)
    t0_val = model_obj.true_t0

    # Also, calculate the reference distribution
    model_obj.set_reference_g(size_reference=size_reference)

    # Get the correct functions
    msnh_sampling_func = model_obj.sample_msnh_algo5
    grid_param = model_obj.grid
    gen_param_fun = model_obj.sample_param_values
    clf_model = classifier_dict[classifier]
    gen_sample_func = model_obj.generate_sample
    t0_grid = model_obj.grid
    gen_obs_func = model_obj.sample_sim
    classifier = classifier.replace('\n', '').replace(' ', '-')

    # Create a sample of observed data that are going to be used later
    # and compute statistics tau value for each t0
    x_obs = gen_obs_func(sample_size=sample_size_obs, true_param=t0_val)

    start_time = datetime.now()
    # Calculate Odds
    print('----- Calculating Odds')
    if t_star:
        train_time = datetime.now()
        pbar = tqdm(total=t0_grid.shape[0], desc=r'Calculating True $\tau$')
        tau_obs = []
        for t0 in t0_grid:
            tau_obs.append(model_obj.compute_exact_tau(x_obs=x_obs, t0_val=t0, meshgrid=grid_param))
            pbar.update(1)
        tau_obs = np.array(tau_obs)
        pred_time = datetime.now()
    else:
        # Compute Odds via classifier
        clf = train_clf(sample_size=b, clf_model=clf_model, gen_function=gen_sample_func, d=model_obj.d,
                        clf_name=classifier)
        train_time = datetime.now()
        print('----- %s Trained' % classifier)

        pbar = tqdm(total=len(t0_grid), desc='Calculate Odds')
        tau_obs = []
        if test_statistic == 'acore':
            for theta_0 in t0_grid:
                tau_obs.append(compute_statistics_single_t0(
                    clf=clf, obs_sample=x_obs, t0=theta_0, d=model_obj.d, d_obs=model_obj.d_obs,
                    grid_param_t1=grid_param))
                pbar.update(1)
        elif test_statistic == 'avgacore':
            for theta_0 in t0_grid:
                tau_obs.append(compute_bayesfactor_single_t0(
                                clf=clf, obs_sample=x_obs, t0=theta_0,
                                gen_param_fun=gen_param_fun, d=model_obj.d, d_obs=model_obj.d_obs))
                pbar.update(1)
        elif test_statistic == 'logavgacore':
            for theta_0 in t0_grid:
                tau_obs.append(compute_bayesfactor_single_t0(
                    clf=clf, obs_sample=x_obs, t0=theta_0, log_out=True,
                    gen_param_fun=gen_param_fun, d=model_obj.d, d_obs=model_obj.d_obs))
                pbar.update(1)
        else:
            raise ValueError('The variable test_statistic needs to be either acore, avgacore, logavgacore.'
                             ' Currently %s' % test_statistic)

        tau_obs = np.array(tau_obs)
        pred_time = datetime.now()

    # Train Quantile Regression
    if c_star:
        pbar = tqdm(total=t0_grid.shape[0], desc=r'Calculating Distribution True $\tau$')
        tau_distr = []
        for t0 in t0_grid:
            tau_distr.append(model_obj.compute_exact_tau_distr(
                t0_val=t0,
                meshgrid=grid_param,
                n_sampled=n_sampled,
                sample_size_obs=sample_size_obs))
            pbar.update(1)
        bprime_time = datetime.now()

        tau_distr = np.array(tau_distr)
        np.save(file='sims/%stau_distr_t0_%s_%s_%ssampled_%ssamplesizeobs.npy' % (
            model_obj.out_directory, b, b_prime, n_sampled, sample_size_obs
        ), arr=tau_distr)
        t0_pred_vec = np.quantile(a=tau_distr, q=alpha, axis=1)
        cutoff_time = datetime.now()

    else:
        print('----- Training Quantile Regression Algorithm')
        theta_mat, sample_mat = msnh_sampling_func(b_prime=b_prime, sample_size=sample_size_obs)

        # Compute the tau values for QR training
        if t_star:
            stats_mat = np.array([model_obj.compute_exact_tau(
                    x_obs=sample_mat[kk, :, :], t0_val=theta_0,
                    meshgrid=grid_param) for kk, theta_0 in enumerate(theta_mat)])
        else:
            if test_statistic == 'acore':
                stats_mat = np.array([compute_statistics_single_t0(
                    clf=clf, d=model_obj.d, d_obs=model_obj.d_obs, grid_param_t1=grid_param,
                    t0=theta_0, obs_sample=sample_mat[kk, :, :]) for kk, theta_0 in enumerate(theta_mat)])
            elif test_statistic == 'avgacore':
                stats_mat = np.array([compute_bayesfactor_single_t0(
                    clf=clf, obs_sample=sample_mat[kk, :, :], t0=theta_0, gen_param_fun=gen_param_fun,
                    d=model_obj.d, d_obs=model_obj.d_obs) for kk, theta_0 in enumerate(theta_mat)])
            elif test_statistic == 'logavgacore':
                stats_mat = np.array([compute_bayesfactor_single_t0(
                    clf=clf, obs_sample=sample_mat[kk, :, :], t0=theta_0, gen_param_fun=gen_param_fun, log_out=True,
                    d=model_obj.d, d_obs=model_obj.d_obs) for kk, theta_0 in enumerate(theta_mat)])
            else:
                raise ValueError('The variable test_statistic needs to be either acore, avgacore, logavgacore.'
                                 ' Currently %s' % test_statistic)
        bprime_time = datetime.now()
        clf_params = classifier_cde_dict[class_cde]

        t0_pred_vec = train_qr_algo(model_obj=model_obj, alpha=alpha, theta_mat=theta_mat, stats_mat=stats_mat,
                                    algo_name=clf_params[0], learner_kwargs=clf_params[1],
                                    pytorch_kwargs=clf_params[2] if len(clf_params) > 2 else None,
                                    prediction_grid=t0_grid)
        cutoff_time = datetime.now()

    # Confidence Region
    print('----- Creating Confidence Region')
    simultaneous_nh_decision = []
    for jj, t0_pred in enumerate(t0_pred_vec):
        simultaneous_nh_decision.append([t0_pred, tau_obs[jj], int(tau_obs[jj] < t0_pred)])

    time_vec = [(train_time - start_time).total_seconds(),
                (pred_time - train_time).total_seconds(),
                (bprime_time - pred_time).total_seconds(),
                (cutoff_time - bprime_time).total_seconds()]
    time_vec.append(sum(time_vec))
    print(time_vec)

    # Saving data
    print('----- Saving Data')
    save_dict = {
        'background': t0_grid[:, 0],
        'signal': t0_grid[:, 1],
        'tau_statistics': tau_obs,
        'simul_nh_cutoff': [el[0] for el in simultaneous_nh_decision],
        'simul_nh_decision': [el[2] for el in simultaneous_nh_decision],
        'b': b,
        'b_prime': b_prime,
        'seed': seed,
        'sample_size_obs': sample_size_obs,
        'classifier': classifier,
        't_star': t_star,
        'time_vec': time_vec
    }
    outfile_name = '2d_confint_%s_data_b_%s_bprime_%s_%s_%s_n%s_%s_%s_%s_%s%s_%s_%s.pkl' % (
        run, b, b_prime, t0_val[0], t0_val[1], sample_size_obs, classifier, class_cde, n_sampled,
        '' if not t_star else '_taustar',
        '' if not c_star else '_cstar',
        test_statistic,
        datetime.strftime(datetime.today(), '%Y-%m-%d')
    )
    outdir = 'sims/%s' % model_obj.out_directory
    pickle.dump(obj=save_dict, file=open(outdir + outfile_name, 'wb'))

    # Visualization
    plot_df = pd.DataFrame.from_dict({
        'background': t0_grid[:, 0],
        'signal': t0_grid[:, 1],
        'tau_statistics': tau_obs,
        'simul_nh_cutoff': [el[0] for el in simultaneous_nh_decision],
        'simul_nh_decision': [el[2] for el in simultaneous_nh_decision]
    })

    col_vec = ['blue']
    alpha_vec = [0.75, 0.1]
    theta_0_plot = plot_df['background'].values
    theta_1_plot = plot_df['signal'].values

    plt.figure(figsize=(12, 8))
    for ii, col in enumerate(['simul_nh_decision']):
        value_temp = plot_df[col].values
        marker = np.array(["x" if el else "o" for el in value_temp])
        unique_markers = set(marker)

        for j, um in enumerate(unique_markers):
            mask = marker == um
            plt.scatter(x=theta_0_plot[mask], y=theta_1_plot[mask],
                        marker=um, color=col_vec[ii], alpha=alpha_vec[j])

        plt.scatter(x=t0_val[0], y=t0_val[1], color='r', marker='*', s=500)
        plt.xlabel('Background', fontsize=25)
        plt.ylabel('Signal', fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title("2D Confidence Interval, %s Example, B=%s, B'=%s, n=%s%s%s" % (
            run.title(), b, b_prime, sample_size_obs,
            '' if not t_star else '\n tau_star',
            '' if not c_star else ', c_star'), fontsize=25)

    plt.tight_layout()
    image_name = '2d_confint_%s_b_%s_bprime_%s_%s_%s_%s_n%s%s%s_%s_%s.pdf' % (
        run, b, b_prime, t0_val[0], t0_val[1], sample_size_obs, classifier,
        '' if not t_star else '_taustar',
        '' if not c_star else '_cstar',
        test_statistic,
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
    parser.add_argument('--t_star', action='store_true', default=False,
                        help='If true, the odds are computed exactly.')
    parser.add_argument('--c_star', action='store_true', default=False,
                        help='If true, the QR rejection is computed via MC.')
    parser.add_argument('--sample_size_obs', action="store", type=int, default=10,
                        help='Sample size of the actual observed data.')
    parser.add_argument('--run', action="store", type=str, default='poisson',
                        help='Problem to run')
    parser.add_argument('--classifier', action="store", type=str, default='qda',
                        help='Classifier to run for learning the odds')
    parser.add_argument('--class_cde', action="store", type=str, default='xgb_d3_n100',
                        help='Classifier to run for QR')
    parser.add_argument('--n_sampled', action="store", type=int, default=500,
                        help='Number of values to sample for c_star calculation.')
    parser.add_argument('--size_reference', action="store", type=int, default=1000,
                        help='Number of samples used for the reference distribution')
    parser.add_argument('--test_statistic', action="store", type=str, default='acore',
                        help='Test statistic to compute confidence intervals. Can be acore|avgacore|logavgacore')
    argument_parsed = parser.parse_args()

    main(
        b=argument_parsed.b,
        b_prime=argument_parsed.b_prime,
        alpha=argument_parsed.alpha,
        debug=argument_parsed.debug,
        sample_size_obs=argument_parsed.sample_size_obs,
        seed=argument_parsed.seed,
        classifier=classifier_conv_dict[argument_parsed.classifier],
        run=argument_parsed.run,
        t_star=argument_parsed.t_star,
        c_star=argument_parsed.c_star,
        class_cde=argument_parsed.class_cde,
        n_sampled=argument_parsed.n_sampled,
        size_reference=argument_parsed.size_reference,
        test_statistic=argument_parsed.test_statistic
    )
