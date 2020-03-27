import numpy as np
import argparse
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime

from utils.functions import compute_exact_tau_distr, compute_exact_tau
from models.toy_poisson import ToyPoissonLoader
from models.toy_gmm import ToyGMMLoader
from models.toy_gamma import ToyGammaLoader

model_dict = {
    'poisson': ToyPoissonLoader,
    'gmm': ToyGMMLoader,
    'gamma': ToyGammaLoader
}


def main(run, rep, alpha, t0_val, sample_size_obs,
         debug=False, seed=7, size_check=1000, verbose=False, n_sampled_true_tau=1000):

    size_check = size_check if not debug else 100
    rep = rep if not debug else 2
    n_sampled_true_tau = 10 if not debug else 10
    model_obj = model_dict[run](marginal=False, size_marginal=1000)

    # Get the correct functions
    grid_param = model_obj.grid
    gen_obs_func = model_obj.sample_sim
    gen_sample_func = model_obj.generate_sample
    t0_grid = model_obj.pred_grid
    tp_func = model_obj.compute_exact_prob
    or_func = model_obj.compute_exact_or

    # Loop over repetitions and classifiers
    # Each time we train the different classifiers, we build the intervals and we record
    # whether the point is in or not.
    np.random.seed(seed)
    out_val = []
    out_cols = ['run', 'rep', 'classifier', 'sample_size_obs', 't0_true_val', 'theta_0_current', 'on_true_t0',
                'in_true_interval', 'size_true_int', 'true_entropy']
    pbar = tqdm(total=rep, desc='Toy Example for Simulations, n=%s' % sample_size_obs)
    for jj in range(rep):

        # Creating sample to check entropy about
        sample_check = gen_sample_func(sample_size=size_check, marginal=False)
        theta_vec = sample_check[:, :model_obj.d]
        x_vec = sample_check[:, (model_obj.d + 1):]
        bern_vec = sample_check[:, model_obj.d]

        true_prob_vec = tp_func(theta_vec=theta_vec, x_vec=x_vec)
        entropy_est = -np.average([np.log(true_prob_vec[kk]) if el == 1
                                   else np.log(1 - true_prob_vec[kk])
                                   for kk, el in enumerate(bern_vec)])

        # TRUE CONFIDENCE INTERVAL
        # print('------ Calculate true Confidence Interval')
        # Generates samples for each t0 values, so to be able to check both coverage and power
        x_obs = gen_obs_func(sample_size=sample_size_obs, true_param=t0_val)

        # # Calculate the true LRT value
        tau_obs = np.array([compute_exact_tau(
            or_func=or_func, x_obs=x_obs, t0_val=theta_0, t1_linspace=grid_param) for theta_0 in t0_grid])
        tau_distr = np.apply_along_axis(arr=t0_grid.reshape(-1, model_obj.d), axis=1,
                                        func1d=lambda t0: compute_exact_tau_distr(
                                            gen_obs_func=gen_obs_func, or_func=or_func, t0_val=t0,
                                            t1_linspace=grid_param, n_sampled=n_sampled_true_tau,
                                            sample_size_obs=sample_size_obs))
        quantile_pred_tau = np.quantile(a=tau_distr, q=alpha, axis=1)
        true_interval = (tau_obs > quantile_pred_tau).astype(int)
        true_interval_size = (np.sum(true_interval) / true_interval.shape[0])

        # At this point all it's left is to record
        for kk, theta_0_current in enumerate(t0_grid):
            out_val.append([
                run, jj, 'Exact', sample_size_obs,
                t0_val, theta_0_current, int(t0_val == theta_0_current),
                true_interval[kk], true_interval_size, entropy_est
            ])
        pbar.update(1)

    # Saving the results
    out_df = pd.DataFrame.from_records(data=out_val, index=range(len(out_val)), columns=out_cols)
    out_dir = 'sims/classifier_cov_pow_toy/'
    out_filename = 'truth_reps_cov_pow_toy_%s_%srep_alpha%s_sampleobs%s_t0val%s_%s.csv' % (
        run, rep, str(alpha).replace('.', '-'), sample_size_obs,
        str(t0_val).replace('.', '-'),
        datetime.strftime(datetime.today(), '%Y-%m-%d')
    )
    out_df.to_csv(out_dir + out_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', action="store", type=int, default=7,
                        help='Random State')
    parser.add_argument('--rep', action="store", type=int, default=10,
                        help='Number of Repetitions for calculating the Pinball loss')
    parser.add_argument('--alpha', action="store", type=float, default=0.1,
                        help='Statistical confidence level')
    parser.add_argument('--run', action="store", type=str, default='poisson',
                        help='Problem to run')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If true, a very small value for the sample sizes is fit to make sure the'
                             'file can run quickly for debugging purposes')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='If true, logs are printed to the terminal')
    parser.add_argument('--sample_size_obs', action="store", type=int, default=10,
                        help='Sample size of the actual observed data.')
    parser.add_argument('--t0_val', action="store", type=float, default=10.0,
                        help='True parameter which generates the observed dataset')
    argument_parsed = parser.parse_args()

    main(
        run=argument_parsed.run,
        rep=argument_parsed.rep,
        alpha=argument_parsed.alpha,
        debug=argument_parsed.debug,
        sample_size_obs=argument_parsed.sample_size_obs,
        t0_val=argument_parsed.t0_val,
        seed=argument_parsed.seed,
        verbose=argument_parsed.verbose
    )
