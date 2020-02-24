from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import argparse
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from skgarden import RandomForestQuantileRegressor
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from statsmodels.api import QuantReg
import lightgbm as lgb

from utils.functions import train_clf, pinball_loss, compute_statistics_single_t0
from models.toy_poisson import ToyPoissonLoader
from models.toy_gmm import ToyGMMLoader


# Classifier Dictionaries
classifier_dict = {
    'poisson': {
        'QDA': QuadraticDiscriminantAnalysis()
    },
    'gmm': {
        'MLP': MLPClassifier(alpha=0, max_iter=10000)
    }
}
classifier_cde_dict = {
    'XGBoost \n (d3, n500)': ('xgb', {'max_depth': 3, 'n_estimators': 100})
}


model_dict = {
    'poisson': ToyPoissonLoader,
    'gmm': ToyGMMLoader
}


def main(run, rep, marginal, b, b_prime, alpha, sample_size_obs, size_marginal=1000,
         debug=False, seed=7, size_check=1000, size_t0_sampled=250, verbose=False):

    # Setup variables
    b = b if not debug else 10
    b_prime = b_prime if not debug else 10
    size_check = size_check if not debug else 100
    rep = rep if not debug else 1
    model_obj = model_dict[run](marginal=marginal, size_marginal=size_marginal)

    # Get the correct functions
    msnh_sampling_func = model_obj.sample_msnh_algo5
    grid_param = model_obj.grid
    gen_obs_func = model_obj.sample_sim
    gen_sample_func = model_obj.generate_sample
    t0_val = model_obj.true_param

    np.random.seed(seed)
    t0_grid = np.random.uniform(low=model_obj.low_int, high=model_obj.high_int, size=size_t0_sampled)

    # Loop over repetitions and classifiers
    # Each time we train the different classifiers, we build the intervals and we record
    # whether the point is in or not.
    np.random.seed(seed)
    out_val = []
    out_cols = ['b_prime', 'b', 'classifier', 'classifier_cde', 'run', 'rep', 'sample_size_obs',
                'pinball_loss', 'theta_0_current',
                'estimated_tau', 'estimated_cutoff', 'in_confint', 'out_confint']
    pbar = tqdm(total=rep, desc='Toy Example for Simulations, n=%s' % sample_size_obs)
    for jj in range(rep):
        # Train the classifier for the odds
        clf_odds_fitted = {}
        clf_cde_fitted = {}
        for clf_name, clf_model in sorted(classifier_dict[run].items(), key=lambda x: x[0]):
            clf_odds = train_clf(sample_size=b, clf_model=clf_model, gen_function=gen_sample_func,
                                 clf_name=clf_name)
            if verbose:
                print('----- %s Trained' % clf_name)

            # Create a validation set for validating the pinball loss
            np.random.seed(seed)
            theta_mat_valid, sample_mat_valid = msnh_sampling_func(b_prime=size_check, sample_size=sample_size_obs)
            full_mat_valid = np.hstack((theta_mat_valid, sample_mat_valid))

            stats_mat_valid = np.apply_along_axis(arr=full_mat_valid, axis=1,
                                                  func1d=lambda row: compute_statistics_single_t0(
                                                      obs_sample=row[model_obj.d:],
                                                      t0=t0_val,
                                                      grid_param_t1=grid_param,
                                                      clf=clf_odds
                                                  ))
            X_val = theta_mat_valid
            y_val = stats_mat_valid

            # Train the quantile regression algorithm for confidence levels
            theta_mat, sample_mat = msnh_sampling_func(b_prime=b_prime, sample_size=sample_size_obs)
            full_mat = np.hstack((theta_mat, sample_mat))
            stats_mat = np.apply_along_axis(arr=full_mat, axis=1,
                                            func1d=lambda row: compute_statistics_single_t0(
                                                clf=clf_odds,
                                                obs_sample=row[model_obj.d:],
                                                t0=row[:model_obj.d],
                                                grid_param_t1=grid_param
                                            ))
            clf_cde_fitted[clf_name] = {}
            for clf_name_qr, clf_params in sorted(classifier_cde_dict.items(), key=lambda x: x[0]):
                # Train the regression quantiles algorithms
                if clf_params[0] == 'xgb':
                    model = GradientBoostingRegressor(loss='quantile', alpha=alpha, **clf_params[1])
                    model.fit(theta_mat.reshape(-1, model_obj.d), stats_mat.reshape(-1, ))
                    t0_pred_vec = model.predict(t0_grid.reshape(-1, model_obj.d))
                    val_pred_vec = model.predict(X_val)
                elif clf_params[0] == 'rf':
                    model = RandomForestQuantileRegressor(**clf_params[1])
                    model.fit(theta_mat.reshape(-1, model_obj.d), stats_mat.reshape(-1, ))
                    t0_pred_vec = model.predict(t0_grid.reshape(-1, model_obj.d), quantile=alpha)
                    val_pred_vec = model.predict(X_val, quantile=alpha * 100)
                elif clf_params[0] == 'lgb':
                    model = lgb.LGBMRegressor(objective='quantile', alpha=alpha, **clf_params[1])
                    model.fit(theta_mat.reshape(-1, model_obj.d), stats_mat.reshape(-1, ))
                    t0_pred_vec = model.predict(t0_grid.reshape(-1, model_obj.d))
                    val_pred_vec = model.predict(X_val)
                elif clf_params[0] == 'linear':
                    t0_pred_vec = QuantReg(theta_mat.reshape(-1, model_obj.d), stats_mat.reshape(-1, )).fit(
                        q=alpha).predict(t0_grid.reshape(-1, model_obj.d))
                    val_pred_vec = QuantReg(theta_mat.reshape(-1, model_obj.d), stats_mat.reshape(-1, )).fit(
                        q=alpha).predict(X_val.reshape(-1, model_obj.d))
                else:
                    raise ValueError('CDE Classifier not defined in the file.')

                loss_value = pinball_loss(y_true=val_pred_vec, y_pred=y_val, alpha=alpha)
                clf_cde_fitted[clf_name][clf_name_qr] = (t0_pred_vec, loss_value)

            # Generates samples for each t0 values
            # Then calculates tau at each t0, but using the sample generated at that t0
            # In other words, we should expect the samples to be included in the confidence intervals
            # everytime
            t0_obs_sampled = {t0: gen_obs_func(
               sample_size=sample_size_obs, true_param=t0) for t0 in t0_grid}
            tau_obs = np.array([
                compute_statistics_single_t0(
                    clf=clf_odds, obs_sample=t0_obs_sampled[theta_0],
                    t0=theta_0, grid_param_t1=grid_param) for theta_0 in t0_grid])
            clf_odds_fitted[clf_name] = tau_obs

        # At this point all it's left is to record
        for clf_name, tau_obs_val in clf_odds_fitted.items():
            for clf_name_qr, (cutoff_val, loss_value) in clf_cde_fitted[clf_name].items():
                for kk, theta_0_current in enumerate(t0_grid):
                    out_val.append([
                        b, b_prime, clf_name, clf_name_qr, run, jj, sample_size_obs, loss_value,
                        theta_0_current,
                        tau_obs_val[kk], cutoff_val[kk], int(tau_obs_val[kk] > cutoff_val[kk]),
                        int(tau_obs_val[kk] <= cutoff_val[kk])
                    ])
        pbar.update(1)

    # Saving the results
    out_df = pd.DataFrame.from_records(data=out_val, index=range(len(out_val)), columns=out_cols)
    out_dir = 'sims/classifier_coverage_toy/'
    out_filename = 'classifier_coverage_toy_%sB_%sBprime_%s_%srep_alpha%s_sampleobs%s_%s.csv' % (
        b, b_prime, run, rep, str(alpha).replace('.', '-'), sample_size_obs,
        datetime.strftime(datetime.today(), '%Y-%m-%d')
    )
    out_df.to_csv(out_dir + out_filename)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', action="store", type=int, default=7,
                        help='Random State')
    parser.add_argument('--rep', action="store", type=int, default=10,
                        help='Number of Repetitions')
    parser.add_argument('--b', action="store", type=int, default=1000,
                        help='Sample size to train the classifier for calculating odds')
    parser.add_argument('--b_prime', action="store", type=int, default=5000,
                        help='Sample size to train the quantile regression algorithm')
    parser.add_argument('--marginal', action='store_true', default=False,
                        help='Whether we are using a parametric approximation of the marginal or'
                             'the baseline reference G')
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
    parser.add_argument('--size_t0_sampled', action="store", type=int, default=250,
                        help='Sample size for the number of theta sampled.')
    parser.add_argument('--size_marginal', action="store", type=int, default=1000,
                        help='Sample size of the actual marginal distribution, if marginal is True.')
    argument_parsed = parser.parse_args()

    b_prime_vec = [100, 500, 1000]
    for b_prime_val in b_prime_vec:
        main(
            run=argument_parsed.run,
            rep=argument_parsed.rep,
            marginal=argument_parsed.marginal,
            b=argument_parsed.b,
            b_prime=b_prime_val,   # argument_parsed.b_prime,
            alpha=argument_parsed.alpha,
            debug=argument_parsed.debug,
            sample_size_obs=argument_parsed.sample_size_obs,
            seed=argument_parsed.seed,
            verbose=argument_parsed.verbose,
            size_t0_sampled=argument_parsed.size_t0_sampled,
            size_marginal=argument_parsed.size_marginal
        )
