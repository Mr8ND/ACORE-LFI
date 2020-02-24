from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import statsmodels.api as sm
import argparse
import sys
sys.path.append("..")

from tqdm.auto import tqdm
from datetime import datetime
from xgboost import XGBClassifier
from utils.functions import Suppressor, train_clf, compute_statistics_single_t0
from models.camelus_wl import CamelusSimLoader
from models.sen_poisson import SenPoissonLoader
from or_classifiers.complete_list import classifier_dict, classifier_conv_dict
from qr_algorithms.complete_list import classifier_cde_dict as classifier_cde_dict_full
from qr_algorithms.small_list import classifier_cde_dict as classifier_cde_dict_small
from utils.qr_functions import train_qr_algo

model_dict = {
    'camelus': CamelusSimLoader,
    'poisson': SenPoissonLoader
}


def main(b, alpha, classifier, sample_size_obs, run, n_eval_grid=101,
         debug=False, seed=7, sample_size_check=1000, size_reference=1000):

    # Setup the variables, also to account for debug runs
    np.random.seed(seed)
    b = b if not debug else 100
    sample_size_obs = sample_size_obs if not debug else 5
    classifier_cde_dict = classifier_cde_dict_full if not debug else classifier_cde_dict_small

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
    classifier = classifier.replace('\n', '').replace(' ', '-')

    # Then generate first the thetas used for checking coverage
    theta_vec, x_vec = model_obj.sample_sim_check(sample_size=sample_size_check, n=sample_size_obs)

    # Compute Odds via classifier
    print('----- Calculating Odds')
    clf = train_clf(sample_size=b, clf_model=clf_model, gen_function=gen_sample_func,
                    d=model_obj.d, clf_name=classifier)
    tau_obs = np.array([
        compute_statistics_single_t0(
            clf=clf, obs_sample=x_vec[kk, :, :].reshape(-1, model_obj.d_obs), d=model_obj.d, d_obs=model_obj.d_obs,
            t0=theta_0, grid_param_t1=grid_param) for kk, theta_0 in enumerate(theta_vec)])
    print('----- %s Trained' % classifier)

    # Loop over B'
    b_prime_vec = model_obj.b_prime_vec if not debug else [500, 1000]
    out_val = []
    out_cols = ['b_prime', 'classifier', 'class_cde', 'run', 'n_eval_grid', 'sample_check',
                'sample_reference', 'percent_correct_coverage', 'average_coverage',
                'percent_correct_coverage_lr', 'average_coverage_lr',
                'percent_correct_coverage_1std', 'average_coverage_1std',
                'percent_correct_coverage_2std', 'average_coverage_2std']
    for b_prime in np.array(b_prime_vec).astype(int):
        # First generate the samples to train b_prime algorithm
        np.random.seed(seed)
        theta_mat, sample_mat = msnh_sampling_func(b_prime=b_prime, sample_size=sample_size_obs)
        stats_mat = np.array([compute_statistics_single_t0(
                clf=clf, d=model_obj.d, d_obs=model_obj.d_obs, grid_param_t1=grid_param,
                t0=theta_0, obs_sample=sample_mat[kk, :, :]) for kk, theta_0 in enumerate(theta_mat)])

        pbar = tqdm(total=len(classifier_cde_dict.keys()),
                    desc=r'Working on QR classifiers, b=%s' % b_prime)
        for clf_name_qr, clf_params in sorted(classifier_cde_dict.items(), key=lambda x: x[0]):

            if b_prime > 10000 and 'RF' in clf_name_qr:
                continue

            t0_pred_vec = train_qr_algo(model_obj=model_obj, theta_mat=theta_mat, stats_mat=stats_mat,
                                        algo_name=clf_params[0], learner_kwargs=clf_params[1],
                                        pytorch_kwargs=clf_params[2] if len(clf_params) > 2 else None,
                                        alpha=alpha, prediction_grid=theta_vec)

            in_vec = np.array([
                int(tau_obs[jj] > t0_pred_vec[jj]) for jj in range(theta_vec.shape[0])
            ])

            # Calculate the mean
            model = XGBClassifier(depth=3, n_estimators=100)
            model.fit(theta_vec.reshape(-1, model_obj.d), in_vec.reshape(-1,))
            pred_grid = model_obj.pred_grid
            pred_cov_mean = model.predict_proba(pred_grid)[:, 1]
            percent_correct_coverage = np.average((pred_cov_mean > (1.0 - alpha)).astype(int))
            average_coverage = np.average(pred_cov_mean)

            # Calculate the upper limit
            x = theta_vec.reshape(-1, 2)
            y = in_vec.reshape(-1,)
            # estimate the model
            X = sm.add_constant(x)

            with Suppressor():
                model = sm.Logit(y, X).fit(full_output=False)
            proba = model.predict(X)

            percent_correct_coverage_lr = np.average((proba > (1.0 - alpha)).astype(int))
            average_coverage_lr = np.average(proba)

            # estimate confidence interval for predicted probabilities
            cov = model.cov_params()
            gradient = (proba * (1 - proba) * X.T).T  # matrix of gradients for each observation
            std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov), g)) for g in gradient])
            c = 1  # multiplier for confidence interval
            upper = np.maximum(0, np.minimum(1, proba + std_errors * c))
            percent_correct_coverage_upper = np.average((upper > (1.0 - alpha)).astype(int))
            average_coverage_upper = np.average(upper)

            upper_2std = np.maximum(0, np.minimum(1, proba + std_errors * 1.96))
            percent_correct_coverage_upper_2std = np.average((upper_2std > (1.0 - alpha)).astype(int))
            average_coverage_upper_2std = np.average(upper_2std)

            out_val.append([
                b_prime, classifier, clf_name_qr, run, n_eval_grid, sample_size_check, size_reference,
                percent_correct_coverage, average_coverage, percent_correct_coverage_lr, average_coverage_lr,
                percent_correct_coverage_upper, average_coverage_upper,
                percent_correct_coverage_upper_2std, average_coverage_upper_2std
            ])

            pbar.update(1)

    # Saving the results
    out_df = pd.DataFrame.from_records(data=out_val, index=range(len(out_val)), columns=out_cols)
    out_dir = 'sims/%s' % model_obj.out_directory
    out_filename = 'b_prime_analysis_%s_%s_alpha%s_ngrid%s_sizecheck%s_bprimemax%s_logregint_%s.csv' % (
        classifier, run, str(alpha).replace('.', '-'),
        n_eval_grid, sample_size_check, np.max(b_prime_vec),
        datetime.strftime(datetime.today(), '%Y-%m-%d')
    )
    out_df.to_csv(out_dir + out_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', action="store", type=int, default=7,
                        help='Random State')
    parser.add_argument('--b', action="store", type=int, default=50000,
                        help='Sample size to train the classifier for calculating odds')
    parser.add_argument('--alpha', action="store", type=float, default=0.1,
                        help='Statistical confidence level')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If true, a very small value for the sample sizes is fit to make sure the'
                             'file can run quickly for debugging purposes')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='If true, logs are printed to the terminal')
    parser.add_argument('--sample_size_obs', action="store", type=int, default=10,
                        help='Sample size of the actual observed data.')
    parser.add_argument('--run', action="store", type=str, default='poisson',
                        help='Problem to run')
    parser.add_argument('--classifier', action="store", type=str, default='qda',
                        help='Classifier to run for learning the odds')
    parser.add_argument('--size_reference', action="store", type=int, default=1000,
                        help='Number of samples used for the reference distribution')
    argument_parsed = parser.parse_args()

    main(
        b=argument_parsed.b,
        alpha=argument_parsed.alpha,
        debug=argument_parsed.debug,
        sample_size_obs=argument_parsed.sample_size_obs,
        seed=argument_parsed.seed,
        classifier=classifier_conv_dict[argument_parsed.classifier],
        run=argument_parsed.run,
        size_reference=argument_parsed.size_reference
    )
