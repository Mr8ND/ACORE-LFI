from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import argparse

from tqdm.auto import tqdm
from sklearn.metrics import log_loss
from datetime import datetime
from or_classifiers.complete_list import classifier_dict as classifier_dict_full
from or_classifiers.small_list import classifier_dict as classifier_dict_small
from models.camelus_wl import CamelusSimLoader
from models.sen_poisson import SenPoissonLoader
from models.inferno import InfernoToyLoader
from utils.functions import clf_prob_value, train_clf


model_dict = {
    'camelus': CamelusSimLoader,
    'poisson': SenPoissonLoader,
    'inferno': InfernoToyLoader
}


def main(alpha, run, debug=False, seed=7, size_check=1000, size_reference=10000, benchmark=1, empirical_marginal=False):

    # Setup the variables, also to account for debug runs
    np.random.seed(seed)
    classifier_dict = classifier_dict_full if not debug else classifier_dict_small

    # Create the loader object, which drives most
    print('----- Loading Simulations In')
    model_obj = model_dict[run](benchmark=benchmark, empirical_marginal=empirical_marginal)

    # Also, get the mean and std of the reference distribution
    model_obj.set_reference_g(size_reference=size_reference)
    mean_instrumental = model_obj.mean_instrumental
    cov_instrumental = model_obj.cov_instrumental

    # Get the correct functions
    gen_sample_func = model_obj.generate_sample

    np.random.seed(seed)
    # Loop to check different values of B
    b_vec = model_obj.b_sample_vec if not debug else [100, 1000]
    out_val = []
    out_cols = ['b', 'classifier', 'entropy_loss', 'alpha', 'run', 'size_check', 'size_marginal']
    for b_val in np.array(b_vec).astype(np.int):

        if b_val > 100000 and model_obj.regen_flag:
            model_obj = model_dict[run]()
            gen_sample_func = model_obj.generate_sample
            model_obj.set_reference_g_no_sample(mean_instrumental=mean_instrumental, cov_instrumental=cov_instrumental)

        np.random.seed(seed)
        sample_check = gen_sample_func(sample_size=size_check, marginal=False)
        theta_vec = sample_check[:, :model_obj.d]
        x_vec = sample_check[:, (model_obj.d + 1):]
        bern_vec = sample_check[:, model_obj.d]

        pbar = tqdm(total=len(classifier_dict.keys()), desc=r'Working on classifiers, b=%s' % b_val)

        for clf_name in sorted(classifier_dict.keys()):
            if b_val > 5000 and 'Gauss' in clf_name or b_val > 50000 and (
                    'NN' in clf_name or 'Log. Regr.' in clf_name):
                continue
            if b_val == 1e6 and 'MLP' not in clf_name:
                continue

            clf_model = classifier_dict[clf_name]
            clf = train_clf(sample_size=b_val, clf_model=clf_model, gen_function=gen_sample_func,
                            d=model_obj.d, clf_name=clf_name, nn_square_root=True)

            est_prob_vec = clf_prob_value(clf=clf, x_vec=x_vec, theta_vec=theta_vec,
                                          d=model_obj.d, d_obs=model_obj.d_obs)
            loss_value = log_loss(y_true=bern_vec, y_pred=est_prob_vec)
            out_val.append([
                b_val, clf_name.replace('\n', '').replace(' ', '-'),
                loss_value, alpha, run, size_check, size_reference
            ])

            if debug:
                print('---------- %s: %s' % (clf_name.replace('\n', '').replace(' ', '-'), loss_value))

            pbar.update(1)

    # Saving the results
    out_df = pd.DataFrame.from_records(data=out_val, index=range(len(out_val)), columns=out_cols)
    out_dir = 'sims/%s' % model_obj.out_directory
    out_filename = 'b_analysis_%s_alpha%s_sizecheck%s_bmax%s_%s.csv' % (
        run, str(alpha).replace('.', '-'), size_check, np.max(b_vec),
        datetime.strftime(datetime.today(), '%Y-%m-%d')
    )
    out_df.to_csv(out_dir + out_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', action="store", type=int, default=7,
                        help='Random State')
    parser.add_argument('--size_check', action="store", type=int, default=1000,
                        help='Number of points used to check entropy')
    parser.add_argument('--size_reference', action="store", type=int, default=1000,
                        help='Number of samples used for the reference distribution')
    parser.add_argument('--alpha', action="store", type=float, default=0.1,
                        help='Statistical confidence level')
    parser.add_argument('--run', action="store", type=str, default='camelus',
                        help='Model type to be passed in')
    parser.add_argument('--benchmark', action="store", type=int, default=1,
                        help='Benchmark to use for the INFERNO class.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='If true, a very small value for the sample sizes is fit to make sure the'
                             'file can run quickly for debugging purposes')
    parser.add_argument('--empirical_marginal', action='store_true', default=False,
                        help='Whether we are sampling directly from the empirical marginal for G')
    argument_parsed = parser.parse_args()

    main(
        alpha=argument_parsed.alpha,
        debug=argument_parsed.debug,
        seed=argument_parsed.seed,
        run=argument_parsed.run,
        size_check=argument_parsed.size_check,
        size_reference=argument_parsed.size_reference,
        benchmark=argument_parsed.benchmark,
        empirical_marginal=argument_parsed.empirical_marginal
    )
