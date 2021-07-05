import os
import logging
from typing import Union, Callable
from itertools import product, repeat
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns

from models.base import BaseModel
from or_classifiers.complete_list import classifier_dict
from utils.functions import stat_algo_analysis_subroutine
from waldo.regressors import regressor_dict

# TODO: this main class


class FrequentistLFI:

    def __init__(self,
                 model,  # BaseModel,
                 b: Union[int, None],
                 b_prime: Union[int, None],
                 b_double_prime: Union[int, None],
                 coverage_probability: float,  # e.g. 0.9 if 90% confidence sets
                 statistics: Union[str, Callable],  # 'bff', 'acore' or 'waldo' for now
                 statistics_algorithm_estimator: Union[str, None],
                 quantile_regressor: Union[str, None],
                 obs_sample_size: int,
                 decision_rule: str,  # one of ["less_equal", "less", "greater_equal", "greater"]
                 seed: Union[int, None] = None,
                 debug: bool = False,
                 verbose: bool = True,
                 processes: int = os.cpu_count() - 1):  # leave one free to avoid freezing

        if debug:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO

        # for now, just to ease debugging
        # TODO: implement a good logging scheme
        logging.basicConfig(
            filename="./debugging.log",
            level=logging_level,
            format='%(asctime)s %(module)s %(levelname)s %(message)s'
        )

        # core
        self.model = model
        self.b = b
        self.b_prime = b_prime
        self.b_double_prime = b_double_prime
        self.coverage_probability = coverage_probability
        if isinstance(statistics, Callable):
            # TODO: allow for custom-defined statistics
            raise NotImplementedError
        else:
            self.statistics = statistics
        self.statistics_algorithm_estimator = statistics_algorithm_estimator
        self.quantile_regressor = quantile_regressor
        self.decision_rule = decision_rule
        self.obs_sample_size = obs_sample_size

        # utils
        self.verbose = verbose
        self.model.verbose = verbose  # TODO: useless if undefined for model class
        self.processes = processes
        self.seed = seed  # TODO: cascade seed down to methods involving randomness

        # results
        self.fit_statistics_estimator = None
        self.fit_quantile_regressor = None
        self.estimated_statistics = None
        self.estimated_cutoffs = None
        self.confidence_region = None
        self.confidence_band = None  # for multiple observed values

    def statistics_algorithm_analysis(self,
                                      algorithm_names: Union[str, list],
                                      b: Union[int, list],
                                      b_eval: Union[int, None],  # to evaluate loss. None if doing cross validation
                                      target_loss: Union[str, Callable],
                                      cv_folds: Union[int, None] = 5,
                                      write_df_path: Union[str, None] = None,
                                      save_fig_path: Union[str, None] = None):

        if not isinstance(b, list):
            b = [b]
        if not isinstance(algorithm_names, list):
            algorithm_names = [algorithm_names]
        if target_loss == "ce":
            loss_name = "Cross Entropy"
            target_loss = log_loss
        elif target_loss == "mse":
            loss_name = "MSE"
            target_loss = mean_squared_error
        elif isinstance(target_loss, Callable):
            # TODO: should check it takes y_true and y_pred
            target_loss = target_loss
        else:
            raise ValueError(f"{target_loss} not currently supported")

        if self.statistics == "waldo":
            algorithms_dict = regressor_dict
        else:
            algorithms_dict = classifier_dict

        # convert names to algorithm objects
        algorithms = [algorithms_dict[algo] for algo in algorithm_names]

        if cv_folds is None:
            # generate b_samples by drawing the biggest one and getting the others as subsets of it
            sorted_b = sorted(b)
            max_b_sample = self.model.generate_sample(sample_size=sorted_b[-1])
            train_sets = [max_b_sample[:b_, :] for b_ in sorted_b[:-1]] + [max_b_sample]
            assert all([sample.shape[0] == sorted_b[idx] for idx, sample in enumerate(train_sets)])

            # evaluation set for cross-entropy loss
            eval_set = self.model.generate_sample(sample_size=b_eval,
                                                  data=np.hstack((self.model.obs_x,
                                                                  self.model.obs_param.reshape(-1, 1))))
            eval_x, eval_y = eval_set[:, 1:], eval_set[:, 0]
            # prepare args for multiprocessing
            pool_args = zip(product(zip(sorted_b, train_sets), zip(algorithms, algorithm_names)),
                            repeat(eval_x),
                            repeat(eval_y),
                            repeat(target_loss))
            # unpack inner tuples
            pool_args = [(x, y, z, w, h, k, l) for ((x, y), (z, w)), h, k, l in pool_args]
        else:
            # e.g. if 5 folds and b=50k, then total sample size needed is 62500 to loop across folds
            sample_sizes = [int(b_val * cv_folds / (cv_folds - 1)) for b_val in sorted(b)]
            max_b_sample = self.model.generate_sample(sample_size=sample_sizes[-1])
            samples = [max_b_sample[:sample_size, :] for sample_size in sample_sizes[:-1]] + [max_b_sample]
            assert all([sample.shape[0] == sample_sizes[idx] for idx, sample in enumerate(samples)])

            kfolds_generators = [KFold(n_splits=cv_folds, shuffle=True).split(sample) for sample in samples]
            pairs_args = []
            for i, fold_gen in enumerate(kfolds_generators):
                folds_idxs = list(fold_gen)
                for train_idx, test_idx in folds_idxs:
                    assert sorted(b)[i] == len(train_idx)
                    pairs_args.append((sorted(b)[i],  # b_train
                                       samples[i][train_idx, :],  # train_set
                                       samples[i][test_idx, :][:, 1:],  # eval_x
                                       samples[i][test_idx, :][:, 0]))  # eval_y

            pool_args = zip(product(pairs_args, zip(algorithms, algorithm_names)),
                            repeat(target_loss))
            # move 3rd and 4th args to respect order in choose_clf_settings_subroutine
            pool_args = [(x, y, h, k, z, w, l) for ((x, y, z, w), (h, k)), l in pool_args]

        with Pool(processes=self.processes) as pool:
            results_df = pd.DataFrame(pool.starmap(stat_algo_analysis_subroutine, pool_args),
                                      columns=['algo_name', 'B', 'train_loss', 'eval_loss'])

        # plot
        # TODO: put plotting facilities in separate module and call them on demand
        if cv_folds is None:
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            sns.lineplot(data=results_df, x="B", y="train_loss", hue="algo_name", markers=True, style="B", ax=ax[0])
            ax[0].set_title(f"{loss_name} training loss")
            sns.lineplot(data=results_df, x="B", y="eval_loss", hue="algo_name", markers=True, style="B", ax=ax[1])
            ax[1].set_title(f"{loss_name} validation loss")
        else:
            # out columns: [B, algo_name, train_loss_mean, train_loss_se, eval_loss_mean, eval_loss_se]
            results_df = results_df.groupby(["B", "algo_name"]).agg([np.mean, lambda x: np.std(x) / np.sqrt(cv_folds)]) \
                .reset_index().rename(columns={"<lambda_0>": "se"})
            results_df.columns = ["_".join(col) for col in results_df.columns.to_flat_index()]
            results_df.rename(columns={"B_": "B", "algo_name_": "algo_name"}, inplace=True)

            plot_temp_df = results_df.copy()
            plot_temp_df.loc[:, "y_train_below"] = plot_temp_df.train_loss_mean - plot_temp_df.train_loss_se
            plot_temp_df.loc[:, "y_train_above"] = plot_temp_df.train_loss_mean + plot_temp_df.train_loss_se
            plot_temp_df.loc[:, "y_eval_below"] = plot_temp_df.eval_loss_mean - plot_temp_df.eval_loss_se
            plot_temp_df.loc[:, "y_eval_above"] = plot_temp_df.eval_loss_mean + plot_temp_df.eval_loss_se

            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            sns.lineplot(data=plot_temp_df,
                         x="B", y="train_loss_mean", hue="algo_name",
                         markers=True, style="algo_name", ax=ax[0])
            sns.lineplot(data=plot_temp_df,
                         x="B", y="eval_loss_mean", hue="algo_name",
                         markers=True, style="algo_name", ax=ax[1])
            for algo_name in algorithm_names:
                algo_df = plot_temp_df.loc[plot_temp_df.clf_name == algo_name, :]
                ax[0].fill_between(x=algo_df.B, y1=algo_df.y_train_below, y2=algo_df.y_train_above, alpha=0.2)
                ax[1].fill_between(x=algo_df.B, y1=algo_df.y_eval_below, y2=algo_df.y_eval_above, alpha=0.2)
            ax[0].set_ylabel(f"Cross-validated {loss_name} loss")
            ax[1].set_ylabel(f"Cross-validated {loss_name} loss")
            ax[0].set_title("Training set")
            ax[1].set_title("Validation set")
            ax[0].set_ylim([np.min(plot_temp_df.y_train_below) - 0.05,  # +- offset for clearer viz TODO: adapt offset
                            np.max([np.max(plot_temp_df.y_eval_above), np.max(plot_temp_df.y_train_above)]) + 0.05])
            ax[1].set_ylim([np.min(plot_temp_df.y_train_below) - 0.05,
                            np.max([np.max(plot_temp_df.y_eval_above), np.max(plot_temp_df.y_train_above)]) + 0.05])

        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches='tight')
        plt.show()

        if write_df_path is not None:
            results_df.to_csv(write_df_path, index=False)
        return results_df
