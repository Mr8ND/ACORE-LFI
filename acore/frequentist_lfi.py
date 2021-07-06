import os
import logging
from typing import Union, Callable
from itertools import product, repeat
from multiprocessing import Pool
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import KFold
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns

from or_classifiers.complete_list import classifier_dict
from waldo.regressors import regressor_dict
from qr_algorithms.complete_list import classifier_cde_dict
from utils.functions import train_clf, stat_algo_analysis_subroutine, _compute_statistics_single_t0
from utils.qr_functions import train_qr_algo

# TODO: this main class


class FrequentistLFI:

    def __init__(self,
                 model,  # BaseModel,
                 b: Union[int, None],
                 b_prime: Union[int, None],
                 b_double_prime: Union[int, None],
                 coverage_probability: float,  # e.g. 0.9 if 90% confidence sets
                 statistics: Union[str, Callable],  # 'bff', 'acore' or 'waldo' for now
                 statistics_algorithm: Union[str, None],
                 quantile_regressor: Union[str, None],
                 obs_sample_size: int,
                 decision_rule: str,  # one of ["less_equal", "less", "greater_equal", "greater"]
                 waldo_se_estimate: Union[str, None] = None,
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
        self.b_sample = None
        self.b_prime = b_prime
        self.b_double_prime = b_double_prime
        self.coverage_probability = coverage_probability
        if isinstance(statistics, Callable):
            # TODO: allow for custom-defined statistics
            raise NotImplementedError
        else:
            self.statistics = statistics
        if statistics == "waldo":
            self.statistics_algorithms_dict = regressor_dict
        else:
            self.statistics_algorithms_dict = classifier_dict
        self.waldo_se_estimate = waldo_se_estimate
        self.statistics_algorithm = statistics_algorithm
        self.quantile_regressor = quantile_regressor
        self.decision_rule = decision_rule
        self.obs_sample_size = obs_sample_size

        # utils
        self.verbose = verbose
        self.model.verbose = verbose  # TODO: useless if undefined for model class
        self.processes = processes
        self.seed = seed  # TODO: cascade seed down to methods involving randomness

        # results
        self.fit_statistics_algorithm = None
        self.fit_quantile_regressor = None
        self.estimated_statistics = None
        self.estimated_cutoffs = None
        self.confidence_region = None
        self.confidence_band = None  # for multiple observed values

    def analyze_statistics_algorithm(self,
                                     algorithm_names: Union[str, list],
                                     b: Union[int, list],
                                     target_loss: Union[str, Callable],
                                     b_eval: Union[int, None] = None,  # to evaluate loss. None if doing cross validation
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

        # convert names to algorithm objects
        algorithms = [self.statistics_algorithms_dict[algo] for algo in algorithm_names]

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
                algo_df = plot_temp_df.loc[plot_temp_df.algo_name == algo_name, :]
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

    def check_coverage(self,
                       quantile_regressors: Union[str, list],
                       b_prime: Union[int, list, None] = None,
                       b_double_prime: Union[int, tuple, None] = None,
                       statistics_algorithm: Union[str, object, None] = None,
                       known_statistics_kwargs: Union[dict, None] = None,
                       b: Union[str, None] = None,
                       clf_estimate_coverage_prob: str = 'logistic_regression',
                       save_fig_path=None,
                       return_df=False):

        if not isinstance(quantile_regressors, list):
            quantile_regressors = [quantile_regressors]

        if known_statistics_kwargs is None:
            # check we have everything we need
            if statistics_algorithm is None:
                if self.statistics_algorithm is None:
                    raise ValueError("Please specify an OR classifier to use")
                else:
                    statistics_algorithm = self.statistics_algorithm
            if b is None:
                if self.b is None:
                    raise ValueError("Please specify B")
                else:
                    b = self.b
            if b_double_prime is None:
                if self.b_double_prime is None:
                    raise ValueError("Please specify B double prime")
                else:
                    b_double_prime = self.b_double_prime
            if not isinstance(b_prime, list):
                b_prime = [b_prime]

            if isinstance(b_double_prime, int):
                # generate observed samples for which to construct confidence sets
                observed_theta, observed_x = self.model.generate_observed_sample(n_samples=b_double_prime,
                                                                                 obs_sample_size=self.obs_sample_size)
            else:
                observed_theta, observed_x = b_double_prime

            # temporary param_grid; use np.unique just to avoid having the same param multiple
            # times in case some true params were already in the grid
            # TODO: if B'' is very large then computing the observed statistics takes too much time
            param_grid = np.unique(np.append(self.model.param_grid.reshape(-1, self.model.d),
                                             observed_theta.reshape(-1, self.model.d), axis=0),
                                   axis=0)

            # estimate observed statistics
            or_clf_fit, tau_obs = self.estimate_tau(algorithm=statistics_algorithm,
                                                    b=b, observed_x=observed_x,
                                                    parameter_grid=param_grid,
                                                    store_results=False)

            # generate b_prime samples by drawing the biggest one and getting the others as subsets of it
            sorted_b_prime = sorted(b_prime)
            max_b_prime_sample = self.model.sample_msnh(b_prime=sorted_b_prime[-1],
                                                        obs_sample_size=self.obs_sample_size)
            b_prime_samples = [(max_b_prime_sample[0][:bprime, :], max_b_prime_sample[1][:bprime, :])
                               for bprime in sorted_b_prime[:-1]] + [max_b_prime_sample]
            assert all([(theta.shape[0] == sorted_b_prime[idx]) and (x.shape[0] == sorted_b_prime[idx])
                        for idx, (theta, x) in enumerate(b_prime_samples)])

            # set unused stuff to None
            qr_statistics = [None] * len(b_prime)
        else:
            # check we have everything we need
            assert "observed_statistics" in known_statistics_kwargs
            assert "qr_statistics" in known_statistics_kwargs
            assert "b_prime_samples" in known_statistics_kwargs
            assert "sorted_b_prime" in known_statistics_kwargs
            assert "observed_theta" in known_statistics_kwargs
            assert "param_grid" in known_statistics_kwargs

            # get what we need
            tau_obs = known_statistics_kwargs["observed_statistics"]
            qr_statistics = known_statistics_kwargs["qr_statistics"]
            b_prime_samples = known_statistics_kwargs["b_prime_samples"]
            sorted_b_prime = known_statistics_kwargs["sorted_b_prime"]
            observed_theta = known_statistics_kwargs["observed_theta"]
            param_grid = known_statistics_kwargs["param_grid"]  # param grid used to estimate observed_statistics

            # set unused stuff to None
            or_clf_fit = None

        # estimate critical values
        args = [(z, x, y, h, w, k, j) for (((x, y, w), z), h, k, j) in zip(product(zip(sorted_b_prime, b_prime_samples, qr_statistics),
                                                                                   quantile_regressors),
                                                                           repeat(or_clf_fit),
                                                                           repeat(param_grid),
                                                                           repeat(False))]
        # list of numpy arrays made of alpha quantiles for each theta (one array for each combination of args)
        # TODO: make parallel?
        predicted_quantiles = [self.estimate_critical_value(*args_combination) for args_combination in tqdm(args, desc="Estimating cutoffs")]

        w = []
        # construct w vector of indicators; one vector for each combination of args
        for idx_args in tqdm(range(len(predicted_quantiles)),  # args combination level
                             desc="Checking coverage across the parameter space"):
            # matrix to check coverage in a fast way using vectors
            check_matrix = np.hstack((
                # one observed stat for each value in param grid, repeated for each different observed sample
                np.array(tau_obs).reshape(len(param_grid) * len(observed_theta), 1),
                # repeat the cutoffs for each observed sample
                np.tile(predicted_quantiles[idx_args], len(observed_theta)).reshape(
                    len(param_grid) * len(observed_theta), 1),
                # repeat the param grid for each observed sample
                np.tile(param_grid.reshape(-1, self.model.d), [len(observed_theta), 1]).reshape(
                    len(param_grid) * len(observed_theta), self.model.d),
                # repeat the same true (observed) theta within each corresponding sample
                np.repeat(observed_theta.reshape(-1, self.model.d), len(param_grid), axis=0).reshape(
                    len(param_grid) * len(observed_theta), self.model.d),
                # repeat vector of zeros for each observed sample. We will put a 1 where param_grid == true_theta if covered.
                # Then reshape into (n_samples, n_params_grid) and sum over axis 1. If we are covering we will have a 1, otherwise not.
                np.repeat(np.zeros(shape=len(observed_theta)), repeats=len(param_grid)).reshape(
                    len(param_grid) * len(observed_theta), 1)
            ))
            assert check_matrix.shape == (
            len(param_grid) * len(observed_theta), 1 + 1 + self.model.d + self.model.d + 1)

            # if in acceptance region
            if self.decision_rule == "less_equal":
                mask_acceptance_region = (check_matrix[:, 0] <= check_matrix[:, 1])
            elif self.decision_rule == "less":
                mask_acceptance_region = (check_matrix[:, 0] < check_matrix[:, 1])
            elif self.decision_rule == "greater_equal":
                mask_acceptance_region = (check_matrix[:, 0] >= check_matrix[:, 1])
            elif self.decision_rule == "greater":
                mask_acceptance_region = (check_matrix[:, 0] > check_matrix[:, 1])

            # AND if is true theta
            mask_true_theta = (np.abs(check_matrix[:, 2:2 + self.model.d] - check_matrix[:,
                                                                            2 + self.model.d:2 + self.model.d + self.model.d]) <= 1e-9).all(
                axis=1)

            # then we are covering!
            check_matrix[mask_acceptance_region & mask_true_theta, -1] = 1

            # sum over param_grid: there will be a single 1, and only if we are covering the true theta
            w_combination = check_matrix[:, -1].reshape(len(observed_theta), len(param_grid)).sum(axis=1)
            assert len(w_combination) == len(observed_theta)
            w.append(w_combination)

        # estimate conditional coverage
        dfs_plot = []
        dfs_barplot = []
        if clf_estimate_coverage_prob == "logistic_regression":
            theta = sm.add_constant(observed_theta)
            if self.model.d == 1:
                n_cols = 2
                n_rows = (len(w) + 1) // n_cols
                fig = plt.figure(figsize=(20, 4 * n_rows))
                color_map = plt.cm.get_cmap("hsv", len(w))
            for idx, w_combination in enumerate(w):
                try:
                    log_reg = sm.Logit(w_combination, theta).fit(full_output=False)
                    probabilities = log_reg.predict(theta)
                    # estimate confidence interval for predicted probabilities -> Delta method
                    cov_matrix = log_reg.cov_params()
                    gradient = (probabilities * (1 - probabilities) * theta.T).T  # matrix of gradients for each obs
                    std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov_matrix), g)) for g in gradient])
                    assert len(std_errors) == len(probabilities)
                    c = 1  # multiplier for confidence interval
                    upper = np.maximum(0, np.minimum(1, probabilities + std_errors * c))
                    lower = np.maximum(0, np.minimum(1, probabilities - std_errors * c))
                    assert len(upper) == len(lower) == len(probabilities)
                except:
                    print(f"Perfect separation occurred, skipping B'={args[idx][1]}, QR clf={args[idx][0]}")
                    continue

                if self.model.d == 1:
                    # plot
                    df_plot = pd.DataFrame({"observed_param": observed_theta.reshape(-1, ),
                                            "probabilities": probabilities,
                                            "lower": lower,
                                            "upper": upper,
                                            "args_comb": [f"B'={args[idx][1]}, QR clf = {args[idx][0]}"] * len(lower)}
                                           ).sort_values(by="observed_param")
                    dfs_plot.append(df_plot)
                    ax = plt.subplot(n_rows, n_cols, idx + 1)
                    sns.lineplot(x=df_plot.observed_param, y=df_plot.probabilities,
                                 ax=ax, color=color_map(idx),
                                 label=f"B'={args[idx][1]}, QR clf = {args[idx][0]}")
                    sns.lineplot(x=df_plot.observed_param, y=df_plot.lower, ax=ax, color=color_map(idx))
                    sns.lineplot(x=df_plot.observed_param, y=df_plot.upper, ax=ax, color=color_map(idx))
                    ax.fill_between(x=df_plot.observed_param, y1=df_plot.lower, y2=df_plot.upper,
                                    alpha=0.2, color=color_map(idx))

                    ax.axhline(y=self.coverage_probability, color='black', linestyle='--', linewidth=2)
                    ax.legend(loc='lower left', fontsize=15)
                    ax.set_xlabel(r"$\Theta$", fontsize=15)
                    ax.set_ylabel("Coverage probability", fontsize=15)
                    ax.set_ylim([np.min(df_plot.lower) - 0.1, 1])  # small offset of 0.1
                    ax.set_xlim([np.min(df_plot.observed_param), np.max(df_plot.observed_param)])

                proportion_UC = np.sum(upper < self.coverage_probability) / len(upper)
                proportion_OC = np.sum(lower > self.coverage_probability) / len(lower)
                dfs_barplot.append(
                    pd.DataFrame({"args_comb": [f"B'={args[idx][1]}, QR clf = {args[idx][0]}"] * 3,
                                  "coverage": ["Undercoverage", "Correct Coverage", "Overcoverage"],
                                  "proportion": [proportion_UC, 1 - (proportion_OC + proportion_UC), proportion_OC]})
                )
            if save_fig_path is not None:
                plt.savefig(os.path.join(save_fig_path, f"whole_parameter_space.png"), bbox_inches="tight")
            plt.show()  # show other plots

            # barplot
            df_barplot = pd.concat(dfs_barplot, ignore_index=True, axis=0)
            fig, ax = plt.subplots(1, 1, figsize=(7 * len(w), 10))
            sns.barplot(data=df_barplot, x="args_comb", y="proportion", hue="coverage", ci=None, ax=ax)
            ax.tick_params(labelsize=17)
            ax.set_xlabel("(B', Quantile Regressor) combination", fontsize=25)
            ax.set_ylabel("Proportion", fontsize=25)
            ax.set_title("Estimated coverage across parameter space", fontdict={"fontsize": 25})
            plt.legend(fontsize="large")
            if save_fig_path is not None:
                plt.savefig(os.path.join(save_fig_path, f"proportions.png"), bbox_inches="tight")
            plt.show()

            if return_df:
                return pd.concat(dfs_plot, ignore_index=True, axis=0), df_barplot
        else:
            raise NotImplementedError

    def estimate_tau(self,
                     algorithm: Union[str, object, None] = None,
                     b: Union[int, None] = None,
                     observed_x: Union[np.array, None] = None,
                     parameter_grid: Union[np.array, None] = None,
                     store_results: bool = True):

        if (not isinstance(algorithm, str)) and (algorithm is not None):
            # TODO: need to have the algo name as well !!!
            raise NotImplementedError
        else:  # if or_classifier is not a fitted object
            if algorithm is None:
                if self.statistics_algorithm is None:
                    raise ValueError("Unspecified algorithm to estimate the test statistics")
                else:
                    algorithm = self.statistics_algorithm
            if b is None:
                if self.b is None:
                    raise ValueError("Unspecified sample size B")
                else:
                    b = self.b
            b_sample = self.model.generate_sample(sample_size=b)
            self.b_sample = b_sample
            algorithm_fit = train_clf(gen_sample=b_sample,
                                      clf_model=self.statistics_algorithms_dict[algorithm],
                                      clf_name=algorithm)
            if self.verbose:
                print('----- %s Trained' % algorithm, flush=True)
        if observed_x is None:
            observed_x = self.model.obs_x
        else:
            assert observed_x.shape[1] == self.model.observed_dims
        if parameter_grid is None:
            parameter_grid = self.model.param_grid
        parameter_grid_length = len(parameter_grid)

        if self.verbose:
            progress_bar = tqdm(total=len(parameter_grid), desc='Compute observed statistics')

        estimated_statistics = []
        # TODO: not general; assumes observed_sample_size == 1
        for theta_0 in parameter_grid.reshape(-1, self.model.d):
            estimated_statistics.append(list(
                _compute_statistics_single_t0(name=self.statistics,
                                              clf_fit=algorithm_fit,
                                              obs_sample=observed_x.reshape(-1, self.model.observed_dims),
                                              t0=theta_0.reshape(-1, self.model.d),
                                              d=self.model.d, d_obs=self.model.observed_dims,
                                              grid_param_t1=parameter_grid,
                                              obs_sample_size=self.obs_sample_size,
                                              n_samples=observed_x.reshape(-1, self.model.observed_dims).shape[0],
                                              waldo_se_estimate=self.waldo_se_estimate,
                                              x_train=b_sample[:, self.model.d:],
                                              y_train=b_sample[:, :self.model.d],
                                              statistics_algorithm=self.statistics_algorithms_dict[algorithm],
                                              bootstrap_iter=1000)
            ))
            if self.verbose:
                progress_bar.update(1)
        if self.verbose:
            progress_bar.close()

        # need a sequence of tau_obs (at each plausible theta_0) for each obs_x
        estimated_statistics = list(zip(*estimated_statistics))
        assert all([len(tau_obs_x) == parameter_grid_length for tau_obs_x in estimated_statistics]), \
            f"{[len(tau_obs_x) == parameter_grid_length for tau_obs_x in estimated_statistics]}"

        if store_results:
            self.fit_statistics_algorithm = algorithm_fit
            self.estimated_statistics = estimated_statistics
        else:
            return algorithm_fit, estimated_statistics

    def estimate_critical_value(self,
                                quantile_regressor: Union[str, None] = None,
                                b_prime: Union[int, None] = None,
                                b_prime_sample: Union[tuple, None] = None,
                                fit_statistics_algorithm: Union[object, None] = None,
                                computed_qr_statistics: Union[np.array, None] = None,
                                parameter_grid: Union[np.array, None] = None,
                                store_results: bool = True):

        if quantile_regressor is None:
            if self.quantile_regressor is None:
                raise ValueError("Unspecified Quantile Regressor to estimate cutoffs")
            else:
                quantile_regressor = self.quantile_regressor
        if (b_prime is None) and (computed_qr_statistics is None):
            if self.b_prime is None:
                raise ValueError("Unspecified sample size B prime")
            else:
                b_prime = self.b_prime
        if (fit_statistics_algorithm is None) and (computed_qr_statistics is None):
            if self.fit_statistics_algorithm is None:
                raise ValueError('Algorithm to estimate test statistics not trained yet')
            else:
                fit_statistics_algorithm = self.fit_statistics_algorithm
                b_sample = self.b_sample
        else:
            raise NotImplementedError("Need to pass b_sample and algo name as well")
        if b_prime_sample is None:
            theta_matrix, sample_matrix = self.model.sample_msnh(b_prime=b_prime, obs_sample_size=self.obs_sample_size)
        else:
            theta_matrix, sample_matrix = b_prime_sample
        if parameter_grid is None:
            parameter_grid = self.model.param_grid

        if computed_qr_statistics is None:
            # Compute the tau values for QR training
            stats_matrix = []
            for kk, theta_0 in tqdm(enumerate(theta_matrix), desc='Compute statistics for critical value'):
                theta_0 = theta_0.reshape(-1, self.model.d)
                sample = sample_matrix[kk, :].reshape(-1, self.model.observed_dims)
                stats_matrix.append(np.array([_compute_statistics_single_t0(name=self.statistics,
                                                                            clf_fit=fit_statistics_algorithm,
                                                                            d=self.model.d,
                                                                            d_obs=self.model.observed_dims,
                                                                            grid_param_t1=parameter_grid,
                                                                            t0=theta_0,
                                                                            obs_sample=sample,
                                                                            obs_sample_size=self.obs_sample_size,
                                                                            n_samples=sample.shape[0],
                                                                            waldo_se_estimate=self.waldo_se_estimate,
                                                                            x_train=b_sample[:, self.model.d:],
                                                                            y_train=b_sample[:, :self.model.d],
                                                                            statistics_algorithm=self.statistics_algorithms_dict[self.statistics_algorithm],
                                                                            bootstrap_iter=1000
                                                                            )]))
            computed_qr_statistics = np.array(stats_matrix)
        else:
            assert computed_qr_statistics.shape[0] == theta_matrix.shape[0]

        if self.verbose:
            print('----- Training Quantile Regression Algorithm', flush=True)

        quantile_regressor = classifier_cde_dict[quantile_regressor]
        if self.statistics == "waldo":
            # e.g., if 90% CI, then waldo needs 95% quantile because of absolute value in test statistics
            alpha = self.coverage_probability + ((1-self.coverage_probability)/2)
        else:
            alpha = self.coverage_probability
        estimated_cutoffs = train_qr_algo(model_obj=self.model, alpha=alpha,
                                          theta_mat=theta_matrix, stats_mat=computed_qr_statistics,
                                          algo_name=quantile_regressor[0], learner_kwargs=quantile_regressor[1],
                                          pytorch_kwargs=quantile_regressor[2] if len(quantile_regressor) > 2 else None,
                                          prediction_grid=parameter_grid)
        if store_results:
            self.estimated_cutoffs = estimated_cutoffs
        else:
            return estimated_cutoffs

    def compute_confidence_region(self,
                                  estimated_statistics: Union[list, None] = None,
                                  estimated_cutoffs: Union[np.array, None] = None,
                                  confidence_band: bool = False,
                                  store_results: bool = True):

        if self.verbose and not confidence_band:
            print('----- Creating Confidence Region', flush=True)

        if estimated_statistics is None:
            if self.estimated_statistics is None:
                raise ValueError('Estimated test statistics not computed yet')
            else:
                estimated_statistics = self.estimated_statistics
        if estimated_cutoffs is None:
            if self.estimated_cutoffs is None:
                raise ValueError('Estimated cutoffs not computed yet')
            else:
                estimated_cutoffs = self.estimated_cutoffs
        if (self.obs_sample_size == 1) and (len(estimated_statistics) == 1):
            estimated_statistics = estimated_statistics[0]
        # , f"{len(estimated_statistics)}, {len(estimated_cutoffs)}, {len(self.model.param_grid)}"
        assert len(estimated_statistics) == len(estimated_cutoffs) == len(self.model.param_grid)

        confidence_region = []
        if self.decision_rule == "less_equal":
            # we have one (tau_observed, cutoff) for each possible theta
            for idx, tau in enumerate(estimated_statistics):
                if tau <= estimated_cutoffs[idx]:
                    confidence_region.append(self.model.param_grid[idx])
        elif self.decision_rule == "less":
            for idx, tau in enumerate(estimated_statistics):
                if tau < estimated_cutoffs[idx]:
                    confidence_region.append(self.model.param_grid[idx])
        elif self.decision_rule == "greater_equal":
            for idx, tau in enumerate(estimated_statistics):
                if tau >= estimated_cutoffs[idx]:
                    confidence_region.append(self.model.param_grid[idx])
        elif self.decision_rule == "greater":
            for idx, tau in enumerate(estimated_statistics):
                if tau > estimated_cutoffs[idx]:
                    confidence_region.append(self.model.param_grid[idx])
        else:
            raise NotImplementedError

        if confidence_band:
            if store_results:
                if self.confidence_band is None:
                    self.confidence_band = []
                self.confidence_band.append(confidence_region)
            else:
                return confidence_region
        else:
            if store_results:
                self.confidence_region = confidence_region
            else:
                return confidence_region
