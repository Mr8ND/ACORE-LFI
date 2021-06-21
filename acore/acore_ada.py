import logging
from datetime import datetime
from tqdm import tqdm
from typing import Union, Callable
import warnings
from itertools import product, repeat
from multiprocessing import Pool
import os
import argparse
import json
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import seaborn as sns
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import statsmodels.api as sm

from models import muon_features
from or_classifiers.complete_list import classifier_dict
from qr_algorithms.complete_list import classifier_cde_dict
from utils.functions import train_clf, compute_statistics_single_t0, _compute_statistics_single_t0, \
    choose_clf_settings_subroutine
from utils.qr_functions import train_qr_algo

# TODO: abstract some duplicated or similar code and put it in functions.py (or somewhere else)


class ACORE:

    def __init__(self,
                 model: muon_features.MuonFeatures,
                 b: Union[int, None],
                 b_prime: Union[int, None],
                 b_double_prime: Union[int, None],
                 alpha: float,
                 statistics: Union[str, Callable],  # 'bff' or 'acore' for now
                 or_classifier_name: Union[str, None],
                 qr_classifier_name: Union[str, None],
                 obs_sample_size: int,
                 seed: Union[int, None] = None,  # TODO: cascade seed down to methods involving randomness
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

        if obs_sample_size > 1:
            warnings.warn("Check code to ensure consistency for obs_sample_size > 1, especially model and functions")

        # settings
        self.model = model
        self.b = b
        self.b_prime = b_prime
        self.b_double_prime = b_double_prime
        self.alpha = alpha
        if isinstance(statistics, Callable):
            # TODO: allow for custom-defined statistics
            raise NotImplementedError
        else:
            self.statistics = statistics
        self.or_classifier_name = or_classifier_name
        self.qr_classifier_name = qr_classifier_name
        self.obs_sample_size = obs_sample_size

        # utils
        self.verbose = verbose
        self.model.verbose = verbose
        self.processes = processes

        # temporary
        self._check_estimates_train_set = None
        self._check_estimates_eval_set = None

        # results
        self.or_classifier_fit = None
        self.tau_obs = None
        self.predicted_quantiles = None
        self.confidence_region = None
        self.confidence_band = None  # for multiple observed values

    def choose_or_clf_settings(self,
                               classifier_names: Union[str, list],  # from complete_list -> classifier_conv_dict
                               b: Union[int, list],
                               b_eval: Union[int, None],  # to evaluate loss. None if doing cross validation
                               target_loss: Union[str, Callable] = "cross_entropy_loss",
                               cv_folds: Union[int, None] = 5,
                               write_df_path: Union[str, None] = None,
                               save_fig_path: Union[str, None] = None):

        # TODO: samples should be "nested" in order to not waste too much data

        if not isinstance(b, list):
            b = [b]
        if not isinstance(classifier_names, list):
            classifier_names = [classifier_names]
        if target_loss == "cross_entropy_loss":
            target_loss = log_loss
        elif isinstance(target_loss, Callable):
            # TODO: should check it takes y_true and y_pred == predict_proba
            target_loss = target_loss
        else:
            raise ValueError(f"{target_loss} not currently supported")

        # convert names to classifiers
        classifiers = [classifier_dict[clf] for clf in classifier_names]

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
            pool_args = zip(product(zip(sorted_b, train_sets), zip(classifiers, classifier_names)),
                            repeat(eval_x),
                            repeat(eval_y),
                            repeat(self.model.generate_sample),
                            repeat(self.model.d),
                            repeat(target_loss))
            # unpack inner tuples
            pool_args = [(x, y, z, w, h, k, j, i, l) for ((x, y), (z, w)), h, k, j, i, l in pool_args]
        else:
            # e.g. if 5 folds and b=50k, then total sample size needed is 62500 to loop across folds
            sample_sizes = [int(b_val*cv_folds/(cv_folds-1)) for b_val in sorted(b)]
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

            pool_args = zip(product(pairs_args, zip(classifiers, classifier_names)),
                            repeat(self.model.generate_sample),
                            repeat(self.model.d),
                            repeat(target_loss))
            # move 3rd and 4th args to respect order in choose_clf_settings_subroutine
            pool_args = [(x, y, h, k, z, w, j, i, l) for ((x, y, z, w), (h, k)), j, i, l in pool_args]

        with Pool(processes=self.processes) as pool:
            results_df = pd.DataFrame(pool.starmap(choose_clf_settings_subroutine, pool_args),
                                      columns=['clf_name', 'B', 'train_loss', 'eval_loss'])

        # plot
        # TODO: put plotting facilities in separate module and call them on demand
        if cv_folds is None:
            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            sns.lineplot(data=results_df, x="B", y="train_loss", hue="clf_name", markers=True, style="B", ax=ax[0])
            ax[0].set_title("Cross-entropy training loss")
            sns.lineplot(data=results_df, x="B", y="eval_loss", hue="clf_name", markers=True, style="B", ax=ax[1])
            ax[1].set_title("Cross-entropy validation loss")
        else:
            # out columns: [B, clf_name, train_loss_mean, train_loss_se, eval_loss_mean, eval_loss_se]
            results_df = results_df.groupby(["B", "clf_name"]).agg([np.mean, lambda x: np.std(x)/np.sqrt(cv_folds)])\
                                                              .reset_index().rename(columns={"<lambda_0>": "se"})
            results_df.columns = ["_".join(col) for col in results_df.columns.to_flat_index()]
            results_df.rename(columns={"B_": "B", "clf_name_": "clf_name"}, inplace=True)

            plot_temp_df = results_df.copy()
            plot_temp_df.loc[:, "y_train_below"] = plot_temp_df.train_loss_mean - plot_temp_df.train_loss_se
            plot_temp_df.loc[:, "y_train_above"] = plot_temp_df.train_loss_mean + plot_temp_df.train_loss_se
            plot_temp_df.loc[:, "y_eval_below"] = plot_temp_df.eval_loss_mean - plot_temp_df.eval_loss_se
            plot_temp_df.loc[:, "y_eval_above"] = plot_temp_df.eval_loss_mean + plot_temp_df.eval_loss_se

            fig, ax = plt.subplots(1, 2, figsize=(20, 10))
            sns.lineplot(data=plot_temp_df,
                         x="B", y="train_loss_mean", hue="clf_name",
                         markers=True, style="clf_name", ax=ax[0])
            sns.lineplot(data=plot_temp_df,
                         x="B", y="eval_loss_mean", hue="clf_name",
                         markers=True, style="clf_name", ax=ax[1])
            for clf in classifier_names:
                clf_df = plot_temp_df.loc[plot_temp_df.clf_name == clf, :]
                ax[0].fill_between(x=clf_df.B, y1=clf_df.y_train_below, y2=clf_df.y_train_above, alpha=0.2)
                ax[1].fill_between(x=clf_df.B, y1=clf_df.y_eval_below, y2=clf_df.y_eval_above, alpha=0.2)
            ax[0].set_title("Cross-entropy training loss")
            ax[1].set_title("Cross-entropy validation loss")
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

    def check_estimates(self,
                        what_to_check: str,  # 'log-likelihood', 'acore', 'log-odds' (bff for n=1 and G=marginal)
                        classifier_name: str,
                        b: int,
                        b_eval: int,
                        p_eval_set=1,  # 1 -> all eval samples from F; 0 -> all eval samples from G
                        n_eval_samples=6,
                        figsize=(30, 15),
                        reuse_train_set=True,
                        reuse_eval_set=True,
                        reuse_sample_idxs: Union[bool, list] = False,
                        save_pickle_path=None,
                        save_fig_path=None):

        assert (what_to_check in ['log-likelihood', 'acore', 'log-odds'])
        assert n_eval_samples == 6, "make plotting more general for any n_eval_samples"  # TODO
        if reuse_sample_idxs is not False:
            assert n_eval_samples == len(reuse_sample_idxs)

        if reuse_eval_set:
            eval_set = self._check_estimates_eval_set
        else:
            # evaluation set to check estimates at some observations
            eval_set = self.model.generate_sample(sample_size=b_eval,
                                                  p=p_eval_set,
                                                  data=np.hstack((self.model.obs_x,
                                                                  self.model.obs_param.reshape(-1, 1))))
            self._check_estimates_eval_set = eval_set
        eval_X, eval_y = eval_set[:, 1:], eval_set[:, 0]

        if reuse_train_set:
            clf = train_clf(gen_sample=self._check_estimates_train_set,
                            clf_model=classifier_dict[classifier_name],
                            clf_name=classifier_name)
        else:
            train_sample = self.model.generate_sample(sample_size=b)
            self._check_estimates_train_set, clf = train_clf(gen_sample=train_sample,
                                                             clf_model=classifier_dict[classifier_name],
                                                             clf_name=classifier_name)

        # select n_eval_samples from eval_set, fix x and make theta vary
        dfs = []
        sample_idxs = []
        for i in tqdm(range(n_eval_samples)):
            if reuse_sample_idxs is False:
                sample_idx = np.random.randint(0, eval_X.shape[0], 1)
            else:
                sample_idx = reuse_sample_idxs[i]
            sample = eval_X[sample_idx, self.model.d:].reshape(-1, self.model.observed_dims)
            sample_vary_theta = np.hstack((
                self.model.param_grid.reshape(-1, self.model.d),
                np.tile(sample, self.model.t0_grid_granularity).reshape(-1, self.model.observed_dims)
            ))
            assert sample_vary_theta.shape == (self.model.t0_grid_granularity,
                                               self.model.observed_dims + self.model.d)

            sample_idxs.append(sample_idx)
            est_prob_vec = clf.predict_proba(sample_vary_theta)

            if what_to_check == 'log-likelihood':
                if p_eval_set == 0:
                    p_eval_set = 1e-15
                # log-likelihood
                estimates = np.log(est_prob_vec[:, 1] / p_eval_set)
            elif what_to_check == 'log-odds':
                est_prob_vec[est_prob_vec[:, 0] == 0, 0] = 1e-15
                estimates = np.log(est_prob_vec[:, 1] / est_prob_vec[:, 0])
            else:  # what_to_check == 'acore':
                estimates = []
                t1_mask = np.full(self.model.t0_grid_granularity, True)
                for i, t0 in tqdm(enumerate(self.model.param_grid), desc='Computing acore statistics'):
                    t1_mask[i] = False
                    odds_t0 = np.log(est_prob_vec[i, 1]) - np.log(est_prob_vec[i, 0])
                    odds_t1 = np.log(est_prob_vec[t1_mask, 1]) - np.log(est_prob_vec[t1_mask, 0])
                    assert odds_t1.shape[0] == (self.model.t0_grid_granularity - 1)
                    estimates.append(odds_t0 / np.max(odds_t1))
                    t1_mask[i] = True
            dfs.append(pd.DataFrame({"theta": self.model.param_grid, f"{what_to_check}": estimates}))

        if save_pickle_path is not None:
            with open(os.path.join(save_pickle_path, f"./{what_to_check}_dfs.pickle"), "wb") as f:
                pickle.dump(dfs, f)
            with open(os.path.join(save_pickle_path, f"./{what_to_check}_sample_idxs.pickle"), "wb") as f:
                pickle.dump(sample_idxs, f)

        # plot
        fig, ax = plt.subplots(2, 3, figsize=figsize)
        for i in range(2):
            for j in range(3):
                sns.lineplot(data=dfs.pop(), x="theta", y=f"{what_to_check}", ax=ax[i][j])
                idx = sample_idxs.pop()
                ax[i][j].axvline(x=eval_X[idx, :self.model.d],
                                 c='red', label=f'theta = {eval_X[idx, :self.model.d]}')
                ax[i][j].legend()
                label = 'simulator F' if eval_y[idx] == 1 else 'reference G'
                ax[i][j].set_title(f'Sample from {label}')

        if save_fig_path is not None:
            plt.savefig(os.path.join(save_fig_path, f'./{what_to_check}.png'), bbox_inches='tight')
        plt.show()

    def check_coverage(self,
                       b_prime: Union[int, list],
                       qr_classifier_names: Union[str, list],
                       b_double_prime: Union[int, None] = None,
                       or_classifier: Union[str, object, None] = None,
                       b: Union[str, None] = None,
                       clf_estimate_coverage_prob: str = 'logistic_regression',
                       save_fig_path=None,
                       return_df=False):

        if or_classifier is None:
            if self.or_classifier_name is None:
                raise ValueError("Please specify an OR classifier to use")
            else:
                or_classifier = self.or_classifier_name
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
        if not isinstance(qr_classifier_names, list):
            qr_classifier_names = [qr_classifier_names]

        # generate observed sample
        observed_theta, observed_x = self.model.sample_msnh(b_double_prime, self.obs_sample_size)

        # estimate observed statistics
        or_clf_fit, tau_obs = self.estimate_tau(or_classifier=or_classifier,
                                                b=b, observed_x=observed_x,
                                                store_results=False)

        # generate b_prime samples by drawing the biggest one and getting the others as subsets of it
        sorted_b_prime = sorted(b_prime)
        max_b_prime_sample = self.model.sample_msnh(b_prime=sorted_b_prime[-1], obs_sample_size=self.obs_sample_size)
        b_prime_samples = [(max_b_prime_sample[0][:bprime, :], max_b_prime_sample[1][:bprime, :])
                           for bprime in sorted_b_prime[:-1]] + [max_b_prime_sample]
        assert all([(theta.shape[0] == sorted_b_prime[idx]) and (x.shape[0] == sorted_b_prime[idx])
                    for idx, (theta, x) in enumerate(b_prime_samples)])

        # estimate critical values
        args = [(z, x, y, w, k) for (((x, y), z), w, k) in zip(product(zip(sorted_b_prime, b_prime_samples),
                                                                       qr_classifier_names),
                                                               repeat(or_clf_fit), repeat(False))]
        # list of numpy arrays of alpha quantiles for each theta (one array for each combination of args)
        predicted_quantiles = [self.estimate_critical_value(*args_combination) for args_combination in args]

        # construct confidence *sets* for each observed x; one confidence *band* for each combination of args
        confidence_bands = [self.compute_confidence_band(tau_obs=tau_obs,
                                                         predicted_quantiles=predicted_quantiles[idx],
                                                         store_results=False)
                            for idx in tqdm(range(len(predicted_quantiles)),
                                            desc="Computing confidence bands for each QR classifier and B prime")]

        # construct W vector of indicators; one vector for each combination of args
        w = [np.array([1 if (np.min(confidence_set) <= observed_theta[idx] <= np.max(confidence_set)) else 0
                       for idx, confidence_set in enumerate(confidence_band)])
             for confidence_band in confidence_bands]
        assert all([len(w_combination) == observed_theta.shape[0] for w_combination in w])

        # estimate conditional coverage
        dfs_plot = []
        if clf_estimate_coverage_prob == "logistic_regression":
            theta = sm.add_constant(observed_theta)
            fig, ax = plt.subplots(nrows=len(w), ncols=1, figsize=(10, 4*len(w)))
            color_map = plt.cm.get_cmap("hsv", len(w))
            for idx, w_combination in enumerate(w):
                log_reg = sm.Logit(w_combination, theta).fit(full_output=False)
                probabilities = log_reg.predict(theta)
                # estimate confidence interval for predicted probabilities -> Delta method
                cov_matrix = log_reg.cov_params()
                gradient = (probabilities * (1 - probabilities) * theta.T).T  # matrix of gradients for each obs
                std_errors = np.array([np.sqrt(np.dot(np.dot(g, cov_matrix), g)) for g in gradient])
                c = 1  # multiplier for confidence interval
                upper = np.maximum(0, np.minimum(1, probabilities + std_errors * c))
                lower = np.maximum(0, np.minimum(1, probabilities - std_errors * c))

                # plot
                df_plot = pd.DataFrame({"observed_param": observed_theta.reshape(-1,),
                                        "probabilities": probabilities,
                                        "lower": lower,
                                        "upper": upper,
                                        "args_comb": [f"B'={args[idx][1]}, QR clf = {args[idx][0]}"]*len(lower)}
                                       ).sort_values(by="observed_param")
                dfs_plot.append(df_plot)
                sns.lineplot(x=df_plot.observed_param, y=df_plot.probabilities,
                             ax=ax[idx], color=color_map(idx),
                             label=f"B'={args[idx][1]}, QR clf = {args[idx][0]}")
                sns.lineplot(x=df_plot.observed_param, y=df_plot.lower, ax=ax[idx], color=color_map(idx))
                sns.lineplot(x=df_plot.observed_param, y=df_plot.upper, ax=ax[idx], color=color_map(idx))
                ax[idx].fill_between(x=df_plot.observed_param, y1=df_plot.lower, y2=df_plot.upper,
                                     alpha=0.2, color=color_map(idx))

                ax[idx].axhline(y=1-self.alpha, color='black', linestyle='--', linewidth=2)
                ax[idx].legend(loc='lower left', fontsize=15)
                ax[idx].set_ylim([np.min(df_plot.lower) - 0.1, 1])  # small offset of 0.1
                ax[idx].set_xlim([np.min(df_plot.observed_param), np.max(df_plot.observed_param)])

            if save_fig_path is not None:
                plt.savefig(save_fig_path, bbox_inches="tight")
            if return_df:
                return pd.concat(dfs_plot, ignore_index=True, axis=0)
            plt.show()
        else:
            raise NotImplementedError

    def estimate_tau(self,
                     or_classifier: Union[str, object, None] = None,
                     b: Union[int, None] = None,
                     observed_x: Union[np.array, None] = None,
                     store_results: bool = True):

        if isinstance(or_classifier, object):
            clf = or_classifier
        else:
            if or_classifier is None:
                if self.or_classifier_name is None:
                    raise ValueError("Unspecified Odds Ratios Classifier")
                or_classifier = self.or_classifier_name
            if b is None:
                if self.b is None:
                    raise ValueError("Unspecified sample size B")
                b = self.b
            b_sample = self.model.generate_sample(sample_size=b)
            clf = train_clf(gen_sample=b_sample,
                            clf_model=classifier_dict[or_classifier],
                            clf_name=or_classifier)
            if self.verbose:
                print('----- %s Trained' % self.or_classifier_name, flush=True)
        if observed_x is None:
            observed_x = self.model.obs_x
        else:
            assert observed_x.shape[1] == self.model.observed_dims

        if self.verbose:
            progress_bar = tqdm(total=len(self.model.param_grid), desc='Calculate observed statistics')

        tau_obs = []
        # TODO: not general; assumes observed_sample_size == 1
        for theta_0 in self.model.param_grid:
            tau_obs.append(list(_compute_statistics_single_t0(name=self.statistics,
                                                              clf_fit=clf, obs_sample=observed_x, t0=theta_0,
                                                              d=self.model.d, d_obs=self.model.observed_dims,
                                                              grid_param_t1=self.model.param_grid,
                                                              obs_sample_size=self.obs_sample_size,
                                                              n_samples=observed_x.shape[0])))
            if self.verbose:
                progress_bar.update(1)
        if self.verbose:
            progress_bar.close()


        # need a sequence of tau_obs (at each plausible theta_0) for each obs_x
        tau_obs = list(zip(*tau_obs))
        assert all([len(tau_obs_x) == self.model.t0_grid_granularity for tau_obs_x in tau_obs])

        if store_results:
            self.or_classifier_fit = clf
            self.tau_obs = tau_obs
        else:
            return clf, tau_obs

    def estimate_critical_value(self,
                                qr_classifier_name: Union[str, None] = None,
                                b_prime: Union[int, None] = None,
                                b_prime_sample: Union[tuple, None] = None,
                                or_classifier_fit: Union[object, None] = None,
                                store_results: bool = True):

        if qr_classifier_name is None:
            if self.qr_classifier_name is None:
                raise ValueError("Unspecified Odds Ratios Classifier and Classifier Name")
            else:
                qr_classifier_name = self.qr_classifier_name
        if b_prime is None:
            if self.b_prime is None:
                raise ValueError("Unspecified sample size B prime")
            else:
                b_prime = self.b_prime
        if or_classifier_fit is None:
            if self.or_classifier_fit is None:
                raise ValueError('Classifier for Odds Ratios not trained yet')
            else:
                or_classifier_fit = self.or_classifier_fit
        if b_prime_sample is None:
                theta_matrix, sample_matrix = self.model.sample_msnh(b_prime=b_prime, obs_sample_size=self.obs_sample_size)
        else:
            theta_matrix, sample_matrix = b_prime_sample

        # Compute the tau values for QR training
        stats_matrix = []
        for kk, theta_0 in tqdm(enumerate(theta_matrix), desc='Calculate statistics for critical value'):
            theta_0 = theta_0.reshape(-1, self.model.d)
            sample = sample_matrix[kk, :].reshape(-1, self.model.observed_dims)
            stats_matrix.append(np.array([_compute_statistics_single_t0(name=self.statistics,
                                                                        clf_fit=or_classifier_fit, d=self.model.d,
                                                                        d_obs=self.model.observed_dims,
                                                                        grid_param_t1=self.model.param_grid,
                                                                        t0=theta_0,
                                                                        obs_sample=sample,
                                                                        obs_sample_size = self.obs_sample_size,
                                                                        n_samples = sample.shape[0])]))
        if self.verbose:
            print('----- Training Quantile Regression Algorithm', flush=True)

        qr_classifier = classifier_cde_dict[qr_classifier_name]
        predicted_quantiles = train_qr_algo(model_obj=self.model, alpha=self.alpha,
                                            theta_mat=theta_matrix, stats_mat=stats_matrix,
                                            algo_name=qr_classifier[0], learner_kwargs=qr_classifier[1],
                                            pytorch_kwargs=qr_classifier[2] if len(qr_classifier) > 2 else None,
                                            prediction_grid=self.model.param_grid)
        cutoff_time = datetime.now()
        if store_results:
            self.predicted_quantiles = predicted_quantiles
        else:
            return predicted_quantiles

    def compute_confidence_region(self,
                                  tau_obs: Union[list, None] = None,
                                  predicted_quantiles: Union[np.array, None] = None,
                                  confidence_band: bool = False,
                                  store_results: bool = True):

        if self.verbose and not confidence_band:
            print('----- Creating Confidence Region', flush=True)

        if tau_obs is None:
            if self.tau_obs is None:
                raise ValueError('Observed Tau statistics not computed yet')
            else:
                tau_obs = self.tau_obs
        if predicted_quantiles is None:
            if self.predicted_quantiles is None:
                raise ValueError('Critical values not computed yet')
            else:
                predicted_quantiles = self.predicted_quantiles
        assert len(tau_obs) == len(predicted_quantiles) == len(self.model.param_grid)

        confidence_region = []
        for idx, tau in enumerate(tau_obs):  # we have one (tau_observed, cutoff) for each possible theta
            if tau > predicted_quantiles[idx]:
                confidence_region.append(self.model.param_grid[idx])

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

    def compute_confidence_band(self,
                                tau_obs: Union[list, None] = None,  # list of lists, one for each observed x
                                predicted_quantiles: Union[np.array, None] = None,  # one value for each theta
                                store_results: bool = True):  # one confidence interval for each observed x

        if self.verbose:
            print('----- Creating Confidence Band', flush=True)

        if tau_obs is None:
            if self.tau_obs is None:
                raise ValueError('Observed Tau statistics not computed yet')
            else:
                tau_obs = self.tau_obs
        if predicted_quantiles is None:
            if self.predicted_quantiles is None:
                raise ValueError('Critical values not computed yet')
            else:
                predicted_quantiles = self.predicted_quantiles
        if not all((isinstance(elem, list) or isinstance(elem, tuple)) for elem in tau_obs):
            raise ValueError('Need a list of lists of tau_obs to compute a confidence set for each observed x')

        if store_results:
            for tau_obs_x in tau_obs:
                self.compute_confidence_region(tau_obs=tau_obs_x, predicted_quantiles=predicted_quantiles,
                                               confidence_band=True, store_results=store_results)
        else:
            confidence_band = []
            for tau_obs_x in tau_obs:
                confidence_band.append(
                    self.compute_confidence_region(tau_obs=tau_obs_x, predicted_quantiles=predicted_quantiles,
                                                   confidence_band=True, store_results=store_results)
                )
            return confidence_band

    def plot_confidence_band(self, return_df=False):

        # TODO: check this plots the expected thing

        df_plot = pd.DataFrame({"obs_theta": self.model.obs_param,  # plot empty set if conf region somehow is empty
                                "lower": [np.min(conf_region) if len(conf_region) > 0 else np.min(self.model.param_grid)
                                          for conf_region in self.confidence_band],
                                "upper": [np.max(conf_region) if len(conf_region) > 0 else np.min(self.model.param_grid)
                                          for conf_region in self.confidence_band]})

        df_plot.loc[:, "covered"] = (df_plot.obs_theta >= df_plot.lower) & (df_plot.obs_theta <= df_plot.upper)

        # error bar "center" is the starting point from where to plot upper and lower error
        # this is observed theta if covering for data point, (lower+upper)/2 otherwise
        df_plot.loc[:, "error_bar_center"] = df_plot.obs_theta
        df_plot.loc[df_plot.covered == False, "error_bar_center"] = \
            (df_plot.loc[df_plot.covered == False, "lower"] + df_plot.loc[df_plot.covered == False, "upper"]) / 2

        # compute upper and lower error wrt "center"
        df_plot.loc[:, "lower_err"] = df_plot.error_bar_center - df_plot.lower
        df_plot.loc[:, "upper_err"] = df_plot.upper - df_plot.error_bar_center

        # plot
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 13))

        # TODO: add plot of expected size and type I error dividing theta in bins

        # draw bisector to have a reference; give a little bit of offset at the corners (-+1)
        offset = 0.01*(self.model.true_param_high - self.model.true_param_low)
        sns.lineplot(x=np.array([self.model.true_param_low-offset, self.model.true_param_high+offset]),
                     y=np.array([self.model.true_param_low-offset, self.model.true_param_high+offset]),
                     ax=ax, label=r"true $\theta$", color="darkcyan")
        ax.lines[0].set_linestyle("--")

        # plot a confidence set for all parameters
        ax.fill_between(df_plot.sort_values(by="obs_theta").loc[:, "obs_theta"],
                        y1=df_plot.sort_values(by="obs_theta").loc[:, "lower"],
                        y2=df_plot.sort_values(by="obs_theta").loc[:, "upper"],
                        color='b', alpha=.1)

        # plot confidence sets as error bars for uncovered instances --> allows to highlight them
        ax.errorbar(x=df_plot.loc[df_plot.covered == False, "obs_theta"],
                    y=df_plot.loc[df_plot.covered == False, "error_bar_center"],
                    yerr=df_plot.loc[df_plot.covered == False, ["lower_err", "upper_err"]].T.to_numpy(),
                    ecolor="crimson", fmt='none', capsize=3)

        plt.legend()
        ax.set(xlabel=r'true $\theta$', ylabel='confidence set',
               xlim=(self.model.true_param_low-offset, self.model.true_param_high+offset),
               ylim=(self.model.true_param_low-offset, self.model.true_param_high+offset))
        plt.show()

        if return_df:
            return df_plot.sort_values(by="obs_theta")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--sim_data', action="store", type=str,
                        help='Simulated data to use to train ACORE')
    parser.add_argument('--obs_data', action="store", type=str,
                        help='Observed data to use to evaluate ACORE')
    parser.add_argument('--which_feat', action='store', type=str,
                        help='Which features to use: ig (integrated_energy), v0v1, af (all_features)')
    parser.add_argument('--what_to_do', action='store', type=str,
                        help='power, check_estimates, coverage, confidence_band')
    parser.add_argument('--json_args', action='store', type=str,
                        help='Json file containing all required args apart from data')
    parser.add_argument('--write_path', action='store', type=str, default="../../acore_runs/",
                        help='Path where outputs are stored if desired')

    argument_parsed = parser.parse_args()

    with open(argument_parsed.json_args) as json_file:
        json_args = json.load(json_file)
    if json_args["debug"] == "True":
        debug = True
    else:
        debug = False

    # get features df
    if argument_parsed.which_feat == "ig":
        def integrated_energy_df(data):
            integrated_energy = data.iloc[:, [0, 1]].sum(axis=1)
            output_df = data.iloc[:, [-1]].rename(columns={16: "true_energy"})
            output_df.loc[:, "integrated_energy"] = integrated_energy
            return output_df.iloc[:, [1, 0]]

        simulated_data = integrated_energy_df(pd.read_csv(argument_parsed.sim_data, sep=" ", header=None))
        observed_data = integrated_energy_df(pd.read_csv(argument_parsed.obs_data, sep=" ", header=None))

    elif argument_parsed.which_feat == "v0v1":
        simulated_data = pd.read_csv(argument_parsed.sim_data, sep=" ", header=None).loc[:, [0, 1, 16]]
        observed_data = pd.read_csv(argument_parsed.obs_data, sep=" ", header=None).loc[:, [0, 1, 16]]

    elif argument_parsed.which_feat == "af":  # all features
        simulated_data = pd.read_csv(argument_parsed.sim_data, sep=" ", header=None)
        observed_data = pd.read_csv(argument_parsed.obs_data, sep=" ", header=None)
    else:
        raise NotImplementedError(f"Not implemented for {argument_parsed.which_feat}")

    print(f"Simulated data shape: {simulated_data.shape}")
    print(f"Observed data shape: {observed_data.shape}")

    model = muon_features.MuonFeatures(simulated_data=simulated_data,
                                       observed_data=observed_data,
                                       t0_grid_granularity=json_args["t0_grid_granularity"],
                                       true_param_low=json_args["true_param_low"],
                                       true_param_high=json_args["true_param_high"],
                                       param_dims=json_args["param_dims"],
                                       observed_dims=json_args["observed_dims"],
                                       reference_g=json_args["reference_g"],
                                       param_column=json_args["param_column"],
                                       debug=debug)

    b = None if json_args["b"] == "None" else json_args["b"]
    b_prime = None if json_args["b_prime"] == "None" else json_args["b_prime"]
    b_double_prime = None if json_args["b_double_prime"] == "None" else json_args["b_double_prime"]
    b_eval = None if json_args["b_eval"] == "None" else json_args["b_eval"]
    or_classifier = None if json_args["or_classifier"] == "None" else json_args["or_classifier"]
    qr_classifier = None if json_args["qr_classifier"] == "None" else json_args["qr_classifier"]
    cv_folds = None if json_args["cv_folds"] == "None" else json_args["cv_folds"]

    acore = ACORE(model=model,
                  b=b,
                  b_prime=b_prime,
                  b_double_prime=b_double_prime,
                  alpha=json_args["alpha"],
                  statistics=json_args["statistics"],
                  or_classifier_name=or_classifier,
                  qr_classifier_name=qr_classifier,
                  obs_sample_size=json_args["obs_sample_size"],
                  processes=json_args["processes"],
                  debug=debug)

    if argument_parsed.what_to_do == 'power':
        acore.choose_or_clf_settings(classifier_names=eval(json_args["choose_classifiers"]),
                                     b=eval(json_args["b_train"]),
                                     b_eval=b_eval,
                                     target_loss=json_args["target_loss"],
                                     cv_folds=cv_folds,
                                     write_df_path=os.path.join(argument_parsed.write_path,
                                                                f'{argument_parsed.which_feat}_power_analysis.csv'),
                                     save_fig_path=os.path.join(argument_parsed.write_path,
                                                                f'{argument_parsed.which_feat}_power_analysis.png'))
    elif argument_parsed.what_to_do == 'check_estimates':
        pass
    elif argument_parsed.what_to_do == 'coverage':
        filename = f'{argument_parsed.which_feat}_check_coverage.png'
        acore.check_coverage(b_prime=eval(json_args["b_prime_train"]),
                             qr_classifier_names=eval(json_args["choose_classifiers"]),
                             save_fig_path=os.path.join(argument_parsed.write_path, filename))
    elif argument_parsed.what_to_do == 'confidence_band':
        acore.confidence_band()
        filename = f'acore_{argument_parsed.which_feat}_grid{json_args["t0_grid_granularity"]}_b{b}_bp{json_args["b_prime"]}_a{json_args["alpha"]}_clfOR-{or_classifier}_clfQR-{json_args["classifier_qr"]}.pickle'
        with open(os.path.join(argument_parsed.write_path, filename), "wb") as file:
            pickle.dump(acore, file)
    else:
        raise NotImplementedError(f"{argument_parsed.what_to_do} not available")
