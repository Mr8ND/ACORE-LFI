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
import seaborn as sns
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

from models import muon_features
from or_classifiers.complete_list import classifier_dict, classifier_conv_dict
from qr_algorithms.complete_list import classifier_cde_dict
from utils.functions import train_clf, _train_clf, compute_statistics_single_t0, \
    _compute_statistics_single_t0, choose_clf_settings_subroutine
from utils.qr_functions import train_qr_algo


class ACORE:

    def __init__(self,
                 model: muon_features.MuonFeatures,
                 b: Union[int, None],
                 b_prime: int,
                 alpha: float,
                 statistics: Union[str, Callable],  # 'bff' or 'acore' for now
                 classifier_or: Union[str, None],
                 classifier_qr: str,
                 obs_sample_size: int,
                 seed: Union[int, None] = None,  # TODO: cascade seed down to methods involving randomness
                 debug: bool = False,
                 verbose: bool = True):
        if debug:
            logging_level = logging.DEBUG
        else:
            logging_level = logging.INFO

        # just to ease debugging
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
        self.alpha = alpha
        if isinstance(statistics, Callable):
            # TODO: allow for custom-defined statistics
            raise NotImplementedError
        self.statistics = statistics
        if classifier_or is not None:
            self.classifier_or = classifier_dict[classifier_or]
            self.classifier_or_name = classifier_or.replace('\n', '').replace(' ', '-')
        else:
            self.classifier_or = None
            self.classifier_or_name = None
        if classifier_qr is not None:
            self.classifier_qr = classifier_cde_dict[classifier_qr]
        else:
            self.classifier_qr = None
        self.obs_sample_size = obs_sample_size

        # utils
        self.verbose = verbose
        self.model.verbose = verbose
        self.time = dict()

        # results
        self.classifier_or_fit = None
        self.tau_obs = None
        self.t0_pred = None
        self.conf_region = None
        self.conf_band = None

    def choose_OR_clf_settings(self,
                               classifier_names: Union[str, list], # from complete_list -> classifier_conv_dict
                               b_train: Union[int, list],
                               b_eval: Union[int, None],  # None if doing cross validation
                               target_loss: Union[str, Callable] = "cross_entropy_loss",
                               cv_folds: Union[None, int] = 5,
                               write_df_path: Union[str, None] = None,
                               save_fig_path: Union[str, None] = None,
                               processes=None):

        if processes is None:
            processes = os.cpu_count()
        if not isinstance(b_train, list):
            b_train = [b_train]
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
        classifiers = [classifier_dict[classifier_conv_dict[clf]] for clf in classifier_names]

        if cv_folds is None:
            with Pool(processes=processes) as pool:
                train_sets = pool.map(self.model.generate_sample, b_train)
            # evaluation set for cross-entropy loss
            eval_set = self.model.generate_sample(sample_size=b_eval,
                                                  data=np.hstack((self.model.obs_x,
                                                                  self.model.obs_param.reshape(-1, 1))))
            eval_x, eval_y = eval_set[:, 1:], eval_set[:, 0]

            pool_args = zip(product(zip(b_train, train_sets), zip(classifiers, classifier_names)),
                            repeat(eval_x),
                            repeat(eval_y),
                            repeat(self.model.generate_sample),
                            repeat(self.model.d),
                            repeat(target_loss))
            pool_args = [(a, b, c, d, e, f, g, h, i) for ((a,b), (c, d)), e, f, g, h, i in pool_args]
        else:
            # e.g. if 5 folds and b=50k, then total sample size needed is 62500 to loop across folds
            sample_sizes = [int(b*cv_folds/(cv_folds-1)) for b in b_train]
            with Pool(processes=processes) as pool:
                samples = pool.map(self.model.generate_sample, sample_sizes)
            kfolds_generators = [KFold(n_splits=cv_folds, shuffle=True).split(sample) for sample in samples]
            pairs_args = []
            for i, fold_gen in enumerate(kfolds_generators):
                folds_idxs = list(fold_gen)
                for train_idx, test_idx in folds_idxs:
                    assert b_train[i] == len(train_idx)
                    pairs_args.append((b_train[i],  # b_train
                                       samples[i][train_idx, :],  # train_set
                                       samples[i][test_idx, :][:, 1:],  # eval_x
                                       samples[i][test_idx, :][:, 0]))  # eval_y

            pool_args = zip(product(pairs_args, zip(classifiers, classifier_names)),
                            repeat(self.model.generate_sample),
                            repeat(self.model.d),
                            repeat(target_loss))
            # move 3rd and 4th args to respect order in choose_clf_settings_subroutine
            pool_args = [(a, b, e, f, c, d, g, h, i) for ((a, b, c, d), (e, f)), g, h, i in pool_args]

        with Pool(processes=processes) as pool:
            results_df = pd.DataFrame(pool.starmap(choose_clf_settings_subroutine, pool_args),
                                      columns=['clf_name', 'B', 'train_loss', 'eval_loss'])

        # plot
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

        if save_fig_path is not None:
            plt.savefig(save_fig_path, bbox_inches='tight')
        plt.show()

        if write_df_path is not None:
            results_df.to_csv(write_df_path, index=False)
        return results_df

    def check_estimates(self,
                        what_to_check,
                        classifier,
                        b_train,
                        b_eval,
                        p_eval_set=1,  # 1 -> all eval samples from F; 0 -> all eval samples from G
                        n_eval_samples=6,
                        figsize=(30, 15),
                        reuse_train_set=True,
                        reuse_eval_set=True,
                        reuse_sample_idxs: Union[bool, list] = False,
                        save_pickle_path=None,
                        save_fig_path=None):

        assert (what_to_check in ['likelihood', 'acore', 'bff'])
        assert n_eval_samples == 6, "make plotting more general for any n_eval_samples"
        if reuse_sample_idxs is not False:
            assert n_eval_samples == len(reuse_sample_idxs)

        if reuse_eval_set:
            eval_set = self._check_likld_eval_set
        else:
            # evaluation set to check likelihood at some observations
            eval_set = self.model.generate_sample(sample_size=b_eval,
                                                  p=p_eval_set,
                                                  data=np.hstack((self.model.obs_x,
                                                                  self.model.obs_param.reshape(-1, 1))))
            self._check_likld_eval_set = eval_set
        eval_X, eval_y = eval_set[:, 1:], eval_set[:, 0]

        if reuse_train_set:
            clf = _train_clf(sample=self._check_likld_train_set,
                             sample_size=b_train, clf_model=classifier_dict[classifier_conv_dict[classifier]],
                             gen_function=self.model.generate_sample, d=self.model.d, clf_name=classifier)
        else:
            self._check_likld_train_set, clf = _train_clf(sample=None, sample_size=b_train,
                                                          clf_model=classifier_dict[classifier_conv_dict[classifier]],
                                                          gen_function=self.model.generate_sample,
                                                          d=self.model.d, clf_name=classifier)

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

            if what_to_check == 'likelihood':
                if p_eval_set == 0:
                    p_eval_set = 1e-15
                estimates = est_prob_vec[:, 1] / p_eval_set
            elif what_to_check == 'bff':
                est_prob_vec[est_prob_vec[:, 0] == 0, 0] = 1e-15
                estimates = est_prob_vec[:, 1] / est_prob_vec[:, 0]
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

    def evaluate_coverage(self,
                          classifier,
                          b: int,
                          b_prime: Union[int, list],
                          b_double_prime: int,
                          b_eval):

        if not isinstance(b_prime, list):
            b_prime = [b_prime]
        pass

    def estimate_tau(self):

        if self.b is None:
            raise ValueError("Unspecified B")
        if self.classifier_or is None:
            raise ValueError("Unspecified Odds Ratios Classifier and Classifier Name")

        start_time = datetime.now()
        clf = train_clf(sample_size=self.b,
                        clf_model=self.classifier_or,
                        gen_function=self.model.generate_sample,
                        d=self.model.d,
                        clf_name=self.classifier_or_name)
        self.classifier_or_fit = clf

        train_time = datetime.now()
        if self.verbose:
            print('----- %s Trained' % self.classifier_or_name, flush=True)
            # TODO: not general; assumes observed_sample_size == 1
            progress_bar = tqdm(total=len(self.model.param_grid)*len(self.model.obs_x),
                                desc='Calculate Odds')

        tau_obs = []
        # TODO: should this loop over self.model.obs_x as well? Since my observed_sample_size == 1; assume YES for now
        for obs_x in self.model.obs_x:
            tau_obs_x = []
            # TODO: not general; assumes observed_sample_size == 1
            for theta_0 in self.model.param_grid:
                tau_obs_x.append(compute_statistics_single_t0(clf=clf, obs_sample=obs_x, t0=theta_0,
                                                              d=self.model.d, d_obs=self.model.observed_dims,
                                                              grid_param_t1=self.model.param_grid,
                                                              obs_sample_size=self.obs_sample_size))
                if self.verbose:
                    progress_bar.update(1)
            tau_obs.append(tau_obs_x)
        progress_bar.close()
        pred_time = datetime.now()
        self.time['or_training_time'] = (train_time - start_time).total_seconds()
        self.time['tau_prediction_time'] = (pred_time - train_time).total_seconds()
        self.tau_obs = tau_obs

    def _estimate_tau(self):

        if self.b is None:
            raise ValueError("Unspecified B")
        if self.classifier_or is None:
            raise ValueError("Unspecified Odds Ratios Classifier and Classifier Name")

        start_time = datetime.now()
        clf = train_clf(sample_size=self.b,
                        clf_model=self.classifier_or,
                        gen_function=self.model.generate_sample,
                        d=self.model.d,
                        clf_name=self.classifier_or_name)
        self.classifier_or_fit = clf

        train_time = datetime.now()
        if self.verbose:
            print('----- %s Trained' % self.classifier_or_name, flush=True)
            progress_bar = tqdm(total=len(self.model.param_grid), desc='Calculate observed statistics')

        tau_obs = []
        # TODO: not general; assumes observed_sample_size == 1
        for theta_0 in self.model.param_grid:
            tau_obs.append(list(_compute_statistics_single_t0(name=self.statistics,
                                                              clf=clf, obs_sample=self.model.obs_x, t0=theta_0,
                                                              d=self.model.d, d_obs=self.model.observed_dims,
                                                              grid_param_t1=self.model.param_grid,
                                                              obs_sample_size=self.obs_sample_size,
                                                              n_samples=self.model.obs_x.shape[0])))
            if self.verbose:
                progress_bar.update(1)
        if self.verbose:
            progress_bar.close()
        pred_time = datetime.now()
        self.time['or_training_time'] = (train_time - start_time).total_seconds()
        self.time['tau_prediction_time'] = (pred_time - train_time).total_seconds()

        # need a sequence of plausible theta_0 for each obs_x
        self.tau_obs = list(zip(*tau_obs))

    def estimate_critical_value(self):

        if self.classifier_or_fit is None:
            raise ValueError('Classifier for Odds Ratios not trained yet')

        start_time = datetime.now()
        theta_matrix, sample_matrix = self.model.sample_msnh(b_prime=self.b_prime, sample_size=self.obs_sample_size)

        # Compute the tau values for QR training
        stats_matrix = np.array([_compute_statistics_single_t0(name=self.statistics,
                                                               clf=self.classifier_or_fit, d=self.model.d,
                                                               d_obs=self.model.observed_dims,
                                                               grid_param_t1=self.model.param_grid,
                                                               t0=theta_0, obs_sample=sample_matrix[kk, :],
                                                               obs_sample_size=self.obs_sample_size)
                                 for kk, theta_0 in tqdm(enumerate(theta_matrix),
                                                         desc='Calculate statistics for critical value')])
        bprime_time = datetime.now()

        if self.verbose:
            print('----- Training Quantile Regression Algorithm', flush=True)

        t0_pred_vec = train_qr_algo(model_obj=self.model, alpha=self.alpha,
                                    theta_mat=theta_matrix, stats_mat=stats_matrix,
                                    algo_name=self.classifier_qr[0], learner_kwargs=self.classifier_qr[1],
                                    pytorch_kwargs=self.classifier_qr[2] if len(self.classifier_qr) > 2 else None,
                                    prediction_grid=self.model.param_grid)
        cutoff_time = datetime.now()
        self.t0_pred = t0_pred_vec
        self.time['bprime_time'] = (bprime_time - start_time).total_seconds()
        self.time['cutoff_time'] = (cutoff_time - bprime_time).total_seconds()

    def confidence_region(self,
                          tau_obs: Union[list, None] = None,
                          conf_band: bool = False):

        if self.verbose and not conf_band:
            print('----- Creating Confidence Region', flush=True)

        if tau_obs is None:
            if self.tau_obs is None:
                raise ValueError('Observed Tau statistics not computed yet')
            else:
                tau_obs = self.tau_obs
        if self.t0_pred is None:
            raise ValueError('Critical values not computed yet')

        simultaneous_nh_decision = []
        for jj, t0_pred in enumerate(self.t0_pred):                 # TODO: in acore.py this was <. Typo?
            simultaneous_nh_decision.append([t0_pred, tau_obs[jj], int(tau_obs[jj] > t0_pred)])

        confidence_region = [theta for jj, theta in enumerate(self.model.param_grid)
                             if simultaneous_nh_decision[jj][2]]
        if conf_band:
            if self.conf_band is None:
                self.conf_band = []
            self.conf_band.append(confidence_region)
        else:
            self.conf_region = confidence_region

    # need one confidence interval for each observed x
    def confidence_band(self):

        self._estimate_tau()
        self.estimate_critical_value()

        if self.verbose:
            print('----- Creating Confidence Band', flush=True)

        if not all((isinstance(elem, list) or isinstance(elem, tuple)) for elem in self.tau_obs):
            raise ValueError('Need a list of lists of tau_obs to compute a confidence set for each observed x')


        for tau_obs_x in self.tau_obs:
            self.confidence_region(tau_obs=tau_obs_x, conf_band=True)

    def plot_confidence_band(self, return_df=False):

        df_plot = pd.DataFrame({"obs_theta": self.model.obs_param,
                                "lower": [min(conf_region) for conf_region in self.conf_band],
                                "upper": [max(conf_region) for conf_region in self.conf_band]})

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

        """
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
                run.title(), b, b_prime, obs_sample_size,
                '' if not t_star else '\n tau_star',
                '' if not c_star else ', c_star'), fontsize=25)
    
        plt.tight_layout()
        image_name = '2d_confint_%s_b_%s_bprime_%s_%s_%s_%s_n%s%s%s_%s.pdf' % (
            run, b, b_prime, t0_val[0], t0_val[1], obs_sample_size, classifier,
            '' if not t_star else '_taustar',
            '' if not c_star else '_cstar',
            datetime.strftime(datetime.today(), '%Y-%m-%d'))
        plt.savefig('images/%s/' % model_obj.out_directory + image_name)
        """


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--sim_data', action="store", type=str,
                        help='Simulated data to use to train ACORE')
    parser.add_argument('--obs_data', action="store", type=str,
                        help='Observed data to use to evaluate ACORE')
    parser.add_argument('--which_feat', action='store', type=str,
                        help='Which features to use: ig (integrated_energy), v0v1, af (all_features)')
    parser.add_argument('--what_to_do', action='store', type=str,
                        help='power, likelihood, coverage, confidence_band')
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

    else:  # all features (or full calorimeter data later)
        simulated_data = pd.read_csv(argument_parsed.sim_data, sep=" ", header=None)
        observed_data = pd.read_csv(argument_parsed.obs_data, sep=" ", header=None)

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

    if json_args["b"] == "None":
        b = None
    else:
        b = json_args["b"]
    if json_args["b_prime"] == "None":
        b_prime = None
    else:
        b_prime = json_args["b_prime"]
    if json_args["b_eval"] == "None":
        b_eval = None
    else:
        b_eval = json_args["b_eval"]
    if json_args["classifier_or"] == "None":
        classifier_or = None
    else:
        classifier_or = json_args["classifier_or"]
    if json_args["classifier_qr"] == "None":
        classifier_qr = None
    else:
        classifier_qr = json_args["classifier_qr"]
    if json_args["cv_folds"] == "None":
        cv_folds = None
    else:
        cv_folds = json_args["cv_folds"]

    acore = ACORE(model=model,
                  b=b,
                  b_prime=b_prime,
                  alpha=json_args["alpha"],
                  statistics=json_args["statistics"],
                  classifier_or=classifier_or,
                  classifier_qr=classifier_qr,
                  obs_sample_size=json_args["obs_sample_size"],
                  debug=debug)

    if argument_parsed.what_to_do == 'power':
        acore.choose_OR_clf_settings(classifier_names=eval(json_args["choose_classifiers"]),
                                     b_train=eval(json_args["b_train"]),
                                     b_eval=b_eval,
                                     target_loss=json_args["target_loss"],
                                     cv_folds=cv_folds,
                                     write_df_path=os.path.join(argument_parsed.write_path,
                                                                f'{argument_parsed.which_feat}_power_analysis.csv'),
                                     save_fig_path=os.path.join(argument_parsed.write_path,
                                                                f'{argument_parsed.which_feat}_power_analysis.png'),
                                     processes=os.cpu_count()-1)
    elif argument_parsed.what_to_do == 'likelihood':
        pass
    elif argument_parsed.what_to_do == 'coverage':
        pass
    elif  argument_parsed.what_to_do == 'confidence_band':
        acore.confidence_band()
        filename = f'acore_{argument_parsed.which_feat}_grid{json_args["t0_grid_granularity"]}_b{b}_bp{json_args["b_prime"]}_a{json_args["alpha"]}_clfOR-{classifier_or}_clfQR-{json_args["classifier_qr"]}.pickle'
        with open(os.path.join(argument_parsed.write_path, filename), "wb") as file:
            pickle.dump(acore, file)
    else:
        raise NotImplementedError(f"{argument_parsed.what_to_do} not available")
