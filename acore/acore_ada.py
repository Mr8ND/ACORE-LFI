import logging
from datetime import datetime
from tqdm import tqdm
from typing import Union
import numpy as np
import pandas as pd
import warnings

from models import muon_features
from or_classifiers.complete_list import classifier_dict
from qr_algorithms.complete_list import classifier_cde_dict
from utils.functions import train_clf, compute_statistics_single_t0, _compute_statistics_single_t0
from utils.qr_functions import train_qr_algo
import matplotlib.pyplot as plt
import seaborn as sns


class ACORE:

    def __init__(self,
                 model: muon_features.MuonFeatures,
                 b: int,
                 b_prime: int,
                 alpha: float,
                 classifier_or: str,
                 classifier_qr: str,
                 obs_sample_size: int,
                 seed: Union[int, None] = None,  # TODO: cascade seed down to methods involving randomness
                 debug: bool = False,
                 verbose=True):
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
        self.classifier_or = classifier_dict[classifier_or]
        self.classifier_or_name = classifier_or.replace('\n', '').replace(' ', '-')
        self.classifier_qr = classifier_cde_dict[classifier_qr]
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

    def estimate_tau(self):

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
            progress_bar = tqdm(total=len(self.model.param_grid), desc='Calculate Odds')

        tau_obs = []
        # TODO: not general; assumes observed_sample_size == 1
        for theta_0 in self.model.param_grid:
            tau_obs.append(list(_compute_statistics_single_t0(clf=clf, obs_sample=self.model.obs_x, t0=theta_0,
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
        stats_matrix = np.array([compute_statistics_single_t0(clf=self.classifier_or_fit, d=self.model.d,
                                                              d_obs=self.model.observed_dims,
                                                              grid_param_t1=self.model.param_grid,
                                                              t0=theta_0, obs_sample=sample_matrix[kk, :],
                                                              obs_sample_size=self.obs_sample_size)
                                 for kk, theta_0 in enumerate(theta_matrix)])
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

"""
if __name__ == "__main__":

    import sys
    sys.path.append('../../ada_code/toy_test_acore/')
    from toy_gauss import generate_data

    data = generate_data(sample_size=10000, beta_a=1, beta_b=1, lower_theta=0, higher_theta=100, scale=5, split=False)

    model = muon_features.MuonFeatures(data=data,
                         t0_grid_granularity=100,
                         true_param_low=0,
                         true_param_high=100,
                         param_dims=1,
                         observed_dims=2,
                         observed_sample_fraction=0.02,
                         reference_g='marginal',
                         param_column=0,
                         debug=True)

    acore = ACORE(model=model,
                  b=5000,
                  b_prime=5000,
                  alpha=0.05,
                  classifier_or='QDA',
                  classifier_qr='xgb_d3_n100',
                  obs_sample_size=1,
                  debug=True)

    acore.confidence_band()
"""