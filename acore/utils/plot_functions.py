import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime


def plot_probability_fit(results_df, run, size_obs, hue_vec, hue_col, facet_list, facet_col, out_dir='images/'):

    cols_plot = 3
    rows_plot = len(facet_list) // cols_plot
    rows_plot += len(facet_list) % cols_plot
    color_vec = sns.color_palette("cubehelix", len(hue_vec))

    fig = plt.figure(figsize=(30, 20))
    for j, facet in enumerate(facet_list):

        temp_df = results_df[results_df[facet_col] == facet]

        ax = fig.add_subplot(rows_plot, cols_plot, j + 1)
        for ii, hue in enumerate(hue_vec):
            sns.regplot(x='est_prob', y='true_prob', label=hue,
                        data=temp_df[temp_df[hue_col] == hue],
                        color=color_vec[ii],
                        lowess=True, scatter=False)
        plt.scatter(x=np.arange(0, 1, 0.01), y=np.arange(0, 1, 0.01), linestyle=':', color='black', alpha=0.1)
        plt.ylabel('True Probability', fontsize=20)
        plt.xlabel('Estimated Probability', fontsize=20)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title('%s: %s' % (facet_col, facet), fontsize=20)
        plt.legend(loc='best')

    outfile_name = 'probability_fit_%sobs_%s_%s_hue_%s_facet_%s.pdf' % (
        size_obs, run, hue_col, facet_col,
        datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.tight_layout()
    plt.savefig(out_dir+outfile_name)
    plt.close()


def plot_loss_classifiers(results_df, run, size_obs, class_col, x_col, out_dir='images/'):

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    results_df[x_col + '_log'] = np.log(results_df[x_col].values)
    sns.scatterplot(x=x_col + '_log', y='loss', hue=class_col, data=results_df, palette='cubehelix', s=200)
    sns.lineplot(x=x_col + '_log', y='loss', hue=class_col, data=results_df, palette='cubehelix')
    plt.ylabel('Cross-Entropy Loss (Logarithm)', fontsize=25)
    plt.xlabel('Sample Size', fontsize=25)
    plt.legend(loc='best', fontsize=25)
    plt.title('Loss Function Value vs. Sample Size', fontsize=32)
    plt.xticks(results_df[x_col + '_log'].unique(), [str(x) for x in results_df[x_col].unique()])

    ax.legend(markerscale=3)

    outfile_name = 'probability_loss_function_fit_%sobs_%s_%s.pdf' % (
        size_obs, run, datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.tight_layout()
    plt.savefig(out_dir + outfile_name)
    plt.close()


def plot_loss_classifiers_specific_ss(results_df, run, size_obs, hue, class_col,
                                      loss_col, marginal, out_dir='images/', entropy=False):

    fig = plt.figure(figsize=(20, 10))
    sns.boxplot(x=class_col, y=loss_col, hue=hue, data=results_df, palette='cubehelix')
    label_y = 'Cross-Entropy Loss' if 'cross' in loss_col else 'Brier Score (Logarithm)'
    plt.ylabel('%s' % label_y, fontsize=28)
    plt.xlabel('Classifier', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    if entropy:
        plt.axhline(y=results_df['entropy'].values[0], linestyle='--', color='k', alpha=0.75,
                    label='True Distribution Entropy')
    plt.legend(loc='best', fontsize=28)
    g_label = 'Parametric Fit of Marginal' if marginal else 'Reference'
    plt.title('%s, %s Example, G: %s' % (label_y, run.title(), g_label), fontsize=32)
    outfile_name = 'probability_loss_function_fit_%sobs_%s_%s_%s_%s.pdf' % (
        size_obs, run, loss_col, 'marginalg' if marginal else 'referenceg',
        datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.tight_layout()
    plt.savefig(out_dir + outfile_name)
    plt.close()


def plot_loss_classifiers_cde(results_df, run, b, hue, class_col, loss_col, marginal, out_dir='images/'):

    fig = plt.figure(figsize=(20, 10))
    sns.boxplot(x=class_col, y=loss_col, hue=hue, data=results_df, palette='cubehelix')
    label_y = 'Pinball Loss'
    plt.ylabel('%s (Logarithm)' % label_y, fontsize=28)
    plt.xlabel('Quantile Classifier', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='best', fontsize=28)
    g_label = 'Parametric Fit of Marginal' if marginal else 'Reference'
    plt.title('%s, %s Example, G: %s, B=%s' % (label_y, run.title(), g_label, b), fontsize=32)
    outfile_name = 'quantile_pinball_loss_function_fit_bval%s_%s_%s_%s_%s.pdf' % (
        b, run, loss_col, 'marginalg' if marginal else 'referenceg',
        datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.tight_layout()
    plt.savefig(out_dir + outfile_name)
    plt.close()


def plot_loss_true_cde(results_df, run, hue, class_col, loss_col, out_dir='images/', extra_title=''):

    fig = plt.figure(figsize=(20, 10))
    sns.boxplot(x=class_col, y=loss_col, hue=hue, data=results_df, palette='cubehelix')
    label_y = 'Pinball Loss'
    plt.ylabel('%s (Logarithm)' % label_y, fontsize=28)
    plt.xlabel('Quantile Classifier', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='best', fontsize=28)
    plt.title('%s, %s Example, %s' % (label_y, run.title(), extra_title), fontsize=32)
    outfile_name = 'quantile_pinball_true_loss_function_fit_%s_%s_%s_%s.pdf' % (
        run, loss_col, extra_title.replace(' ', '_'),
        datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.tight_layout()
    plt.savefig(out_dir + outfile_name)
    plt.close()


def plot_cutoff_cde(results_df, run, x_col, y_col, hue, true_col, t0_val,
                    out_dir='images/', extra_title=''):

    b_prime_values = results_df['b_prime'].unique()
    for b_prime in b_prime_values:
        fig = plt.figure(figsize=(20, 10))
        temp_df = results_df[results_df['b_prime'] == b_prime]
        sns.boxplot(x=x_col, y=y_col, hue=hue, data=temp_df, palette='cubehelix')
        line_df = temp_df.sort_values(by=x_col)[[x_col, true_col]].groupby(x_col).mean().reset_index()
        plt.scatter(x=range(len(line_df[x_col].values)), y=line_df[true_col].values, color='blue', s=250, label='True C')
        plt.ylabel('Estimated Cutoff', fontsize=28)
        plt.xlabel(r'$\theta_0$', fontsize=28)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(loc='best', fontsize=28)
        plt.title(r'Estimated and True Cutoffs, %s Example, %s, %s, $\theta_0$=%s' % (
            run.title(), b_prime, extra_title, t0_val), fontsize=32)
        outfile_name = 'cutoff_estimates_true_fit_%s_t0val_%s_bprime%s_%s_%s.pdf' % (
            run, t0_val, b_prime.split('=')[-1].replace(' ', '_'), extra_title.replace(' ', '_'),
            datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
        )
        plt.tight_layout()
        plt.savefig(out_dir + outfile_name)
        plt.close()


def plot_diff_cov_cde(results_df, run, x_col, y_col, hue, out_dir='images/', extra_title=''):
    b_prime_values = results_df['b_prime'].unique()
    class_values = results_df[hue].unique()

    cols_plot = 3
    rows_plot = len(class_values) // cols_plot
    rows_plot += len(class_values) % cols_plot if len(class_values) > 3 else 1
    color_vec = sns.color_palette("cubehelix", len(class_values))

    for b_prime in b_prime_values:
        fig = plt.figure(figsize=(20, 8))
        for j, classifier in enumerate(class_values):
            ax = fig.add_subplot(rows_plot, cols_plot, j + 1)
            temp_df = results_df[(results_df['b_prime'] == b_prime) & (results_df[hue] == classifier)]

            plt.scatter(temp_df[x_col].values, temp_df[y_col].values, color=color_vec[j])
            extra_sign = '^\star' if 'true' in extra_title.lower() else ''
            plt.ylabel(r'$\mathbb{I}(\tau%s \leq C) - \mathbb{I}(\tau%s \leq \hat{C}_\theta)$' % (
                extra_sign, extra_sign), fontsize=20)
            plt.xlabel(r'$\Theta$', fontsize=20)
            plt.title('%s, %s, %s' % (classifier.replace('\n', ''), '\n' + b_prime, extra_title), fontsize=24)

        outfile_name = 'coverage_diff_plot_%s_%s_%s_%s.pdf' % (
            run, b_prime.split('=')[-1].replace(' ', '_'), extra_title.replace(' ', '_'),
            datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
        )
        plt.tight_layout()
        plt.savefig(out_dir + outfile_name)
        plt.close()


def plot_error_rate_cutoff_true_tau(results_df, run, x_col, y_col, hue,
                                    out_dir='images/cutoff_true_tau_analysis/', extra_title=''):

    results_df_mean = results_df[[x_col, y_col, hue, 'rep']].groupby(
        [x_col, hue, 'rep']).mean().reset_index()
    results_df_mean[x_col + 'plot'] = results_df_mean[x_col].apply(lambda x: int(x.split('=')[1]))
    fig = plt.figure(figsize=(12, 8))
    sns.lineplot(x=x_col + 'plot', y=y_col, hue=hue, data=results_df_mean)
    plt.xlabel("Training Sample Size B'", fontsize=20)
    plt.ylabel("Average Accuracy \n (Across Repetitions)", fontsize=20)
    plt.title(r'Average Accuracy in Estimating $k(\tau^{\star})$ vs. $\hat{k}(\tau^{\star})$', fontsize=24)
    plt.ylim([0, 1.05])
    plt.axhline(y=1, linestyle='--')
    plt.legend(loc='best')
    outfile_name = 'error_rate_true_tau_plot_%s_%s_%s.pdf' % (
        run, extra_title.replace(' ', '_'),
        datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.tight_layout()
    plt.savefig(out_dir + outfile_name)
    plt.close()


def plot_theta_errors_cutoff_true_tau(results_df, run, x_col, y_col, hue, across_col,
                                      out_dir='images/cutoff_true_tau_analysis/', extra_title=''):

    results_df_mean = results_df[[x_col, y_col, hue, across_col, 'rep']].groupby(
        [x_col, hue, across_col, 'rep']).mean().reset_index()
    results_df_mean[y_col] = 1 - results_df_mean[y_col].values

    across_values = results_df[across_col].unique()
    cols_plot = 3
    rows_plot = len(across_values) // cols_plot
    rows_plot += len(across_values) % cols_plot if len(across_values) > 3 else 1

    fig = plt.figure(figsize=(20, 8))
    for j, classifier in enumerate(across_values):
        ax = fig.add_subplot(rows_plot, cols_plot, j + 1)
        temp_df = results_df_mean[(results_df_mean[across_col] == classifier)]

        sns.lineplot(x=x_col, y=y_col, hue=hue, data=temp_df, color='cubehelix')
        plt.xlabel(r'$\theta_0$')
        plt.ylabel("Average Accuracy \n (Across Repetitions)")
        plt.title(r'Average Accuracy in Estimating $k(\tau^{\star})$ vs. $\hat{k}(\tau^{\star})$ %s' % (
            '\n' + classifier.replace('\n', '-')))
    plt.ylim([0, 1.05])
    plt.axhline(y=1, linestyle='--')
    outfile_name = 'theta_error_rate_plot_%s_%s_%s.pdf' % (
        run, extra_title.replace(' ', '_'),
        datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.tight_layout()
    plt.savefig(out_dir + outfile_name)
    plt.close()


def plot_error_rate_cutoff_est_tau(results_df, run, x_col, y_col, hue, t0_val, classifier,
                                    out_dir='images/classifier_tau_analysis/', extra_title=''):
    results_df_mean = results_df[[x_col, y_col, hue, 'rep']].groupby(
        [x_col, hue, 'rep']).mean().reset_index()
    results_df_mean[x_col + 'plot'] = results_df_mean[x_col].apply(lambda x: int(x.split('=')[1]))
    fig = plt.figure(figsize=(12, 8))
    sns.lineplot(x=x_col + 'plot', y=y_col, hue=hue, data=results_df_mean)
    plt.xlabel("Training Sample Size B'", fontsize=20)
    plt.ylabel("Average Accuracy \n (Across Repetitions)", fontsize=20)
    plt.title(r'Average Accuracy in Estimating $k(\tau)$ vs. $\hat{k}(\tau)$ %s $\theta_0$=%s, Classifier=%s' % (
        '\n', t0_val, classifier
    ), fontsize=24)
    plt.ylim([0.5, 1.05])
    plt.axhline(y=1, linestyle='--')
    plt.legend(loc='best')
    outfile_name = 'error_rate_classifier_%s_tau_plot_t0val_%s_%s_%s_%s.pdf' % (
        classifier.replace('\n', '-'), run, extra_title.replace(' ', '_'), t0_val,
        datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.tight_layout()
    plt.savefig(out_dir + outfile_name)
    plt.close()


def plot_theta_errors_cutoff_est_tau(results_df, run, x_col, y_col, hue, across_col, t0_val, classifier_discr,
                                     out_dir='images/classifier_tau_analysis/', extra_title=''):

    results_df_mean = results_df[[x_col, y_col, hue, across_col, 'rep']].groupby(
        [x_col, hue, across_col, 'rep']).mean().reset_index()
    results_df_mean[y_col] = 1 - results_df_mean[y_col].values

    across_values = results_df[across_col].unique()
    cols_plot = 3
    rows_plot = len(across_values) // cols_plot
    rows_plot += len(across_values) % cols_plot if len(across_values) > 3 else 1

    fig = plt.figure(figsize=(20, 8))
    for j, classifier in enumerate(across_values):
        ax = fig.add_subplot(rows_plot, cols_plot, j + 1)
        temp_df = results_df_mean[(results_df_mean[across_col] == classifier)]

        sns.lineplot(x=x_col, y=y_col, hue=hue, data=temp_df, color='cubehelix')
        plt.xlabel(r'$\theta_0$')
        plt.ylabel("Average Accuracy \n (Across Repetitions)")
        plt.title(r'Average Accuracy in Estimating $k(\tau)$ vs. $\hat{k}(\tau)$ %s $\theta_0$=%s, Classifier=%s ' % (
            '\n' + classifier.replace('\n', '-'), t0_val, classifier_discr))
    plt.ylim([0, 1.05])
    plt.axhline(y=1, linestyle='--')
    outfile_name = 'theta_classifier_%s_error_rate_plot_t0val_%s_%s_%s_%s.pdf' % (
        classifier_discr.replace('\n', '-'), t0_val, run, extra_title.replace(' ', '_'),
        datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.tight_layout()
    plt.savefig(out_dir + outfile_name)
    plt.close()


def plot_odds_fit(results_df, run, size_obs, hue_vec, hue_col, facet_list, facet_col, out_dir='images/'):

    cols_plot = 3
    rows_plot = len(facet_list) // cols_plot
    rows_plot += len(facet_list) % cols_plot
    color_vec = sns.color_palette("cubehelix", len(hue_vec))

    fig = plt.figure(figsize=(30, 20))
    for j, facet in enumerate(facet_list):

        temp_df = results_df[results_df[facet_col] == facet]

        ax = fig.add_subplot(rows_plot, cols_plot, j + 1)
        for ii, hue in enumerate(hue_vec):
            sns.regplot(x='est_odds', y='true_odds', label=hue,
                        data=temp_df[temp_df[hue_col] == hue],
                        color=color_vec[ii], lowess=True,
                        line_kws={'lw': 2}, scatter=False)
        est_odds = temp_df['est_odds'].values
        est_odds[est_odds == np.inf] = 0
        x_95 = np.quantile(est_odds[~np.isnan(est_odds)], q=.75)
        y_95 = np.quantile(temp_df['true_odds'].values, q=.75)
        plt.scatter(x=np.linspace(start=0, stop=x_95, num=100), y=np.linspace(start=0, stop=y_95, num=100),
                    linestyle=':', color='black', alpha=0.1)
        plt.ylabel('True Odds')
        plt.xlabel('Estimated Odds')
        plt.xlim([0, x_95])
        plt.ylim([0, y_95])
        plt.title('%s: %s' % (facet_col, facet))
        plt.legend(loc='best')

    outfile_name = 'odds_fit_%sobs_%s_%s_hue_%s_facet_%s.pdf' % (
        size_obs, run, hue_col, facet_col,
        datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.tight_layout()
    plt.savefig(out_dir+outfile_name)
    plt.close()


def or_over_t1(results_df, t1_linspace, or_vals, run, classifier, ss, t0_val, marginal, out_dir='images/'):

    fig, ax = plt.subplots(figsize=(20, 10))

    # Calculate minimum of results_df
    plot_df = results_df[['t1_round', 'OR', 'sample_size_str']]
    mean_df = plot_df.groupby(['t1_round', 'sample_size_str']).mean().reset_index()
    min_value_true = results_df['true_min_val'].values[0]
    g_label = 'Parametric Fit of Marginal' if marginal else 'Reference'

    dict_min = {}
    for ss_str in mean_df['sample_size_str'].unique():
        temp_df = mean_df[mean_df['sample_size_str'] == ss_str]
        idx = np.where(temp_df['OR'].values == np.min(temp_df['OR'].values))[0]
        dict_min[ss_str] = t1_linspace[idx][0]

    # plot_df['sample_size_and_min'] = plot_df['sample_size_str'].apply(
    #     lambda x: '%s, Minimum at: %s' % (x, round(dict_min[x], 3))
    # )

    sns.lineplot(x='t1_round', y='OR', hue='sample_size_str', data=plot_df)
    or_plot_df = pd.DataFrame.from_dict(data={'t1': t1_linspace, 'OR': or_vals})
    min_true_or = round(t1_linspace[np.where(or_vals == np.min(or_vals))[0]][0], 3)
    sns.lineplot(x='t1', y='OR', color='red', label='True Log Odds-Ratios', data=or_plot_df)
    plt.axvline(x=min_value_true, linestyle='-.', color='black')
    plt.legend(loc='best', fontsize=24)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], fontsize=24)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel(r'$\theta_1$', fontsize=26)
    plt.ylabel('Parametrized Sum of Log Odds-Ratios', fontsize=26)
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.title(r'%s Example, %s $\theta_0 = %s$, %s Data Observed, G=%s' % (
        run.title(), classifier + ' Classifier,\n', t0_val, ss, g_label), fontsize=30)
    outfile_name = 'odds_ratio_fit_or_over_t1_%s_%s_%s_%sobs_%sthetaval_%s.pdf' % (
        run, classifier.replace(' ', '-'), 'marginalg' if marginal else 'referenceg', ss,
        t0_val, datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.savefig(out_dir + outfile_name)
    plt.close()

    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(x='t1_round', y='OR', hue='sample_size_str', data=plot_df)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=min_value_true, linestyle='-.', color='black')
    plt.legend(loc='best', fontsize=24)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:], fontsize=24)
    plt.xlabel(r'$\theta_1$', fontsize=26)
    plt.ylabel('Parametrized Sum of Log Odds-Ratios', fontsize=26)
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.title(r'%s Example, %s $\theta_0 = %s$, %s Data Observed, G=%s' % (
        run.title(), classifier + ' Classifier,\n', t0_val, ss, g_label), fontsize=30)
    outfile_name = 'odds_ratio_fit_notrue_or_over_t1_%s_%s_%s_%sobs_%sthetaval_%s.pdf' % (
        run, classifier.replace(' ', '-'), 'marginalg' if marginal else 'referenceg', ss,
        t0_val, datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.savefig(out_dir + outfile_name)
    plt.close()


def or_distance_from_min(results_df, run, classifier, ss, t0_val, marginal, out_dir='images/'):
    g_label = 'Parametric Fit of Marginal' if marginal else 'Reference'
    min_or_df = results_df.loc[
        results_df.groupby(['rep', 'sample_size'])['OR'].idxmin()]
    min_true_df = results_df.loc[
        results_df.groupby(['rep', 'sample_size'])['true_or_values'].idxmin()]

    min_or_df = min_or_df[['rep', 't1', 'OR', 'true_or_values', 'sample_size']]
    min_true_df = min_true_df[['rep', 't1', 'OR', 'true_or_values', 'sample_size']]

    plot_df = pd.merge(min_or_df, min_true_df, on=['rep', 'sample_size'], how='left')
    plot_df['diff'] = plot_df[['t1_x', 't1_y']].apply(lambda x: np.abs(x[0] - x[1]), axis=1)

    plt.subplots(figsize=(20, 10))
    sns.boxplot(x='sample_size', y='diff', data=plot_df)
    plt.ylabel('Diff. of Estimated Minimum - True Minimum', fontsize=26)
    plt.xlabel('Training Size B', fontsize=26)
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title(r'%s Example, %s $\theta_0 = %s$, %s Data Observed, G=%s' % (
        run.title(), classifier + ' Classifier,\n', t0_val, ss, g_label), fontsize=30)
    outfile_name = 'odds_ratio_fit_diff_min_t1_%s_%s_%s_%sobs_%sthetaval_%s.pdf' % (
        run, classifier.replace(' ', '-'), 'marginalg' if marginal else 'referenceg', ss,
        t0_val, datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.savefig(out_dir + outfile_name)
    plt.close()


def relative_or_diff_over_t1(results_df, run, classifier, ss, t0_val, out_dir='images/'):

    plt.figure(figsize=(20, 10))
    sns.lineplot(x='t1_round', y='relative_diff', hue='sample_size_str', data=results_df)
    plt.legend(loc='best', fontsize=16)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel(r'$\theta_1$', fontsize=20)
    plt.title('%s Example, %s Classifier - %s Data Observed' % (
        run.title(), classifier, ss), fontsize=20)
    plt.ylabel('Relative Difference between Estimated and True OR', fontsize=20)

    outfile_name = 'or_fit_relative_diff_over_t1_%s_%s_%sobs_%sthetaval_%s.pdf' % (
        run, classifier, ss, t0_val, datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.savefig(out_dir + outfile_name)
    plt.close()


def lr_over_t1(results_df, t1_linspace, lr_vals, run, classifier, ss, out_dir='images/'):

    plt.figure(figsize=(20, 10))
    sns.boxplot(x='t1_round', y='OR', hue='sample_size_str', data=results_df)
    plt.scatter(t1_linspace, lr_vals, s=200, color='red', label='True LR')
    plt.legend(loc='best', fontsize=16)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel(r'$\mu_1$', fontsize=20)
    plt.ylabel('LR Value', fontsize=20)
    plt.title('%s Example, %s Classifier - %s Data Observed' % (
        run.title(), classifier, ss), fontsize=20)
    outfile_name = 'lr_fit_lr_over_t1_%s_%s_%sobs_%s.pdf' % (
        run, classifier, ss, datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.savefig(out_dir + outfile_name)
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.boxplot(x='t1_round', y='OR', hue='sample_size_str', data=results_df)
    plt.legend(loc='best', fontsize=16)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel(r'$\mu_1$', fontsize=20)
    plt.ylabel('LR Value', fontsize=20)
    plt.title('%s Example, %s Classifier - %s Data Observed' % (
        run.title(), classifier, ss), fontsize=20)
    outfile_name = 'lr_fit_notrue_lr_over_t1_%s_%s_%sobs_%s.pdf' % (
        run, classifier, ss, datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.savefig(out_dir + outfile_name)
    plt.close()


def relative_diff_over_t1(results_df, run, classifier, ss, out_dir='images/'):

    plt.figure(figsize=(20, 10))
    sns.boxplot(x='t1_round', y='relative_diff', hue='sample_size_str', data=results_df)
    plt.legend(loc='best', fontsize=16)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel(r'$\mu_1$', fontsize=20)
    plt.title('%s Example, %s Classifier - %s Data Observed' % (
        run.title(), classifier, ss), fontsize=20)
    plt.ylabel('Relative Difference between OR and LR', fontsize=20)

    outfile_name = 'lr_fit_relative_diff_over_t1_%s_%s_%sobs_%s.pdf' % (
        run, classifier, ss, datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.savefig(out_dir + outfile_name)
    plt.close()


def confidence_region_cutoff(values_vector, t1_linspace, run, classifier, ss, t0_val, out_dir='images/'):

    title_vec = ['%s: True LR/Cutoff, %s Data, (Theta: %s)' % (run, ss, t0_val),
                 '%s: %s Estimated OR/ QR Cutoff, %s Data, (Theta: %s)' % (
                     run, classifier, ss, t0_val)]
    col_vec = ['red', 'blue']
    plt.figure(figsize=(20, 10))
    for ii, (or_values, cutoff) in enumerate(values_vector):
        plt.subplot(1, 2, ii + 1)
        or_plot_df = pd.DataFrame.from_dict(data={'t1': t1_linspace, 'OR': or_values})
        sns.lineplot(x='t1', y='OR', color=col_vec[ii], data=or_plot_df)
        plt.axhline(y=cutoff, color=col_vec[ii], linestyle='--', alpha=0.5)
        plt.xlabel(r'$\mu_1$', fontsize=20)
        plt.ylabel('Value', fontsize=20)
        plt.title(title_vec[ii], fontsize=20)

    outfile_name = 'ci_fit_over_t1_%s_%s_%sobs_%s.pdf' % (
        run, classifier, ss, datetime.strftime(datetime.today(), '%Y-%m-%d-%H-%M')
    )
    plt.savefig(out_dir + outfile_name)
    plt.close()