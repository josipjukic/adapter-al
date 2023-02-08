from functools import partial
import os

import pickle
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import softmax

from al.sampler_mapping import AL_SAMPLERS


def load_results(base_dir="results/", dataset="SUBJ", model="BERT"):
    experiments = {}
    start_string = f"{dataset}-{model}"
    print(start_string)
    for filename in os.listdir(base_dir):
        if filename.startswith(start_string) and filename.endswith(".pkl"):
            with open(os.path.join(base_dir, filename), "rb") as f:
                results, meta = pickle.load(f)
                experiments[meta["al_sampler"]] = results

    del meta["al_sampler"]

    return experiments, meta


def al_auc(df, n=0):
    num_al_steps = df.index.get_level_values("al_iter").max() - n
    grouped = (
        df.groupby(["sampler", "experiment"]).agg(list).groupby("sampler").agg(list)
    )
    acc_auc = (
        grouped.test_accuracy.agg(lambda x: np.trapz(np.array(x)[:, n:], axis=1))
        / num_al_steps
    )
    f1_micro_auc = (
        grouped.f1_micro.agg(lambda x: np.trapz(np.array(x)[:, n:], axis=1))
        / num_al_steps
    )
    f1_macro_auc = (
        grouped.f1_macro.agg(lambda x: np.trapz(np.array(x)[:, n:], axis=1))
        / num_al_steps
    )
    acc_avg = grouped.test_accuracy.agg(lambda x: np.mean(x, axis=1))
    f1_micro_avg = grouped.f1_micro.agg(lambda x: np.mean(x, axis=1))
    f1_macro_avg = grouped.f1_macro.agg(lambda x: np.mean(x, axis=1))
    df_auc = pd.DataFrame(
        {
            "acc_auc": acc_auc,
            "micro_auc": f1_micro_auc,
            "macro_auc": f1_macro_auc,
            "acc_avg": acc_avg,
            "micro_avg": f1_micro_avg,
            "macro_avg": f1_macro_avg,
        }
    )
    df_auc["acc_auc_mean"] = df_auc.acc_auc.apply(np.mean)
    df_auc["acc_auc_std"] = df_auc.acc_auc.apply(np.std)
    df_auc["micro_auc_mean"] = df_auc.micro_auc.apply(np.mean)
    df_auc["micro_auc_std"] = df_auc.micro_auc.apply(np.std)
    df_auc["macro_auc_mean"] = df_auc.macro_auc.apply(np.mean)
    df_auc["macro_auc_std"] = df_auc.macro_auc.apply(np.std)
    df_auc["acc_avg_mean"] = df_auc.acc_avg.apply(np.mean)
    df_auc["acc_avg_std"] = df_auc.acc_avg.apply(np.std)
    df_auc["micro_avg_mean"] = df_auc.micro_avg.apply(np.mean)
    df_auc["micro_avg_std"] = df_auc.micro_avg.apply(np.std)
    df_auc["macro_avg_mean"] = df_auc.macro_avg.apply(np.mean)
    df_auc["macro_avg_std"] = df_auc.macro_avg.apply(np.std)

    return df_auc


def results_to_df(experiments, mode="best", epoch=-1):
    if mode not in MODE_DICT:
        raise ValueError(
            f"Mode {mode} is not supported. Choose 'last' or 'best' epoch."
        )

    extract_fn = MODE_DICT[mode]
    if extract_fn == MODE_DICT["ith"]:
        extract_fn = partial(extract_fn, epoch=epoch)

    dfs_tr = []
    for sampler, exp_set in experiments.items():
        df_tr = extract_fn(exp_set)
        df_tr["sampler"] = sampler
        dfs_tr.append(df_tr)

    new_df_tr = pd.concat(dfs_tr)

    return new_df_tr


def extract_cartography(crt, exp_index):
    df_crt = pd.DataFrame(crt)
    df_crt["experiment"] = exp_index
    df_crt["example"] = range(len(df_crt))
    df_crt.set_index(["experiment", "example"], inplace=True)
    return df_crt


def extract_general_cartography(crts, exp_index):
    correctness = []
    confidence = []
    variability = []
    forgetfulness = []
    threshold_closeness = []
    for crt in crts:
        correctness.append(crt["correctness"])
        confidence.append(crt["confidence"])
        variability.append(crt["variability"])
        forgetfulness.append(crt["forgetfulness"])
        threshold_closeness.append(crt["threshold_closeness"])

    df_crt = pd.DataFrame(
        {
            "correctness": correctness,
            "confidence": confidence,
            "variability": variability,
            "forgetfulness": forgetfulness,
            "threshold_closeness": threshold_closeness,
        }
    )
    df_crt["experiment"] = exp_index
    df_crt.set_index("experiment", inplace=True)
    return df_crt


def extract_ith_epoch(exp_set, besov_flag=False, epoch=-1):
    dfs_tr = []
    for exp_index, experiment in enumerate(exp_set):
        train = experiment["train"]
        train_vals = [tr[epoch]["loss"] for tr in train]
        test = experiment["eval"]
        test_vals = [te[epoch]["accuracy"] for te in test]
        labeled_vals = experiment["labeled"]

        iter_vals = list(range(len(labeled_vals)))

        df_tr = pd.DataFrame(
            {
                "al_iter": iter_vals,
                "labeled": labeled_vals,
                "train_loss": train_vals,
                "test_accuracy": test_vals,
            }
        )
        df_tr["experiment"] = exp_index
        df_tr.set_index(["experiment", "al_iter"], inplace=True)

        dfs_tr.append(df_tr)

    new_df_tr = pd.concat(dfs_tr)
    return new_df_tr


def extract_last_epoch(exp_set):
    dfs_tr = []
    for exp_index, experiment in enumerate(exp_set):
        train = experiment["train"]
        train_vals = [tr[-1]["loss"] for tr in train]
        test = experiment["eval"]
        accs = [te[-1]["accuracy"] for te in test]
        f1_micro = [te[-1]["f1_micro"] for te in test]
        f1_macro = [te[-1]["f1_macro"] for te in test]

        labeled_vals = experiment["labeled"]
        iter_vals = list(range(len(labeled_vals)))
        selected = experiment["selected"]
        if len(selected) > len(iter_vals):
            selected = selected[:-1]
        df_tr = pd.DataFrame(
            {
                "al_iter": iter_vals,
                "labeled": labeled_vals,
                "train_loss": train_vals,
                "test_accuracy": accs,
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
                "selected": selected,
            }
        )

        df_tr["experiment"] = exp_index
        df_tr.set_index(["experiment", "al_iter"], inplace=True)

        dfs_tr.append(df_tr)

    new_df_tr = pd.concat(dfs_tr)
    return new_df_tr


def extract_best_epoch(exp_set):
    dfs_tr = []
    for exp_index, experiment in enumerate(exp_set):
        train = experiment["train"]
        test = experiment["eval"]
        sels = experiment["selected"]
        train_vals, f1_micro, f1_macro, accs = [], [], [], []
        indices = []
        for tr, te, sel in zip(train, test, sels):
            test_accs = [t["accuracy"] for t in te]
            i = np.argpartition(test_accs, -1)[-1]
            indices.append(i)
            train_vals.append(tr[i]["loss"])
            f1_micro.append(te[i]["f1_micro"])
            f1_macro.append(te[i]["f1_macro"])
            accs.append(te[i]["accuracy"])

        labeled_vals = experiment["labeled"]
        iter_vals = list(range(len(labeled_vals)))

        selected = experiment["selected"]
        if len(selected) > len(iter_vals):
            selected = selected[:-1]

        df_tr = pd.DataFrame(
            {
                "al_iter": iter_vals,
                "labeled": labeled_vals,
                "train_loss": train_vals,
                "test_accuracy": accs,
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
                "selected": selected,
            }
        )

        df_tr["experiment"] = exp_index
        df_tr.set_index(["experiment", "al_iter"], inplace=True)

        dfs_tr.append(df_tr)

    new_df_tr = pd.concat(dfs_tr)
    return new_df_tr


def plot_al_accuracy(data, metric="f1_micro", figsize=(12, 8), ci=90):
    plt.figure(figsize=figsize)
    sns.lineplot(
        data=data,
        x="labeled",
        y=metric,
        hue="sampler",
        style="sampler",
        markers=True,
        dashes=False,
        ci=ci,
    )


def plot_besov_index(data, figsize=(12, 8), ci=90):
    plt.figure(figsize=figsize)
    sns.lineplot(
        data=data,
        x="labeled",
        y="besov_index",
        hue="sampler",
        style="sampler",
        markers=True,
        dashes=False,
        ci=ci,
    )


def plot_experiment_set(df_tr, meta, sampler, figsize=(12, 16), ci=90):
    df_tr_filt = df_tr[df_tr.sampler == sampler]
    _, axs = plt.subplots(2, figsize=figsize, sharex=True)
    axs[0].set_title(f"{meta['dataset']} - {meta['model']} - {sampler}")
    sns.lineplot(
        ax=axs[0], data=df_tr_filt, x="labeled", y="train_loss", color="r", ci=ci
    )
    sns.lineplot(
        ax=axs[1], data=df_tr_filt, x="labeled", y="test_accuracy", color="g", ci=ci
    )
    plt.show()


def scatter_it(df, meta, hue_metric="correct", show_hist=True):
    # Subsample data to plot, so the plot is not too busy.
    dataframe = df

    if hue_metric == "correct":
        # Normalize correctness to a value between 0 and 1.
        dataframe = dataframe.assign(
            corr_frac=lambda d: d.correctness / d.correctness.max()
        )
        dataframe = dataframe.sort_values("corr_frac")
        dataframe[hue_metric] = [f"{x:.1f}" for x in dataframe["corr_frac"]]
    elif hue_metric == "forget":
        # Normalize forgetfulness to a value between 0 and 1.
        dataframe = dataframe.assign(
            forg_frac=lambda d: d.forgetfulness / d.forgetfulness.max()
        )
        dataframe = dataframe.sort_values("forg_frac")
        dataframe[hue_metric] = [f"{x:.1f}" for x in dataframe["forg_frac"]]
    else:
        raise ValueError(
            f"Hue metric {hue_metric} is not supported. Choose from ['correct', 'forget']."
        )

    main_metric = "variability"
    other_metric = "confidence"

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        ax0 = axs
    else:
        fig = plt.figure(
            figsize=(16, 10),
        )
        gs = fig.add_gridspec(2, 3, height_ratios=[5, 1])

        ax0 = fig.add_subplot(gs[0, :])

    ### Make the scatterplot.

    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(
        x=main_metric,
        y=other_metric,
        ax=ax0,
        data=dataframe,
        hue=hue,
        palette=pal,
        style=style,
        s=30,
    )

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    an1 = ax0.annotate(
        "ambiguous",
        xy=(0.9, 0.5),
        xycoords="axes fraction",
        fontsize=15,
        color="black",
        va="center",
        ha="center",
        bbox=bb("black"),
    )
    an2 = ax0.annotate(
        "easy-to-learn",
        xy=(0.27, 0.85),
        xycoords="axes fraction",
        fontsize=15,
        color="black",
        va="center",
        ha="center",
        bbox=bb("r"),
    )
    an3 = ax0.annotate(
        "hard-to-learn",
        xy=(0.35, 0.25),
        xycoords="axes fraction",
        fontsize=15,
        color="black",
        va="center",
        ha="center",
        bbox=bb("b"),
    )

    if not show_hist:
        plot.legend(
            ncol=1,
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fancybox=True,
            shadow=True,
        )
    else:
        plot.legend(fancybox=True, shadow=True, ncol=1)
    plot.set_xlabel("variability")
    plot.set_ylabel("confidence")

    if show_hist:
        plot.set_title(
            f"{meta['dataset']} Data Map - {meta['model']} model - {len(df)} datapoints",
            fontsize=17,
        )

        # Make the histograms.
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[1, 2])

        plott0 = dataframe.hist(column=["confidence"], ax=ax1, color="#622a87")
        plott0[0].set_title("")
        plott0[0].set_xlabel("confidence")
        plott0[0].set_ylabel("density")

        plott1 = dataframe.hist(column=["variability"], ax=ax2, color="teal")
        plott1[0].set_title("")
        plott1[0].set_xlabel("variability")

        if hue_metric == "correct":
            plot2 = sns.countplot(x="correct", data=dataframe, color="#86bf91", ax=ax3)
            ax3.xaxis.grid(True)  # Show the vertical gridlines

            plot2.set_title("")
            plot2.set_xlabel("correctness")
            plot2.set_ylabel("")

        else:
            plot2 = sns.countplot(x="forget", data=dataframe, color="#86bf91", ax=ax3)
            ax3.xaxis.grid(True)  # Show the vertical gridlines

            plot2.set_title("")
            plot2.set_xlabel("forgetfulness")
            plot2.set_ylabel("")

    fig.tight_layout()
    return fig


def plot_cartography(
    df_crt, sampler, al_iter, meta, hue_metric="correct", show_hist=True
):
    df = convert_cartography_df(df_crt, sampler, al_iter)
    scatter_it(df, meta, hue_metric=hue_metric, show_hist=show_hist)


def plot_general_cartography(df, meta, hue_metric="correct", show_hist=True):
    scatter_it(df, meta, hue_metric=hue_metric, show_hist=show_hist)


def convert_cartography_df(df, sampler, al_iter):
    df_i = df.loc[(sampler, al_iter)]
    new_df = pd.DataFrame(
        {
            "correctness": df_i.correctness.tolist(),
            "confidence": df_i.confidence.tolist(),
            "variability": df_i.variability.tolist(),
            "forgetfulness": df_i.forgetfulness.tolist(),
            "threshold_closeness": df_i.threshold_closeness.tolist(),
        }
    )

    return new_df


def cartography_average(df):
    grouped = df.groupby(level=1)
    new_df = pd.DataFrame()
    new_df["correctness"] = grouped.correctness.agg(np.median)
    new_df["confidence"] = grouped.confidence.agg(np.mean)
    new_df["variability"] = grouped.variability.agg(np.mean)
    new_df["forgetfulness"] = grouped.forgetfulness.agg(np.median)
    new_df["threshold_closeness"] = grouped.threshold_closeness.agg(np.mean)
    return new_df


def general_cartography_average(new_df_crt):
    new_df_crt_avg = pd.DataFrame()
    new_df_crt_avg["correctness"] = np.median(
        np.array(new_df_crt.correctness.tolist()), 0
    )
    new_df_crt_avg["confidence"] = np.array(new_df_crt.confidence.tolist()).mean(0)
    new_df_crt_avg["variability"] = np.array(new_df_crt.variability.tolist()).mean(0)
    new_df_crt_avg["forgetfulness"] = np.median(
        np.array(new_df_crt.forgetfulness.tolist()), 0
    )
    new_df_crt_avg["threshold_closeness"] = np.array(
        new_df_crt.threshold_closeness.tolist()
    ).mean(0)
    return new_df_crt_avg


def load_cartography(base_dir="results/", dataset="SUBJ", model="ELECTRA"):
    for filename in os.listdir(base_dir):
        if filename.startswith(f"{dataset}-{model}-cartography") and filename.endswith(
            ".pkl"
        ):
            with open(os.path.join(base_dir, filename), "rb") as f:
                results, meta = pickle.load(f)
            break

    dfs_crt_train = []
    dfs_crt_test = []
    for exp_index, experiment in enumerate(results):
        df_crt_train = extract_general_cartography(
            experiment["cartography"]["train"], exp_index
        )
        df_crt_test = extract_general_cartography(
            experiment["cartography"]["test"], exp_index
        )

        dfs_crt_train.append(df_crt_train)
        dfs_crt_test.append(df_crt_test)

    new_df_crt_train = pd.concat(dfs_crt_train)
    new_df_crt_test = pd.concat(dfs_crt_test)
    return (
        general_cartography_average(new_df_crt_train),
        general_cartography_average(new_df_crt_test),
        meta,
    )


def load_cartography_from_results(results):
    dfs_crt_train = []
    for exp_index, experiment in enumerate(results):
        df_crt_train = extract_cartography(
            experiment["cartography"]["train"], exp_index
        )
        dfs_crt_train.append(df_crt_train)

    new_df_crt_train = pd.concat(dfs_crt_train)
    return cartography_average(new_df_crt_train)


def besov_score(alphas):
    alphas = np.array(alphas)
    soft_alphas = softmax(alphas)
    return soft_alphas


def best_by_besov(exp_set, metric):
    dfs_tr = []
    for exp_index, experiment in enumerate(exp_set):
        train = experiment["train"]
        test = experiment["eval"]
        train_vals, accs, f1_micro, f1_macro = [], [], [], []
        indices = []
        for tr, te in zip(train, test):
            besov_index = [np.mean(t["repr"]["alpha"]) for t in te]
            i = np.argmax(besov_index)
            indices.append(i)
            train_vals.append(tr[i]["loss"])
            f1_micro.append(te[i]["f1_micro"])
            f1_macro.append(te[i]["f1_macro"])
            accs.append(te[i]["accuracy"])
        labeled_vals = experiment["labeled"]
        iter_vals = list(range(len(labeled_vals)))
        df_tr = pd.DataFrame(
            {
                "al_iter": iter_vals,
                "labeled": labeled_vals,
                "train_loss": train_vals,
                "test_accuracy": accs,
                "f1_micro": f1_micro,
                "f1_macro": f1_macro,
            }
        )
        df_tr["experiment"] = exp_index
        df_tr.set_index(["experiment", "al_iter"], inplace=True)

        dfs_tr.append(df_tr)

    new_df_tr = pd.concat(dfs_tr)
    return new_df_tr


def extract_epoch_data(exp_set):
    dfs_tr = []
    for exp_index, experiment in enumerate(exp_set):
        train = experiment["train"]
        test = experiment["eval"]
        untrained = experiment["untrained"]
        train_vals, test_vals = [], []
        besov = []
        sample_besov = []
        untrained_besov = []
        indices = []
        for tr, te, un in zip(train, test, untrained):
            accs = [t["accuracy"] for t in te]
            i = np.argmax(accs)
            indices.append(i)
            train_vals.append([t["loss"] for t in tr])
            test_vals.append(accs)
            besov.append([t["repr"]["alpha"] for t in te])
            if "sample_repr" in te[0]:
                sample_besov.append([t["sample_repr"]["alpha"] for t in te])
            else:
                sample_besov.append(np.nan)
            untrained_besov.append(un["alpha"])
        labeled_vals = experiment["labeled"]
        iter_vals = list(range(len(labeled_vals)))
        df_tr = pd.DataFrame(
            {
                "al_iter": iter_vals,
                "labeled": labeled_vals,
                "train_loss": train_vals,
                "test_accuracy": test_vals,
                "besov_index": besov,
                "sample_besov": sample_besov,
                "untrained_besov": untrained_besov,
            }
        )
        df_tr["experiment"] = exp_index
        df_tr.set_index(["experiment", "al_iter"], inplace=True)

        dfs_tr.append(df_tr)

    new_df_tr = pd.concat(dfs_tr)
    return new_df_tr


MODE_DICT = {
    "last": extract_last_epoch,
    "best": extract_best_epoch,
    "ith": extract_ith_epoch,
    "besov": best_by_besov,
}

SAMPLERS = [
    "random",
    "entropy",
    "repr",
    "badge",
    "entropy_dropout",
    "anti_entropy",
    "core_set",
    "dal",
    "anti_entropy",
]
