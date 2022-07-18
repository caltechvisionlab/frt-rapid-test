# This visualization code is from a Jupyter notebook entitled
# "Precision-recall curve - conservative estimates".

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# Parameters for figure geometry and fonts
FIG_WIDTH = 10
FIG_HEIGHT = 10
FIG_SIZE = (FIG_WIDTH, FIG_HEIGHT)
LABEL_FONT = 16
TITLE_FONT = 22


def _process_actual_data(provider_aggregate_metrics):
    X0 = provider_aggregate_metrics["no_match_confidences"]
    X1 = provider_aggregate_metrics["true_match_confidences"]

    N0 = len(X0)
    N1 = len(X1)

    Y0 = np.zeros(N0).astype(int)
    Y1 = np.ones(N1).astype(int)

    X = np.concatenate((X0, X1), axis=0)
    Y = np.concatenate((Y0, Y1), axis=0)

    # estimate the pdfs of the two sets using kernel estimation
    # this is used to produce a conservative estimate of the PRC
    bandwidth_method = "scott"
    bandwidth_method = "silverman"
    # kernel estimator of the density
    pDist0 = stats.gaussian_kde(X0, bw_method=bandwidth_method)
    pDist1 = stats.gaussian_kde(X1, bw_method=bandwidth_method)

    return X, Y, pDist0, pDist1


def histogram_data(
    X,
    Y,
    pdist0,
    pdist1,
    tag="",
    output_path=None,
    dataset_label=None,
    provider_name=None,
):
    """Just to take a look at the synthetic data and at the data fits"""
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # figure out histogram bins that work for all data
    M = np.max(X)
    m = np.min(X)
    # figure out a reasonable n. of bins
    N_bins = np.ceil(3 + np.sqrt(X.shape[0]))
    bin_sz = (M - m) / (N_bins - 1)
    BIN_CTRS = np.arange(m - 2 * bin_sz, M + 2.5 * bin_sz, bin_sz)
    BIN_EDGES = np.arange(m - 2.5 * bin_sz, M + 3 * bin_sz, bin_sz)

    class_labels = set(Y)  # figure out the different labels

    label_to_meaningful_text = {
        0: "non-match",
        1: "match",
    }

    # plot the histograms of all the data classes
    N = []
    for i, l in enumerate(class_labels):  # for each label histogram the data
        ind_x = np.nonzero(Y == l)[0]
        N.append(len(ind_x))
        hh = np.histogram(X[ind_x], bins=BIN_EDGES, density=False)
        plt.plot(BIN_CTRS, hh[0], label=f"{label_to_meaningful_text[l]} (raw)", lw=3)

    if pdist0 is not None:
        plt.plot(
            BIN_CTRS,
            pdist0(BIN_CTRS) * bin_sz * N[0],
            label=f"{label_to_meaningful_text[0]} (smoothed)",
            lw=5,
        )
    if pdist1 is not None:
        plt.plot(
            BIN_CTRS,
            pdist1(BIN_CTRS) * bin_sz * N[1],
            label=f"{label_to_meaningful_text[1]} (smoothed)",
            lw=5,
        )
    plt.legend(fontsize=16)
    plt.xlabel("confidence", fontsize=16)
    plt.ylabel("count", fontsize=16)

    provider_txt = f" for {provider_name}" if provider_name else ""
    dataset_txt = f" ({dataset_label})" if dataset_label else ""
    # TODO - figure out how to get better dynamic sized titles
    plt.title(f"Histogram of confidences{provider_txt}{dataset_txt}", fontsize=20)

    # TODO - refactor and borrow from v1
    if output_path is not None:
        plt.savefig(output_path + ".pdf", format="pdf")
        plt.clf()
    else:
        plt.show()


def plot_performance_curve(
    X,
    Y,
    pdist0,
    pdist1,
    ax,
    tag="",
    type="precision recall",
    regularized=False,
    output_path=None,
    dataset_label=None,
    provider_name=None,
):
    """Plot the precision-recall curve (or the false alarm vs false reject)
    X - array of `confidence' values
    Y - ground truth binary label of each sample
    """

    class_labels = list(set(Y))

    # sort the samples of all classes
    idx = np.argsort(-X)
    Xs = X[idx]
    Ys = Y[idx]

    # generate the coordinates to sample the models
    M = np.max(X)
    m = np.min(X)
    n_samples = 1000
    step_sz = (M - m) / n_samples
    xx = np.arange(M, m, -step_sz)
    Nm0 = step_sz * np.sum(Y == class_labels[0]) * np.cumsum(pdist0(xx))
    Nm1 = step_sz * np.sum(Y == class_labels[1]) * np.cumsum(pdist1(xx))

    # determine thresholds for plotting PRC
    T = (Xs[1:] + Xs[0:-1]) / 2  # thresholds are intermediate points

    # compute counts in increasing value of threshold
    class_labels = list(set(Y))
    # number of false alarms above threshold
    N0 = np.cumsum(Ys == class_labels[0])
    N1 = np.cumsum(Ys == class_labels[1])  # number detections above threshold

    provider_txt = f" for {provider_name}" if provider_name else ""
    dataset_txt = f" ({dataset_label})" if dataset_label else ""

    if type == "precision recall":
        P = N1 / (N0 + N1)
        R = N1 / np.max(N1)
        Pm = Nm1 / (Nm0 + Nm1)
        Rm = Nm1 / np.max(Nm1)
        ax.plot(
            R,
            P,
            "o-",
            lw=3,
            label=(tag + " (raw)" if tag else "raw")
            if output_path is not None
            else tag,
        )
        if output_path is not None:
            ax.plot(Rm, Pm, lw=5, label=tag + " (smoothed)" if tag else "smoothed")
        ax.set_xlabel("Recall", fontsize=20)
        ax.set_ylabel("Precision", fontsize=20)
        ax.legend(fontsize=16)
        # TODO - figure out how to get better dynamic sized titles
        ax.set_title(f"Precision-Recall curve{provider_txt}{dataset_txt}", fontsize=12)

    if type == "F1 score":
        P = N1 / (N0 + N1)
        R = N1 / np.max(N1)
        Pm = Nm1 / (Nm0 + Nm1)
        Rm = Nm1 / np.max(Nm1)
        F1 = 2.0 / (1 / R + 1 / P)
        F1m = 2.0 / (1 / Rm + 1 / Pm)
        ax.plot(
            Xs,
            F1,
            "o-",
            lw=3,
            label=(tag + " (raw)" if tag else "raw")
            if output_path is not None
            else tag,
        )
        if output_path is not None:
            ax.plot(xx, F1m, lw=5, label=tag + " (smoothed)" if tag else "smoothed")
        ax.set_xlabel("Threshold", fontsize=16)
        ax.set_ylabel("F1", fontsize=16)
        ax.legend(fontsize=16)
        # TODO - figure out how to get better dynamic sized titles
        ax.set_title(f"F1 score{provider_txt}{dataset_txt}", fontsize=12)

    if type == "precision recall inverted":
        P = N1 / (N0 + N1)
        R = N1 / np.max(N1)
        Pm = Nm1 / (Nm0 + Nm1)
        Rm = Nm1 / np.max(Nm1)

        # compute indices where the error is not zero
        i_R_NZ = np.nonzero(1 - R)
        i_P_NZ = np.nonzero(1 - P)
        i_NZ = np.intersect1d(i_R_NZ, i_P_NZ)

        # plot the data
        ax.plot(
            1 - R[i_NZ],
            1 - P[i_NZ],
            "o-",
            lw=3,
            label=(tag + " (raw)" if tag else "raw")
            if output_path is not None
            else tag,
        )
        if output_path is not None:
            ax.plot(
                1 - Rm, 1 - Pm, lw=5, label=tag + " (smoothed)" if tag else "smoothed"
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("1-Recall", fontsize=16)
        ax.set_ylabel("1-Precision", fontsize=16)
        ax.legend(fontsize=16)
        # TODO - figure out how to get better dynamic sized titles
        ax.set_title(
            f"Inverted Precision-Recall curve{provider_txt}{dataset_txt}", fontsize=12
        )
        ax.set_xlim([0.001, 1])
        ax.set_ylim([0.001, 1])

    if type == "miss vs false alarms":
        FA = N0 / np.max(N0)
        FR = 1 - N1 / np.max(N1)
        FAm = Nm0 / np.max(Nm0)
        FRm = 1 - Nm1 / np.max(Nm1)

        # compute indices where the error is not zero
        i_FA_NZ = np.nonzero(FA)
        i_FR_NZ = np.nonzero(FR)
        i_NZ = np.intersect1d(i_FA_NZ, i_FR_NZ)

        ax.plot(
            FA[i_NZ],
            FR[i_NZ],
            "o-",
            lw=3,
            label=(tag + " (raw)" if tag else "raw")
            if output_path is not None
            else tag,
        )
        if output_path is not None:
            ax.plot(FAm, FRm, lw=5, label=tag + " (smoothed)" if tag else "smoothed")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("False alarm rate", fontsize=16)
        ax.set_ylabel("False reject rate", fontsize=16)
        ax.legend(fontsize=16)
        ax.set_xlim([0.001, 1])
        ax.set_ylim([0.001, 1])
        # TODO - figure out how to get better dynamic sized titles
        ax.set_title(
            f"False rejects vs false alarms{provider_txt}{dataset_txt}", fontsize=12
        )

    # TODO - refactor and borrow from v1
    if output_path is not None:
        plt.savefig(output_path + ".pdf", format="pdf")
        plt.clf()
    # else:
    #     plt.show()


def prepare_empty_plots_v2(num):
    axes = []
    figs = []
    for i in range(num):
        fig, ax = plt.subplots()
        axes.append(ax)
        figs.append(fig)
    return axes, figs


def _format_large_legend(ax, huge_legend=False):
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot-in-matplotlib

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    kwd = dict()
    if huge_legend:
        # If there are too many labels, make the label text smaller.
        kwd = dict(fontsize=5)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), **kwd)


def plot_data_v2_consolidate(axes, figs, output_path_root, huge_legend=False):
    output_path = output_path_root + "_prec_recall"
    _format_large_legend(axes[0], huge_legend=huge_legend)
    figs[0].savefig(output_path + ".pdf", format="pdf")

    output_path = output_path_root + "_F1"
    _format_large_legend(axes[1], huge_legend=huge_legend)
    figs[1].savefig(output_path + ".pdf", format="pdf")

    output_path = output_path_root + "_prec_recall_inv"
    _format_large_legend(axes[2], huge_legend=huge_legend)
    figs[2].savefig(output_path + ".pdf", format="pdf")

    output_path = output_path_root + "_miss_false_alarms"
    _format_large_legend(axes[3], huge_legend=huge_legend)
    figs[3].savefig(output_path + ".pdf", format="pdf")

    plt.clf()

    # TODO - is this necessary and should I apply it elsewhere?
    # for fig in figs:
    #     plt.close(fig)


def plot_data_v2(
    provider_aggregate_metrics,
    output_path_root,
    axes=None,
    dataset_label=None,
    provider_name=None,
    tag="",
):
    try:
        X, Y, pdist0, pdist1 = _process_actual_data(provider_aggregate_metrics)
    except ValueError as e:
        # Could occur if a subperson we are running only has 2 faces
        # annotated as a match, then there's an error like
        # ValueError: `dataset` input should have multiple elements.
        print(e)
        return

    if output_path_root is not None:
        histogram_data(
            X,
            Y,
            pdist0,
            pdist1,
            output_path=output_path_root + "_histogram",
            dataset_label=dataset_label,
            provider_name=provider_name,
        )

    if axes is None:
        fig, ax = plt.subplots()
        output_path = output_path_root + "_prec_recall"
    else:
        ax = axes[0]
        output_path = None
    plot_performance_curve(
        X,
        Y,
        pdist0,
        pdist1,
        ax,
        type="precision recall",
        output_path=output_path,
        tag=tag,
        dataset_label=dataset_label,
        provider_name=provider_name,
    )

    if axes is None:
        fig, ax = plt.subplots()
        output_path = output_path_root + "_F1"
    else:
        ax = axes[1]
        output_path = None
    plot_performance_curve(
        X,
        Y,
        pdist0,
        pdist1,
        ax=ax,
        type="F1 score",
        output_path=output_path,
        tag=tag,
        dataset_label=dataset_label,
        provider_name=provider_name,
    )

    if axes is None:
        fig, ax = plt.subplots()
        output_path = output_path_root + "_prec_recall_inv"
    else:
        ax = axes[2]
        output_path = None
    plot_performance_curve(
        X,
        Y,
        pdist0,
        pdist1,
        ax=ax,
        type="precision recall inverted",
        output_path=output_path,
        tag=tag,
        dataset_label=dataset_label,
        provider_name=provider_name,
    )

    if axes is None:
        fig, ax = plt.subplots()
        output_path = output_path_root + "_miss_false_alarms"
    else:
        ax = axes[3]
        output_path = None
    plot_performance_curve(
        X,
        Y,
        pdist0,
        pdist1,
        ax=ax,
        type="miss vs false alarms",
        output_path=output_path,
        tag=tag,
        dataset_label=dataset_label,
        provider_name=provider_name,
    )
