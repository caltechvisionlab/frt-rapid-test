import matplotlib.pyplot as plt
import numpy as np
import os

from ..analyzer.results_supervised import _compute_match_rates_optimized
from ..analyzer.results_unsupervised import discrete_cdf, compute_FMR_FNMR_discrete_dist


EM_CALIBRATION_THRESHOLDS_TO_PLOT = [0.4, 0.5, 0.6, 0.7, 0.8]


def _output(name):
    if name is None:
        plt.show()
    else:
        plt.savefig(name + ".pdf", format="pdf")
        plt.clf()


def generate_unsupervised_plots(
    true_match_confidences,
    true_nonmatch_confidences,
    kernel_output,
    bootstrapped_kernel_output,
    dirname,
    threshold_points_to_plot=EM_CALIBRATION_THRESHOLDS_TO_PLOT,
    estimate_nonmatch_distribution_each_iteration=False,
):
    version = int(estimate_nonmatch_distribution_each_iteration)
    kde_nm, kde_m, _ = kernel_output

    include_supervised = true_match_confidences is not None

    # 1. Plot the match distribution.
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 200)
    ax.plot(x, kde_m(x), label="EM estimate")
    if include_supervised:
        ax.hist(
            true_match_confidences, bins=20, density=True, alpha=0.1, label="histogram"
        )
    ax.set_xlabel("x")
    ax.set_ylabel("pdf(x)")
    ax.set_title("Match confidences")
    ax.legend()
    _output(dirname + os.sep + "match_distribution_estimate" + f"_v{version}")

    # 2. Plot the non-match distribution.
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 200)
    ax.plot(x, kde_nm(x), label="EM estimate")
    if include_supervised:
        ax.hist(
            true_nonmatch_confidences,
            bins=20,
            density=True,
            alpha=0.1,
            label="histogram",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("pdf(x)")
    ax.set_title("Nonmatch confidences")
    ax.legend()
    _output(dirname + os.sep + "non-match_distribution_estimate" + f"_v{version}")

    # 3. Compute various quantities to prepare upcoming plots.
    # TODO - see if these calls can be made in the analysis package
    # to preserve the appropriate functions for the analyzer and visualizer.
    delta = 0.001
    _, nonmatch_cdf = discrete_cdf(kde_nm, delta)
    _, match_cdf = discrete_cdf(kde_m, delta)
    fmr, fnmr = compute_FMR_FNMR_discrete_dist(nonmatch_cdf, match_cdf)
    if include_supervised:
        _, real_fmr, real_fnmr = _compute_match_rates_optimized(
            true_match_confidences,
            true_nonmatch_confidences,
            delta,
        )

    # 4. Plot FMR/FNMR at non-zero values
    fig, ax = plt.subplots()
    if include_supervised:
        i_FA_NZ = np.nonzero(real_fmr)
        i_FR_NZ = np.nonzero(real_fnmr)
        i_NZ = np.intersect1d(i_FA_NZ, i_FR_NZ)
        ax.plot(
            real_fmr[i_NZ],
            real_fnmr[i_NZ],
            color="red",
            linewidth=1,
            label="Ground truth",
        )
    ax.plot(fmr, fnmr, color="green", label="Estimate")
    ax.set_xlabel("FMR")
    ax.set_ylabel("FNMR")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"FNMR vs FMR")
    _ = ax.legend(loc="center left")
    # TODO - is this bound ok?
    plt.gca().set_xlim(left=10**-5, right=2)
    plt.gca().set_ylim(bottom=10**-5, top=2)
    _output(dirname + os.sep + "fmr_fnmr_estimate" + f"_v{version}")

    # 5. Plot FMR/FNMR at non-zero values (with scatter points at certain thresholds).
    fig, ax = plt.subplots()
    if include_supervised:
        i_FA_NZ = np.nonzero(real_fmr)
        i_FR_NZ = np.nonzero(real_fnmr)
        i_NZ = np.intersect1d(i_FA_NZ, i_FR_NZ)
        ax.plot(
            real_fmr[i_NZ],
            real_fnmr[i_NZ],
            color="red",
            linewidth=1,
            label="Ground truth",
        )
    ax.plot(fmr, fnmr, color="green", label="Estimate")

    # This block plots a (FMR, FNMR) point at various T to show the
    # calibration between our kernel EM estimate FMR/FNMR and the true
    # FMR/FMNR. This is important because our estimate could be shifted
    # (e.g., estimate FMR(T) = real FMR(T+0.1) and same with the FNMR's)
    # yet appear to perfectly match the true curve.
    if threshold_points_to_plot is not None and include_supervised:
        matches = np.array(true_match_confidences)
        nonmatches = np.array(true_nonmatch_confidences)

        for t in threshold_points_to_plot:
            # For true data
            pt_fmr = np.sum(nonmatches >= t) / len(nonmatches)
            pt_fnmr = np.sum(matches < t) / len(matches)

            ax.scatter(pt_fmr, pt_fnmr, color="red")

            # For kernel fit
            idx = int(t // delta)
            pt_fmr = fmr[idx]
            pt_fnmr = fnmr[idx]

            ax.scatter(pt_fmr, pt_fnmr, color="green")
    ax.set_xlabel("FMR")
    ax.set_ylabel("FNMR")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"FNMR vs FMR")
    _ = ax.legend(loc="center left")
    # TODO - is this bound ok?
    plt.gca().set_xlim(left=10**-5, right=2)
    plt.gca().set_ylim(bottom=10**-5, top=2)
    _output(dirname + os.sep + "calibrated_fmr_fnmr_estimate" + f"_v{version}")

    # 6. Plot curves from running EM on bootstrapped datasets.
    fnmr_q, x_fmr_arr, _ = bootstrapped_kernel_output

    fig, ax = plt.subplots()
    ax.fill_between(
        x_fmr_arr,
        fnmr_q[0.1],
        fnmr_q[0.9],
        color="green",
        alpha=0.1,
        label="10-90% CI estimate",
    )
    ax.fill_between(
        x_fmr_arr,
        fnmr_q[0.25],
        fnmr_q[0.75],
        color="green",
        alpha=0.3,
        label="25-75% CI estimate",
    )
    ax.plot(x_fmr_arr, fnmr_q[0.5], color="green", label="Median estimate")

    # Plot FMR/FNMR at non-zero values
    if include_supervised:
        i_FA_NZ = np.nonzero(real_fmr)
        i_FR_NZ = np.nonzero(real_fnmr)
        i_NZ = np.intersect1d(i_FA_NZ, i_FR_NZ)
        ax.plot(
            real_fmr[i_NZ],
            real_fnmr[i_NZ],
            color="red",
            linewidth=1,
            label="Ground truth",
        )

    ax.set_xlabel("FMR")
    ax.set_ylabel("FNMR")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"FNMR vs FMR")
    _ = ax.legend(loc="center left")
    # TODO - is this bound ok?
    plt.gca().set_xlim(left=10**-5, right=2)
    plt.gca().set_ylim(bottom=10**-5, top=2)
    _output(dirname + os.sep + "bootstrapped_fmr_fnmr_estimate" + f"_v{version}")
