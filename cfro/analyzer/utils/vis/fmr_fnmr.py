from typing import List, Dict, Tuple
from copy import deepcopy
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cfro.analyzer.utils.fmr_fnmr import FNMR_vs_FMR
from cfro.analyzer.utils.vis.histogram import my_histogram

import warnings
warnings.filterwarnings("ignore")

THICK_LINEWIDTH = 5
PRED_LINESTYLE = "-"
ALPHA = .35


def fmr_fnmr_joint(data: Dict, service_meta: Dict, savepath: str, error_range=(0.0, 1.0)):

    service_id2name = {value['id']: value['name'] for key, value in service_meta.items()}

    plt.figure()

    for i, provider_id in enumerate(data.keys()):
        prov_data = data[provider_id]

        if prov_data.get("YY") is not None:
            FNMR, FMR, conf = FNMR_vs_FMR(prov_data.get("C"), prov_data.get("YY"), error_range=error_range)
        else:
            FNMR, FMR, conf = None, None, None
        FNMR1, FMR1, conf1 = FNMR_vs_FMR(prov_data.get("C_est"), prov_data.get("YY_est"), error_range=error_range)

        if FMR is not None:
            plt.plot(FMR, FNMR, lw=THICK_LINEWIDTH, label=f'ground truth, provider={provider_id}', c=f'C{i}',
                    alpha=ALPHA)
        plt.plot(FMR1, FNMR1, label=f'estimated, provider={provider_id}', c=f'C{i}', linestyle=PRED_LINESTYLE)

        plt.xlabel('FMR')
        plt.ylabel('FNMR')
        plt.xscale('log')
        plt.yscale('log')

    legend_handles = [
        Line2D([0], [0], color=f"C{i}", linewidth=2, label=service_id2name[sid]) for i, sid in enumerate(data.keys())
    ]
    legend_handles.append(Line2D([0], [0], color="w"))
    legend_handles.append(
        Line2D([0], [0], color=f"black", linewidth=THICK_LINEWIDTH, label="Manual Labels", alpha=ALPHA))
    legend_handles.append(Line2D([0], [0], color=f"black", linestyle=PRED_LINESTYLE, label="Estimated Labels"))

    plt.legend(handles=legend_handles, loc='best', ncol=1, frameon=True, fontsize="small")
    if savepath:
        plt.savefig(savepath)
    plt.close()


def fmr_fnmr_stacked(data: Dict[int, Dict], service_meta: dict, suptitle=None, share_axis=False,
                     logscale_hist=False, toprow_legend_only=True, error_range=(0.0, 1.0)):
    "subplots stacked by provider"
    service_id2name = {value['id']: value['name'] for key, value in service_meta.items()}

    figsize = (9, len(data) * (8 / 3))
    fig, ax = plt.subplots(nrows=len(data), ncols=3, figsize=figsize, sharey=share_axis,
                           sharex=share_axis)

    for i, (provider_id, provider_results) in enumerate(data.items()):
        C = provider_results['C']
        C_est = provider_results.get('C_est', None)
        YY = provider_results['YY']
        YY_est = provider_results.get('YY_est', None)

        fmr_fnmr_subplots(C, YY, YY_est, C_est=C_est, provider_id=None, fig_ax=(fig, ax[i, :]),
                          logscale_hist=logscale_hist, error_range=error_range)
        ax[i, 0].set_ylabel(f"{service_id2name[provider_id]}\nFrequency")

    if toprow_legend_only:
        for i in range(len(data)):
            for j in range(3):
                _ax = ax[i, j]
                if i == 0:
                    _ax.legend(fontsize=8)
                else:
                    _ax.legend_.remove()

    fig.tight_layout()

    if suptitle:
        fig.suptitle(suptitle)
    return fig


def fmr_fnmr_subplots(C, YY, YY_est, C_est=None, provider_id=None, savepath=None, title_suffix='', show=False,
                      fig_ax=None, logscale_hist=True, error_range=(0.0, 1.0)):
    """3 subplots"""
    if fig_ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9, 3.3), constrained_layout=True)
    else:
        fig, ax = fig_ax
        show = False
        savepath = None

    if C_est is None:
        print("WARNING C_est not set. use C instead.")
        C_est = C

    C0 = C[np.where(YY == 0)]
    C1 = C[np.where(YY == 1)]
    C0_est = C_est[np.where(YY_est == 0)]
    C1_est = C_est[np.where(YY_est == 1)]

    if len(C0) > 0:
        b, n = my_histogram(C0)
        ax[0].plot(n, b, lw=THICK_LINEWIDTH, label='P(C|Y=0)', c="C0", alpha=ALPHA)
    if len(C1) > 0:
        b, n = my_histogram(C1)
        ax[0].plot(n, b, lw=THICK_LINEWIDTH, label='P(C|Y=1)', c="C3", alpha=ALPHA)
    b, n = my_histogram(C0_est)
    ax[0].plot(n, b, label='P(C|Y_est=0)', c="C0", linestyle=PRED_LINESTYLE)
    b, n = my_histogram(C1_est)
    ax[0].plot(n, b, label='P(C|Y_est=1)', c="C3", linestyle=PRED_LINESTYLE)
    ax[0].legend()
    ax[0].set_xlabel('Confidence')
    ax[0].set_ylabel('Frequency')
    #ax[0].xaxis.set_tick_params(labelbottom=True)
    if logscale_hist:
        ax[0].set_yscale("log")

    if len(C) > 0 and len(YY) > 0:
        FNMR, FMR, conf = FNMR_vs_FMR(C, YY, error_range=error_range)
    else:
        FNMR, FMR, conf = None, None, None
    FNMR1, FMR1, conf1 = FNMR_vs_FMR(C_est, YY_est, error_range=error_range)

    if FNMR is not None and FMR is not None:
        ax[1].plot(conf, FMR, lw=THICK_LINEWIDTH, label='FMR', c="C2", alpha=ALPHA)
        ax[1].plot(conf, FNMR, lw=THICK_LINEWIDTH, label='FNMR', c="C1", alpha=ALPHA)
    ax[1].plot(conf1, FMR1, label='FMR est.', c="C2", linestyle=PRED_LINESTYLE)
    ax[1].plot(conf1, FNMR1, label='FNMR est.', c="C1", linestyle=PRED_LINESTYLE)
    ax[1].set_xlabel('Confidence Threshold')
    ax[1].set_ylabel('Error Rates')
    ax[1].legend()
    #ax[1].xaxis.set_tick_params(labelbottom=True)

    if FNMR is not None and FMR is not None:
        ax[2].plot(FMR, FNMR, lw=THICK_LINEWIDTH, label='Manual Labels', c="C4", alpha=ALPHA)
    ax[2].plot(FMR1, FNMR1, label='Estimated Labels', c="C4", linestyle=PRED_LINESTYLE)
    ax[2].set_xlabel('FMR')
    ax[2].set_ylabel('FNMR')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].legend()

    if provider_id is not None:
        fig.suptitle('Provider ID: ' + str(provider_id) + title_suffix)
    if savepath:
        plt.savefig(savepath)
    if show:
        plt.show()
    return fig
