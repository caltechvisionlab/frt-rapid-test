""" Plots for accuarcy and bias"""

import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from cfro.analyzer.utils.vis.fmr_fnmr import fmr_fnmr_stacked, fmr_fnmr_joint, FNMR_vs_FMR
from cfro.analyzer.data_interface import AnalysisDataInterface
from cfro.analyzer.utils.affinity_methods import calculate_rank1_approximation

COLORBLIND_TOL = ["#332288", "#117733", "#44AA99", "#88CCEE", "#DDCC77", "#CC6677", "#AA4499", "#882255"]
# https://davidmathlogic.com/colorblind/#%23332288-%23117733-%2344AA99-%2388CCEE-%23DDCC77-%23CC6677-%23AA4499-%23882255

# adapted to default matplotlib color cycle
COLORBLIND_WONG_CUSTOM = ["#56B4E9", "#D55E00", "#009E73", "#CC79A7", "#000000"]
# COLORBLIND_CUSTOM = ["#1E88E5", "#FFC107", "#004D40", "#D81B60", "#000000"]
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=COLORBLIND_WONG_CUSTOM)


def fmr_fnmr(
        estimation_results: list,
        data_interface: AnalysisDataInterface,
        service_meta: dict,
        savepath: str = None,
        force_same_image_set=False,
        logscale_hist=False,
        impostor_definition="across_query",
        demographics_filter=None,
        test_perfect_alignment=False,
        error_range=(0.0, 1.0)
):
    assert impostor_definition in ["both", "within_query", "across_query"]
    person_list = estimation_results
    all_provider_ids = sorted(list(person_list[0].provider_results.keys()))

    data = dict()
    for provider_id in all_provider_ids:
        provider_results = [p.provider_results[provider_id] for p in person_list if p.include()]

        Y_est = pd.DataFrame({
            "face_id": np.concatenate([pr.ids for pr in provider_results]),
            "Y_est": np.concatenate([pr.Y_est for pr in provider_results]),
            f"z_provider_{provider_id}": np.concatenate(
                [calculate_rank1_approximation(pr.C)[1] for pr in provider_results])
        }).set_index("face_id")

        matches = data_interface.load_all_matches(join_detection_info=True, drop_unmatched=True)
        person_meta = data_interface.get_person_meta().set_index("person_id")
        gt = data_interface.load_annotations()

        Y_est.drop(columns=f"z_provider_{provider_id}", inplace=True)

        df = matches[["face0_id", "person0_id", "face1_id", "person1_id", "confidence", "provider_id"]].copy()

        df.set_index("face0_id", inplace=True)
        df["Y0"] = gt.astype("Int16")
        df["Y_est0"] = Y_est.astype("Int16")
        df.reset_index(inplace=True)
        df.set_index("face1_id", inplace=True)
        df["Y1"] = gt.astype("Int16")
        df["Y_est1"] = Y_est.astype("Int16")
        df.reset_index(inplace=True)

        if demographics_filter:
            df.set_index("person0_id", inplace=True)
            df["person0_demo"] = person_meta["demo"]
            df.reset_index(inplace=True)
            df.set_index("person1_id", inplace=True)
            df["person1_demo"] = person_meta["demo"]
            df.reset_index(inplace=True)
            df = df[(df["person0_demo"] == demographics_filter) & (df["person1_demo"] == demographics_filter)]

        if force_same_image_set:
            df.dropna(subset=["Y0", "Y1", "Y_est0", "Y_est1"], inplace=True)

        if test_perfect_alignment:
            df.loc[df['Y0'].notna(), 'Y_est0'] = df.loc[df['Y_est0'].notna(), 'Y0']
            df.loc[df['Y1'].notna(), 'Y_est1'] = df.loc[df['Y_est1'].notna(), 'Y1']

        has_same_query_id = df["person0_id"] == df["person1_id"]
        has_different_query_id = df["person0_id"] != df["person1_id"]
        provider_filter = df["provider_id"] == provider_id
        genuine_filter_est = (df["Y_est0"] == 1) & (df["Y_est1"] == 1) & has_same_query_id

        impostor_filter_est_across = (df["Y_est0"] == 1) & (df["Y_est1"] == 1) & has_different_query_id
        impostor_filter_est_within = (df["Y_est0"] != df["Y_est1"]) & has_same_query_id
        if impostor_definition == "across_query":
            impostor_filter_est = impostor_filter_est_across
        elif impostor_definition == "within_query":
            impostor_filter_est = impostor_filter_est_within
        else:
            impostor_filter_est = impostor_filter_est_across | impostor_filter_est_within

        genuine_filter_anno = (df["Y0"] == 1) & (df["Y1"] == 1) & has_same_query_id

        impostor_filter_anno_across = (df["Y0"] == 1) & (df["Y1"] == 1) & has_different_query_id
        impostor_filter_anno_within = (df["Y0"] != df["Y1"]) & has_same_query_id
        if impostor_definition == "across_query":
            impostor_filter_anno = impostor_filter_anno_across
        elif impostor_definition == "within_query":
            impostor_filter_anno = impostor_filter_anno_within
        else:
            impostor_filter_anno = impostor_filter_anno_across | impostor_filter_anno_within

        # estimate
        genuine_subset_est = df[provider_filter & genuine_filter_est].copy()
        genuine_subset_est.dropna(subset=["confidence", "Y_est0", "Y_est1"], inplace=True)
        genuine_C_est = genuine_subset_est["confidence"].values
        genuine_YY_est = np.ones_like(genuine_C_est)

        impostor_subset_est = df[provider_filter & impostor_filter_est].copy()
        impostor_subset_est.dropna(subset=["confidence", "Y_est0", "Y_est1"], inplace=True)
        impostor_C_est = impostor_subset_est["confidence"].values
        impostor_YY_est = np.zeros_like(impostor_C_est)

        # annotated
        genuine_subset_anno = df[provider_filter & genuine_filter_anno].copy()
        genuine_subset_anno.dropna(subset=["confidence", "Y0", "Y1"], inplace=True)
        genuine_C_anno = genuine_subset_anno["confidence"].values
        genuine_YY_anno = np.ones_like(genuine_C_anno)

        impostor_subset_anno = df[provider_filter & impostor_filter_anno].copy()
        impostor_subset_anno.dropna(subset=["confidence", "Y0", "Y1"], inplace=True)
        impostor_C_anno = impostor_subset_anno["confidence"].values
        impostor_YY_anno = np.zeros_like(impostor_C_anno)

        data[provider_id] = {
            "C": np.concatenate([genuine_C_anno, impostor_C_anno]),
            "C_est": np.concatenate([genuine_C_est, impostor_C_est]),
            "YY": np.concatenate([genuine_YY_anno, impostor_YY_anno]),
            "YY_est": np.concatenate([genuine_YY_est, impostor_YY_est]),
        }

    fmr_fnmr_stacked(data, service_meta, suptitle="", logscale_hist=logscale_hist, error_range=error_range)
    if savepath:
        plt.savefig(savepath)
    plt.close()

    return data


def fmr_fnmr_demographics(estimation_results: list,
                          data_interface: AnalysisDataInterface,
                          service_info: dict,
                          savepath=None,
                          error_range=(0.0, 1.0),
                          **kwargs):
    service_id2name = {value['id']: value['name'] for key, value in service_info.items()}
    provider_ids = sorted([s["id"] for s in service_info.values()])
    n_providers = len(provider_ids)

    fig, axs = plt.subplots(nrows=1, ncols=n_providers, figsize=(n_providers * (16 / 3), 6), constrained_layout=True,
                            sharex=False, sharey=False)

    person_meta = data_interface.get_person_meta()
    all_demographics = np.unique(person_meta["demo"])

    legend_handles = list()

    for i_demo, demo in enumerate(all_demographics):
        demo_color = f"C{i_demo}"
        legend_handles.append(Line2D([0], [0], lw=3, color=demo_color, label=demo))

        data_all = fmr_fnmr(estimation_results, data_interface, service_info, demographics_filter=demo, **kwargs)

        for i_prov, provider_id in enumerate(provider_ids):
            data = data_all[provider_id]
            C = data['C']
            C_est = data.get('C_est', None)
            YY = data['YY']
            YY_est = data['YY_est']
            if len(YY) > 0:
                FNMR, FMR, conf = FNMR_vs_FMR(C, YY, error_range=error_range)
            else:
                FNMR, FMR, conf = None, None, None
            FNMR1, FMR1, conf1 = FNMR_vs_FMR(C_est, YY_est, error_range=error_range)

            if FNMR is not None:
                axs[i_prov].plot(FMR, FNMR, lw=2, label=f'ground truth, demo={demo}', c=demo_color)

            axs[i_prov].plot(FMR1, FNMR1, lw=2, label=f'estimated, demo={demo}', c=demo_color,
                             linestyle='dotted')

    for ax, prov_id in zip(axs, provider_ids):
        ax.set_xlabel('FMR')
        ax.set_ylabel('FNMR')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(service_id2name[prov_id])

    legend_handles.append(Line2D([0], [0], color="w"))
    legend_handles.append(Line2D([0], [0], color="black", linewidth=2, label="Manual Labels"))
    legend_handles.append(Line2D([0], [0], color="black", linestyle="dotted", label="Estimated Labels"))
    plt.legend(handles=legend_handles, loc='best', ncol=1, frameon=True)
    fig.suptitle("FMR-FNMR per demographic group")

    if savepath:
        plt.savefig(savepath)

    plt.close()


def main(estimation_results: list, data_interface: AnalysisDataInterface, service_info: dict, out_dir: str, error_range=(0.0, 1.0)):
    data = fmr_fnmr(estimation_results, data_interface, service_info,
                    savepath=os.path.join(out_dir, "plots/fmr_fnmr.pdf"),
                    force_same_image_set=False, logscale_hist=False,
                    impostor_definition="across_query", test_perfect_alignment=False, error_range=error_range)

    fmr_fnmr_joint(data, service_info,
                   savepath=os.path.join(out_dir, "plots/fmr_fnmr_joint.pdf"),
                   error_range=error_range)

    fmr_fnmr_demographics(estimation_results, data_interface, service_info,
                          savepath=os.path.join(out_dir, "plots/fmr_fnmr_demo.pdf"),
                          error_range=error_range
                          )

