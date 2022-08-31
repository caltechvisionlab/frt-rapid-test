"""TODO this whole module could need some refactoring for maintainability"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages

from cfro.analyzer.utils.affinity_methods import calculate_confidence_matrix
from cfro.analyzer.utils.vis.matrix import display_dressed_image
from cfro.analyzer.utils.spectral_factorization import SpectralFactorizationClustering

from cfro.analyzer.data_interface import AnalysisDataInterface

MIN_FACES = 8
MIN_POS_FACES = 5
STANDARD_FIG_SIZE_1row = (10, 4)
STANDARD_FIG_SIZE_2row = (9, 5)
DPI = 100


class ProviderResults:

    def __init__(self, confidence_matrix: pd.DataFrame, groundtruth, cluster_labels, evs=None, provider_id=None,
                 person_id=None,
                 use_majority_vote_gt_estimate=True):
        self.ids = confidence_matrix.index.values
        self.C = confidence_matrix.values
        self.Y = groundtruth
        self.cluster_labels = cluster_labels
        self.eigenvectors = evs
        self.majority_cluster_labels = None
        self.majority_vote = use_majority_vote_gt_estimate

        self.n_clusters = cluster_labels.max() + 1
        self.n_faces_total = len(self.cluster_labels)
        self.n_faces_positive = (self.cluster_labels > -1).sum()  # count non-negative entries in numpy array

        self.person_id = person_id
        self.provider_id = provider_id

        self.include = None

    def set_majority_labels(self, maj_labels):
        assert self.majority_cluster_labels is None
        self.majority_cluster_labels = maj_labels

    @property
    def Y_est(self):
        if self.majority_vote:
            if self.majority_cluster_labels is None:
                raise Exception("Majority Labels not set!")
            return self.majority_cluster_labels
        else:
            return self.cluster_labels + 1


class Person:

    def __init__(self, person_id: int):
        self.person_id = person_id
        self.provider_results = dict()

    def add_provider_results(self, provider_id: int, prov_results: ProviderResults):
        assert provider_id not in self.provider_results.keys()
        self.provider_results[provider_id] = prov_results

    def include(self):
        cluster_criterion = all([pr.n_clusters == 1 for pr in self.provider_results.values()])
        n_faces_criterion = all([pr.n_faces_total >= MIN_FACES for pr in self.provider_results.values()])
        n_pos_faces_criterion = all([pr.n_faces_positive >= MIN_POS_FACES for pr in self.provider_results.values()])

        _ids = [pr.ids for pr in self.provider_results.values()]
        all_same_ids = all(np.array_equal(arr, _ids[0]) for arr in _ids)

        return cluster_criterion and n_faces_criterion and n_pos_faces_criterion and all_same_ids

    @property
    def all_estimates(self):
        return np.stack([pr.cluster_labels for pr in self.provider_results.values()]) + 1

    def majority_gt_estimate(self):
        return (self.all_estimates.sum(axis=0) > len(self.provider_results) / 2).astype(int)

    def set_majority_gt_estimate(self):
        maj_est = self.majority_gt_estimate()
        for prov_result in self.provider_results.values():
            prov_result.set_majority_labels(maj_est)

    def set_include(self):
        for prov_result in self.provider_results.values():
            prov_result.include = self.include()


class ConfidenceSampler:

    def __init__(self, data_interface: AnalysisDataInterface):
        self.matches = data_interface.load_all_matches(join_detection_info=True,
                                                       drop_unmatched=True,
                                                       normalize=True)
        self.matches_unnorm = data_interface.load_all_matches(join_detection_info=True, drop_unmatched=True,
                                                              normalize=False)  # TODO refactor, make cleaner

        self.person_meta = data_interface.get_person_meta().set_index("person_id")
        self.gt = dict(data_interface.load_annotations()["match_type"])

    def get_gt(self, idx, none_val=-1):
        gt = self.gt.get(idx)
        if gt is None:
            return none_val
        return gt

    @property
    def person_ids(self):
        return self.person_meta.index.values

    @property
    def provider_ids(self):
        return self.matches["provider_id"].unique()

    def get_confidence_matrix(self, provider_id: int, person_id: int, reindex=None, normalized=True):
        if normalized:
            matches = self.matches
        else:
            matches = self.matches_unnorm  # TODO refactor, cleaner

        matches = matches.query(
            f'person0_id=={person_id} and person1_id=={person_id} and provider_id=={provider_id}')

        matrix = calculate_confidence_matrix(matches,
                                             bbox0_col="face0_id",
                                             bbox1_col="face1_id",
                                             conf_col="confidence",
                                             return_full_matrix=True)
        if reindex:
            matrix = matrix.loc[reindex, reindex]

        return matrix


def main(data_interface, service_info: dict, out_dir: str,
         majority_vote_gt=True, return_annotations=True,
         ):

    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    person_list = list()
    n_providers = len(service_info)
    outPDF = PdfPages(os.path.join(out_dir, 'plots/gt_est_spectral_factorization.pdf'))

    sampler = ConfidenceSampler(data_interface)
    gt = sampler.gt
    for person_id in tqdm(sampler.person_ids, desc="Estimating labels"):
        plotted_anything = False
        person = Person(person_id)
        fig, ax = plt.subplots(1, n_providers, figsize=STANDARD_FIG_SIZE_1row, squeeze=False)
        fig.suptitle(f"Spectral Factorization - Person: {person_id}\n"
                     f"GT: \u2713=true, \u2717=false, ?=unknown\n")

        ### estimate common indexes  # TODO cleaner
        indexes = [list(sampler.get_confidence_matrix(provider_id, person_id).index.values) for provider_id in
                   sampler.provider_ids]
        subset_indexes = sorted(list(set(indexes[0]).intersection(*indexes[1:])))

        for prov_nr, provider_id in enumerate(sampler.provider_ids):
            matrix = sampler.get_confidence_matrix(provider_id, person_id, reindex=subset_indexes, normalized=True)
            matrix_unnorm = sampler.get_confidence_matrix(provider_id, person_id, reindex=subset_indexes,
                                                          normalized=False)  # TODO refactor, cleaner
            if matrix.size == 0:
                continue
            else:
                plotted_anything = True
            matrix_cluster_ids, eigenvectors = SpectralFactorizationClustering(matrix.values).cluster(return_evs=True)

            gt_array = np.array([sampler.get_gt(pid) for pid in matrix.index.values])
            if not return_annotations:
                gt_array = np.full_like(gt_array, -1)

            prov_results = ProviderResults(matrix_unnorm, gt_array, matrix_cluster_ids, evs=eigenvectors,
                                           person_id=person_id,
                                           provider_id=provider_id, use_majority_vote_gt_estimate=majority_vote_gt)
            person.add_provider_results(provider_id, prov_results)
            display_dressed_image(matrix, ax[0][prov_nr], ticklabel_colors=matrix_cluster_ids,
                                  title=f"provider={provider_id}, n_clusters={matrix_cluster_ids.max() + 1}",
                                  gt_annotation=gt
                                  )

        fig.text(0.5, 0.825, "include" if person.include() else "exclude", ha='center', fontsize=12,
                 color='green' if person.include() else "red")

        if plotted_anything:
            if majority_vote_gt:
                person.set_majority_gt_estimate()
            person.set_include()

            plt.tight_layout(pad=1.5)
            outPDF.savefig(fig, dpi=DPI)
            person_list.append(person)
        plt.close(fig)

    # Add stats as text
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    all_estimates = np.concatenate([p.all_estimates for p in person_list], axis=1)
    all_agree = np.all(all_estimates == all_estimates[0, :], axis=0)
    info_text = f"""
    Query Person ID filter criteria:\n
    - Exactly 1 cluster found for each provider\n
    - Minimum number of faces: {MIN_FACES}\n
    - Minimum number of positive faces: {MIN_POS_FACES}\n
    N persons included: {len([p for p in person_list if p.include()])}/{len(person_list)}\n
    \n
    GT estimation method: {'majority vote' if majority_vote_gt else 'per provider'}\n
    Agreement: {all_agree.sum() / len(all_agree) * 100:.2f}%\n
    """
    plt.text(0.5, 0.5, info_text, ha="center", va="center")
    outPDF.savefig(fig, dpi=DPI)
    outPDF.close()

    # 2) save to CSV
    out_dfs = list()
    for person in person_list:
        for provider_id, provider_results in person.provider_results.items():
            person_id = provider_results.person_id
            face_ids = provider_results.ids
            Y_est = provider_results.Y_est if provider_results.include else np.full(len(face_ids), -1)
            _df = pd.DataFrame({
                "provider_id": provider_id,
                "person_id": person_id,
                "face_id": face_ids,
                "Y_est": Y_est,
                # TODO also join bbox_id
            })
            out_dfs.append(_df)
    out_df = pd.concat(out_dfs)
    out_df.to_csv(os.path.join(out_dir, "estimation.csv"), index=False)

    return person_list