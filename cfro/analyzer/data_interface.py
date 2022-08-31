import os
import sys
import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from scipy.stats import norm

from cfro.database import PERSON_CSV, PHOTO_CSV, DETECTION_CSV, PHOTO_DIR, ANNOTATION_CSV, MATCHES_CSV
from cfro.analyzer.utils import bbox_iou_matrix

IOU_THRES = 0.4

tqdm.pandas()


class AnalysisDataInterface:

    def __init__(self, database_path: str, service_info: dict, face_id_matching="single-detection"):
        assert face_id_matching in ["single-detection", "bbox"]
        self.database_path = database_path
        self.service_info = service_info
        self.face_id_matching = face_id_matching

        self.create_face_ids_for_annotations()

    def create_face_ids_for_annotations(self):
        # add provider_id and face_id to annotated labels file
        df = pd.read_csv(os.path.join(self.database_path, f"{ANNOTATION_CSV}.csv"))
        if len(df) == 0:
            df = pd.DataFrame(columns=["bbox_id", "match_type", "provider_id", "face_id"])
        else:
            df.set_index("bbox_id", inplace=True)
            detections = self.load_all_detections()
            detections.set_index("bbox_id", inplace=True)
            df["provider_id"] = detections["provider_id"]
            df["face_id"] = detections["face_id"]
            df.reset_index(inplace=True)
        df.to_csv(os.path.join(self.database_path, f"{ANNOTATION_CSV}.csv"), index=False)

    def get_image_path(self, photo_id: int):
        return os.path.join(self.database_path, PHOTO_DIR, f"{photo_id:06d}.jpg")

    def get_photo_meta(self):
        df = pd.read_csv(os.path.join(self.database_path, f"{PHOTO_CSV}.csv"))
        return df[["photo_id", "query_person_id"]]

    def get_person_meta(self, add_demo_col=True):
        df = pd.read_csv(os.path.join(self.database_path, f"{PERSON_CSV}.csv"))
        if add_demo_col:
            df["demo"] = df["race"] + " " + df["gender"]
        return df

    def load_all_matches(self, join_detection_info: bool = False, drop_unmatched=False,
                         normalize: bool = False, verbose: bool = False) -> pd.DataFrame:
        """
        drop_unmatched: drop photos that couldn't be matched among all services
        drop_duplicates: drop photos that did not pass the image duplicate check
        normalize: normalize confidence values to [0, 1] using mode estimation
        """

        matches = pd.read_csv(os.path.join(self.database_path, f"{MATCHES_CSV}.csv"))

        # mute print statements within the following block
        _original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        detections = self.load_all_detections(format="long",
                                              drop_unmatched=drop_unmatched)
        sys.stdout = _original_stdout
        matches = matches.join(detections.set_index("bbox_id")["provider_id"], on="bbox0_id")

        matches.dropna(subset=["provider_id"], inplace=True)
        matches["provider_id"] = matches["provider_id"].astype(int)

        if join_detection_info:
            detections.set_index("bbox_id", inplace=True)
            matches.set_index("bbox0_id", inplace=True)
            matches["person0_id"] = detections["query_person_id"].astype("Int64")
            matches["face0_id"] = detections["face_id"]
            matches["photo0_id"] = detections["photo_id"].astype("Int64")
            matches.reset_index(inplace=True)
            matches.set_index("bbox1_id", inplace=True)
            matches["person1_id"] = detections["query_person_id"].astype("Int64")
            matches["face1_id"] = detections["face_id"]
            matches["photo1_id"] = detections["photo_id"].astype("Int64")
            matches.reset_index(inplace=True)

        if normalize:
            print("Loading matches, normalizing min/max with mode estimation")
            matches["conf_minmode"] = np.nan
            matches["conf_maxmode"] = np.nan
            for provider_id in np.unique(matches["provider_id"]):

                if provider_id == 4:  # TODO quickfix for Amazon Rekognition, to be solved
                    min_mode, max_mode = 0.0, 1.0
                    print(f"Provider: {provider_id}, min mode: {min_mode:.2f}, max mode: {max_mode:.2f}")
                    matches.loc[matches["provider_id"] == provider_id, "conf_minmode"] = min_mode
                    matches.loc[matches["provider_id"] == provider_id, "conf_maxmode"] = max_mode
                else:
                    data = matches.loc[matches["provider_id"] == provider_id, "confidence"].values.reshape(-1, 1)
                    gmm = GaussianMixture(n_components=2, random_state=42)
                    gmm.fit(data)
                    means = gmm.means_.flatten()
                    means.sort()
                    min_mode, max_mode = means
                    print(f"Provider: {provider_id}, min mode: {min_mode:.2f}, max mode: {max_mode:.2f}")
                    matches.loc[matches["provider_id"] == provider_id, "conf_minmode"] = min_mode
                    matches.loc[matches["provider_id"] == provider_id, "conf_maxmode"] = max_mode

                    if verbose:
                        covariances = gmm.covariances_.flatten()
                        weights = gmm.weights_.flatten()

                        # Display results
                        x = np.linspace(min(data), max(data), 1000).reshape(-1, 1)

                        plt.hist(data, bins=1000, density=True, alpha=0.5)
                        plt.plot(x, weights[0] * norm.pdf(x, means[0], np.sqrt(covariances[0])), label='Gaussian 1')
                        plt.plot(x, weights[1] * norm.pdf(x, means[1], np.sqrt(covariances[1])), label='Gaussian 2')
                        plt.xlabel('Data Value')
                        plt.ylabel('Probability Density')
                        plt.legend()
                        plt.title(f'GMM Fit to Data, provider id: {provider_id}')
                        plt.show()

            matches["confidence"] = (matches["confidence"] - matches["conf_minmode"]) / (
                    matches["conf_maxmode"] - matches["conf_minmode"])
            matches["confidence"].clip(lower=0.0, upper=1.0, inplace=True)

            matches.drop(columns=["conf_minmode", "conf_maxmode"], inplace=True)

        return matches

    @staticmethod
    def _detections_long_to_wide(df: pd.DataFrame, add_bbox_coords_to_wide: bool = False) -> pd.DataFrame:
        if add_bbox_coords_to_wide:

            def merge_bbox_coords(rows):
                coord_lists = rows.str.split(",")
                x1 = min([c[0] for c in coord_lists])
                y1 = min([c[1] for c in coord_lists])
                x2 = max([c[2] for c in coord_lists])
                y2 = max([c[3] for c in coord_lists])
                return f"{x1},{y1},{x2},{y2}"

            joined_bboxes = df.groupby("face_id")["bbox_coords"].apply(merge_bbox_coords)
        else:
            joined_bboxes = None

        df.drop(columns=["bbox_coords"], inplace=True)

        # put aside those columns that are not pivoted
        df_untouched_cols = df.drop(columns=["bbox_id"]).set_index("face_id").drop(columns=["provider_id"])
        df_untouched_cols = df_untouched_cols.reset_index().drop_duplicates().set_index("face_id")

        # pivot the face_id subtable so that provider_id values become columns
        df = df.reset_index()
        df = df.dropna(subset=["face_id"])
        face_ids_wide = df.pivot(index="face_id", columns="provider_id", values="bbox_id")
        face_ids_wide = face_ids_wide.add_prefix("provider_").add_suffix("_bbox_id")

        # join the two sub-dfs and return
        df = df_untouched_cols.join(face_ids_wide)

        if add_bbox_coords_to_wide:
            df = df.join(joined_bboxes)
        return df

    def load_all_detections(self,
                            format: str = "long",
                            drop_unmatched: bool = True,
                            add_bbox_coords_to_wide: bool = False
                            ) -> pd.DataFrame:

        assert format in ["long", "wide"]

        df = pd.read_csv(os.path.join(self.database_path, f"{DETECTION_CSV}.csv"))
        df = df[["bbox_id", "photo_id", "bbox_coords", "provider_id"]]

        # add person_id column
        photo_meta = self.get_photo_meta().set_index("photo_id")
        df = df.join(photo_meta, on="photo_id")

        df = self._create_face_id(df, self.face_id_matching)

        df = df.sort_values("bbox_id")

        if drop_unmatched:
            df = df.dropna(subset=["face_id"])

        if format == "wide":
            df = self._detections_long_to_wide(df, add_bbox_coords_to_wide=add_bbox_coords_to_wide)

        return df

    def _create_face_id(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Create a face_id column for a given detection df.

        method: one of the following:
            1) 'single-detection': match bboxes where all services found exactly one face in an image.
            Does not require bounding box coordinates.

            2) 'bbox': Match by overlapping bounding boxes. Expects 'bbox_coords' column in the input df.

        """

        if method == "single-detection":
            print("Face_id matching method: single detection rule.")
            df = self._create_face_id_by_single_detection(df)
        elif method == "bbox":
            print(f"Face_id matching method: overlapping bounding boxes (IOU={IOU_THRES}).")
            df = self._create_face_id_by_bbox(df)
        else:
            raise ValueError(f"Unknown method {method}")

        n_bboxes_in = len(df)
        n_bboxes_assigned = len(df.dropna(subset="face_id"))
        n_face_ids = len(df.dropna(subset="face_id")["face_id"].unique())
        print(
            f"Successfully assigned {n_bboxes_assigned} out of {n_bboxes_in} bboxes "
            f"({n_bboxes_assigned / n_bboxes_in * 100:.2f}%) to {n_face_ids} unique face_ids."
        )
        return df

    def _create_face_id_by_single_detection(self, df: pd.DataFrame):
        n_providers = len(self.service_info)

        # count the number of faces found, per person, photo, and provider
        num_faces_found = df.groupby(["photo_id", "provider_id"])["bbox_id"].count()

        # groupy new table by person_id and photo_id
        nff_grouped = num_faces_found.groupby("photo_id")

        # create boolean series that says if all providers detected exactly one face in a photo
        all_found_1_face = ((nff_grouped.count() == n_providers) & (nff_grouped.sum() == n_providers))
        all_found_1_face.name = "found_1_face"
        df = df.join(all_found_1_face, on="photo_id")

        # create face_id for those entries where the boolean series is true
        df = df.set_index("bbox_id")
        df["face_id"] = (df["query_person_id"].astype(str) + "-" + df["photo_id"].astype(str) + "-0").where(
            df["found_1_face"])
        df["face_id"] = df["face_id"].astype("string")  # string data type uses <NA> instead of NaN.
        df = df.reset_index()
        df = df.drop(columns=["found_1_face"])
        return df

    def _create_face_id_by_bbox(self, df: pd.DataFrame):
        n_providers = len(self.service_info)

        bbox_coords = df["bbox_coords"].str.split(",", expand=True).astype(int)
        df["bbox_x1"] = bbox_coords[0]
        df["bbox_y1"] = bbox_coords[1]
        df["bbox_x2"] = bbox_coords[2]
        df["bbox_y2"] = bbox_coords[3]

        def assign_face_id_by_bbox_overlap(subset, iou_thres=IOU_THRES):
            person_id = subset["query_person_id"].unique()
            assert len(person_id) == 1
            person_id = person_id[0]
            photo_id = subset["photo_id"].unique()
            assert len(photo_id) == 1
            photo_id = photo_id[0]

            bbox_vals = subset[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values
            iou_matrix = bbox_iou_matrix(bbox_vals)
            matches = iou_matrix >= iou_thres
            np.fill_diagonal(matches, True)
            unique_rows = np.unique(matches, axis=0)
            ids = np.full(len(iou_matrix), np.nan, dtype=object)
            i = 0
            for u_row in unique_rows:
                rows_match = (matches == u_row).all(axis=1)
                count = rows_match.sum()
                unique_providers = len(subset[rows_match]["provider_id"].unique())
                if count == n_providers and unique_providers == n_providers:
                    ids[(matches == u_row).all(axis=1)] = f"{person_id}-{photo_id}-{i}"
                    i += 1
            subset.loc[:, "face_id"] = ids

            return subset

        df = df.groupby(["query_person_id", "photo_id"]).progress_apply(assign_face_id_by_bbox_overlap)
        df = df.reset_index(drop=True)
        df = df.drop(["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"], axis=1)

        return df

    def load_annotations(self):
        # check groundtruth consistency
        filename = os.path.join(self.database_path, f"{ANNOTATION_CSV}.csv")

        if not os.path.isfile(filename):
            return pd.DataFrame(columns=["bbox_id", "match_type", "provider_id", "face_id"])

        gt = pd.read_csv(filename)
        gt = gt.drop_duplicates(subset=["face_id", "match_type"])
        gt[gt["face_id"].duplicated(keep=False)].sort_values("face_id")

        gt = gt[["face_id", "match_type"]].drop_duplicates().set_index("face_id")  # only use 2 relevant cols
        return gt
