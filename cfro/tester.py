from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import numpy as np
import itertools

from .labeler import Labeler
from .provider import gen_provider_class, provider_to_label
from .analyzer import (
    compute_results_supervised,
    compute_results_unsupervised,
    compute_dataset_statistics,
)
from .faces import FaceClusterer
from .scraper import ImageScraper


class Benchmark:
    """
    This is the top-level class for a benchmark instance (barring the user wrapper).
    It manages the providers, runs detection/comparison loops, spawns a labeling server,
    and provides functions to run analytics.
    """

    def __init__(self, database, dataset, provider_enums, credentials, benchmark_id):
        # Map from photo id -> list of `ResolvedFaces` for that photo.
        self.resolved_faces_per_photo = {}
        # Map from face id -> `ResolvedFaces` group it belongs to.
        self.face_to_resolved_faces = {}

        self.database = database
        self.dataset = dataset
        self.providers_enums = provider_enums

        self.providers = []
        for provider_enum in provider_enums:
            provider_class = gen_provider_class(provider_enum)
            assert provider_enum in credentials
            provider = provider_class(database, credentials[provider_enum])
            self.providers.append(provider)

        self.benchmark_id = benchmark_id

    def set_constants_config(self, config):
        """
        Used to set user-specified config for constants (currently face resolution).
        """
        self.config = config

    def run_providers_detect(self):
        """
        Calls `self._run_provider_detect()` for each provider.
        """
        for provider in self.providers:
            print(
                "Starting to detect faces for provider",
                provider_to_label(provider.provider_enum),
            )
            self._run_provider_detect(provider)
            self.database._flush_detections()

        if len(self.resolved_faces_per_photo) == 0:
            self._resolve_detected_faces()

    def _run_provider_detect(self, provider):
        """
        Calls `provider.detect_faces()` for each photo.

        Loads results into `provider.detected_faces` per provider.
        """
        for version in self.dataset.versions:
            print(
                f"Detecting for person {version.person.person_id}: ", end="", flush=True
            )
            for photo in version.get_photos():
                provider.detect_faces(
                    self.benchmark_id, version.person.person_id, photo
                )
                print(".", end="", flush=True)
            print()

            # Save detected faces after each id is processed, so we lose less
            # data if we have to stop the detection early or it crashes.
            self.database._flush_detections()

    def _test_face_resolution(self, photo_id, resolved_faces):
        img_path = ImageScraper.get_image_filename(
            self.database.get_photo_dir(), photo_id
        )
        img = mpimg.imread(img_path)
        fig, ax = plt.subplots()
        ax.imshow(img)
        colors = list(mcolors.XKCD_COLORS)
        for i, faces in enumerate(resolved_faces):
            for j, face in enumerate(faces.faces):
                (
                    left,
                    upper,
                    width,
                    height,
                ) = face.bounding_box.get_top_left_width_height()
                ax.add_patch(
                    patches.Rectangle(
                        (left, upper),
                        width,
                        height,
                        fill=False,
                        edgecolor=colors[i],
                        lw=2,
                    )
                )
                t = ax.annotate(f"{i}", xy=(left, upper))
                # https://stackoverflow.com/questions/23696898/adjusting-text-background-transparency
                t.set_bbox(dict(facecolor=colors[i], alpha=0.5, edgecolor=colors[i]))
        plt.show()

    def _resolve_detected_faces(self, run_test_demo=False):
        """
        For each photo, resolves the `DetectedFaces` from each provider
        into a list of `ResolvedFaces` per photo.

        Loads the results into `self.resolved_faces_per_photo`
        """
        for version in self.dataset.versions:
            for photo in version.get_photos():
                photo_id = photo.photo_id
                provider_to_faces = {}
                for provider in self.providers:
                    if len(provider.detected_faces.get(photo_id, [])) > 0:
                        provider_to_faces[
                            provider.provider_enum
                        ] = provider.detected_faces[photo_id]
                clusterer = FaceClusterer(
                    provider_to_faces,
                    self.config["INTERSECTION_OVER_UNION_IOU_THRESHOLD"],
                )
                clusters = clusterer.compute_clusters()
                self.resolved_faces_per_photo[photo_id] = clusters
                for cluster in clusters:
                    for face in cluster.faces:
                        self.face_to_resolved_faces[face.face_id] = cluster

                if run_test_demo:
                    self._test_face_resolution(
                        photo_id, self.resolved_faces_per_photo[photo_id]
                    )

    def _gen_subsampled_photos(self, subsampling_seed):
        """
        This generates a subset of all possible diff-id comparisons
        that were selected by random sampling.

        We set the random seed via RandomState so it is deterministic
        (for a fixed choice of seed).
        """
        # We want to take x% of diff-id pairs to obtain an equal number of
        # diff-id comparisons as same-id comparisons. This computes 1/x%.
        subsampling_ratios = []
        for provider in self.providers:
            subsampling_ratios.append(
                (
                    provider.get_num_diffid(
                        photo_id_has_single_face=self.photo_id_has_single_face,
                    )
                    / provider.get_num_sameid(
                        photo_id_has_single_face=self.photo_id_has_single_face,
                    )
                )
            )
        # Assuming we only use photos that have 1 face detected by all providers,
        # we do not need to take the average. This is from a previous version
        # of the code. Now, every element of `subsampling_ratios` should be identical.
        average_subsampling_ratio = sum(subsampling_ratios) / len(subsampling_ratios)

        # Considers all pairs of faces from diff people, shuffle, then subsample by
        # taking the first x% of diff-id pairs.
        subsampled_photo_pairs = set()
        rng = np.random.RandomState(subsampling_seed)
        for i, version_i in enumerate(self.dataset.versions):
            photos_i = [photo.get_photo_id() for photo in version_i.get_photos()]
            photos_i = [id for id in photos_i if self.photo_id_has_single_face(id)]

            for j, version_j in enumerate(self.dataset.versions):
                if i >= j:
                    continue
                assert i < j
                photos_j = [photo.get_photo_id() for photo in version_j.get_photos()]
                photos_j = [id for id in photos_j if self.photo_id_has_single_face(id)]

                pairs = list(itertools.product(photos_i, photos_j))
                rng.shuffle(pairs)
                truncated = round(len(pairs) / average_subsampling_ratio)
                if truncated == 0:
                    # We could raise the lower bound from 1 but this seems fine/fair.
                    truncated = 1
                pairs = pairs[:truncated]
                subsampled_photo_pairs.update(pairs)
        return subsampled_photo_pairs

    def run_providers_compare(self, subsampling_seed=None):
        """
        Calls `self._run_provider_compare()` for each provider.
        """
        if len(self.resolved_faces_per_photo) == 0:
            self._resolve_detected_faces()

        if subsampling_seed:
            subsampling_dict = self._gen_subsampled_photos(subsampling_seed)
        else:
            subsampling_dict = None

        for provider in self.providers:
            self._run_provider_compare(provider, subsampling_dict=subsampling_dict)

            self.database._flush_results()

    def allow_pair_with_subsampling(self, subsampling_dict, face1, face2):
        if subsampling_dict is None:
            return True
        return (face1.photo_id, face2.photo_id) in subsampling_dict or (
            face2.photo_id,
            face1.photo_id,
        ) in subsampling_dict

    def _run_provider_compare(self, provider, subsampling_dict=None):
        """
        Calls `provider.compare_faces()` for pairs of faces.

        If `subsampling_seed` is specified, then sample over photos
        only for the diff-id comparisons (to roughly ensure there is
        an equal number of same-id and diff-id).
        Otherwise, use all images.
        """
        diff = 0
        same = 0

        total_faces = provider.get_num_detected_faces()
        for idx, face1 in enumerate(provider.get_all_detected_faces()):
            self.database._flush_results()

            print(f"Running comparisons for face {idx+1} / {total_faces}: ", end="")
            if not self.photo_id_has_single_face(face1.photo_id):
                print("skipped since source photo has 2+ faces")
                continue
            for face2 in provider.get_all_detected_faces():
                if face1.face_id == face2.face_id:
                    # No point in comparing the exact same face
                    # because any facial comparison algo could just
                    # test if two images are the same.
                    continue
                if not self.photo_id_has_single_face(face2.photo_id):
                    continue

                # If subsampling is specified, still consider all same-person
                # pairs (this is why we subsample).
                # TODO - if there are budget constraints, sampling same-person pairs
                # is a possible solution to reduce the number of API calls.
                # This would be a non-trivial code update, though.
                if face1.person_id == face2.person_id:
                    same += 1
                    provider.compare_faces(face1, face2)

                elif self.allow_pair_with_subsampling(subsampling_dict, face1, face2):
                    diff += 1
                    provider.compare_faces(face1, face2)

                print(".", end="", flush=True)

            print()

    def photo_id_has_single_face(self, photo_id, require_all_providers_detect=True):
        resolved_faces = self.resolved_faces_per_photo[photo_id]
        if len(resolved_faces) != 1:
            return False

        if require_all_providers_detect:
            # Make sure the number of providers that detect this face
            # is the same as the total number of providers.
            return len(resolved_faces[0].faces) == len(self.providers)
        return True

    def photo_has_single_face(self, photo, require_all_providers_detect=True):
        return self.photo_id_has_single_face(
            photo.get_photo_id(),
            require_all_providers_detect=require_all_providers_detect,
        )

    def label_detected_faces(self, bypass=False, ngrok=False, port=5000):
        if bypass:
            return

        if len(self.resolved_faces_per_photo) == 0:
            self._resolve_detected_faces()

        photos_accounted_for = set()

        # Collect the faces for a benchmark run that do not have
        # an annotation.
        name_to_faces = {}
        for version in self.dataset.versions:
            name_to_faces[version.person.name] = []
            for photo in version.get_photos():

                if not self.photo_has_single_face(photo):
                    continue

                for provider in self.providers:
                    faces = provider.detected_faces.get(photo.get_photo_id(), None)
                    if faces is None:
                        # No faces may be detected for a given photo.
                        continue
                    for face in faces:
                        if face is not None and not face.is_annotated():
                            if photo.get_photo_id() not in photos_accounted_for:
                                name_to_faces[version.person.name].append(face)
                                photos_accounted_for.add(photo.get_photo_id())

        Labeler(
            name_to_faces,
            self.database.photo_id_to_url,
            self.database,
            port=port,
            use_ngrok=ngrok,
            face_to_resolved_faces=self.face_to_resolved_faces,
        )

    def _filter_annotated_faces(self, provider, comparison_dict):
        """
        Returns all comparisons where both faces in the comparison
        match the seed person (based on manual annotation).
        """
        # This is a set of faces that are labeled and match the seed person.
        safe_ids = set()
        for face in provider.get_all_detected_faces():
            if not face.is_match_with_seed_person():
                continue

            safe_ids.add(face.face_id)

        # This is the subset of comparisons where both faces match the seed person.
        safe_comparisons = {}
        for key, value in comparison_dict.items():
            (face1_id, _), (face2_id, _) = key
            if face1_id in safe_ids and face2_id in safe_ids:
                safe_comparisons[key] = value

        return safe_comparisons

    def _filter_subsampled_comparisons(self, provider, subsampling_dict):
        """
        Returns a subset of comparisons that either are
        1. Same-id comparisons.
        2. Diff-id comparisons permitted by the random subsampling.
        """
        if subsampling_dict is None:
            return provider.comparisons, provider.comparisons

        face_id_to_photo_id = {}
        for version in self.dataset.versions:
            for photo in version.get_photos():
                photo_id = photo.photo_id
                if photo_id in provider.detected_faces:
                    for face in provider.detected_faces[photo_id]:
                        face_id_to_photo_id[face.face_id] = photo_id

        subsampled_comparisons = {}
        for key, value in provider.comparisons.items():
            (face1_id, person1_id), (face2_id, person2_id) = key

            photo1_id = face_id_to_photo_id[face1_id]
            photo2_id = face_id_to_photo_id[face2_id]
            if (photo1_id, photo2_id) in subsampling_dict or (
                photo2_id,
                photo1_id,
            ) in subsampling_dict:
                subsampled_comparisons[key] = value

            if person1_id == person2_id:
                subsampled_comparisons[key] = value

        return subsampled_comparisons

    def run_providers_evaluate(
        self,
        EM_config,
        subsampling_seed=None,
        groups=None,
        person_id_to_was_uploaded=None,
    ):
        if len(self.resolved_faces_per_photo) == 0:
            self._resolve_detected_faces()

        if subsampling_seed:
            subsampling_dict = self._gen_subsampled_photos(subsampling_seed)
        else:
            subsampling_dict = None

        provider_to_results = {}
        for provider in self.providers:
            subsampled_comparisons = self._filter_subsampled_comparisons(
                provider, subsampling_dict
            )
            unsupervised_comparisons = subsampled_comparisons

            subsampled_comparisons = self._filter_annotated_faces(
                provider, subsampled_comparisons
            )
            supervised_comparisons = subsampled_comparisons

            unsupervised_results = compute_results_unsupervised(
                EM_config,
                unsupervised_comparisons,
                groups=groups,
                verbose=True,
                person_id_to_was_uploaded=person_id_to_was_uploaded,
                animation_root=self.database.path,
                # TODO - remove this arg once we don't need the EM convergence gifs anymore.
                labeled_comparisons=supervised_comparisons,
            )

            if supervised_comparisons == {}:
                supervised_results = None
            else:
                supervised_results = compute_results_supervised(
                    supervised_comparisons, groups=groups
                )

            provider_to_results[provider.provider_enum] = {
                "unsupervised": unsupervised_results,
                "supervised": supervised_results,
            }

        # Database stats are computed using the last provider.
        metadata = compute_dataset_statistics(
            unsupervised_comparisons, supervised_comparisons
        )
        return provider_to_results, metadata

    def get_detected_faces(self):
        provider_to_detected_faces = {}
        for provider in self.providers:
            face_id_to_face = {}
            for faces in provider.detected_faces.values():
                for face in faces:
                    face_id_to_face[face.face_id] = face
            provider_to_detected_faces[provider.provider_enum] = face_id_to_face
        return provider_to_detected_faces
