from PIL import Image
from matplotlib import patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import itertools
import shutil
import os

from tqdm import tqdm
from keras.applications import MobileNet
from sklearn.metrics.pairwise import cosine_similarity

# This library is used to run multithreaded downloads.
# https://docs.python.org/3/library/concurrent.futures.html
import concurrent.futures

from .labeler import Labeler
from .provider import gen_provider_class, provider_to_label
#from .analyzer import (
#    compute_results_supervised,
#    compute_results_unsupervised,
#    compute_dataset_statistics,
#)
from .faces import FaceClusterer
from .scraper import ImageScraper
from .data_generator import DataGenerator


class Benchmark:
    """
    This is the top-level class for a benchmark instance (barring the user wrapper).
    It manages the providers, runs detection/comparison loops, spawns a labeling server,
    and provides functions to run analytics.
    """

    NUM_WORKERS_PROVIDER = 2

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
        for i, provider in enumerate(self.providers):
            print(
                "Starting to detect faces for provider",
                provider_to_label(provider.provider_enum),
            )
            if i > 0:
                self._run_provider_detect(self.providers[i], self.providers[i-1])
            else:
                self._run_provider_detect(self.providers[i], None)

            self.database._flush_detections()

        if len(self.resolved_faces_per_photo) == 0:
            self._resolve_detected_faces()

    def _run_provider_detect(self, provider, prev_provider):
        """
        Calls `provider.detect_faces()` for each photo.

        Loads results into `provider.detected_faces` per provider.

        Providers are called in sequence.
        - Detection is done in every picture for the first provider 
            (prev_provider=None and skip=False)
        - Detection is done by the following provider only if the previous
            provider detected just one face (skip=False)
        """
        skip = False
        for version in self.dataset.versions:
            already_done = False
            for photo in version.get_photos():
                if len(provider.detected_faces.get(photo.photo_id, [])) > 0:
                    already_done = True
                    break
            if already_done:
                continue
            print(
                f"Detecting for person {version.person.person_id}: ", end="", flush=True
            )
            for photo in version.get_photos():
                if prev_provider:
                    # If previous provider didn't detect exactly one face, is useless
                    # to keep detecting it on the others (agreement across providers)
                    skip = (1 != len(prev_provider.detected_faces.get(photo.photo_id, [])))
                prov_response = provider.detect_faces(
                    self.benchmark_id, version.person.person_id, photo, skip
                )
                if prov_response is None: # some exception occured
                    print(".", end="", flush=True)
                else:
                    print("✓", end="", flush=True)
            print()

            # Save detected faces after each id is processed, so we lose less
            # data if we have to stop the detection early or it crashes.
            self.database._flush_detections()

    def _test_face_resolution(self, photo_id, clusters_faces, pdf):
        img_path = ImageScraper.get_image_filename(
            self.database.get_photo_dir(), photo_id
        )
        try:
            img = mpimg.imread(img_path)
        except:
            return
        fig, ax = plt.subplots()
        ax.imshow(img)
        colors = list(mcolors.XKCD_COLORS)
        for i, cluster in enumerate(clusters_faces):
            for face in cluster.faces:
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
                        edgecolor=colors[face.provider.value],
                        lw=2,
                    )
                )
                t = ax.annotate(f"{face.provider.value}", xy=(left, upper))
                # https://stackoverflow.com/questions/23696898/adjusting-text-background-transparency
                t.set_bbox(dict(facecolor=colors[face.provider.value], alpha=0.5, edgecolor=colors[face.provider.value]))

        if pdf is None:
            plt.show()
        else:
            pdf.savefig(fig)
            plt.clf()

    def _resolve_detected_faces(self, run_test_demo=False, pdf=None):
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
                        photo_id, self.resolved_faces_per_photo[photo_id], pdf
                    )

    def _gen_subsampled_photos(self, subsampling_seed, groups):
        """
        This generates a subset of all possible diff-id comparisons
        that were selected by random sampling.

        We set the random seed via RandomState so it is deterministic
        (for a fixed choice of seed).
        """

        subsampling_ratios = {}
        for key, g in groups.items(): 
            subsampling_ratios[key] = (
                    self.providers[0].get_num_diffid(
                            photo_id_has_single_face=self.photo_id_has_single_face, group=g,
                        )
                    / self.providers[0].get_num_sameid(
                            photo_id_has_single_face=self.photo_id_has_single_face, group=g,
                        )
                )

        # Considers all pairs of faces from diff people, shuffle, then subsample by
        # taking the first x% of diff-id pairs.
        # ONLY TAKE PAIRS FROM WITHIN THE SAME GROUP
        subsampled_photo_pairs = set()
        rng = np.random.RandomState(subsampling_seed)
        for i, version_i in enumerate(self.dataset.versions):
            photos_i = [photo.get_photo_id() for photo in version_i.get_photos()]
            photos_i = [id for id in photos_i if self.photo_id_has_single_face(id)]

            # Group of person i
            for k, g in groups.items():
                if version_i.person.person_id in g:
                    group = g
                    key = k
                    # Do not break to get the last group (most restrictive one)

            for j, version_j in enumerate(self.dataset.versions):
                if i >= j:
                    continue
                if version_j.person.person_id not in group:
                    continue

                assert i < j
                photos_j = [photo.get_photo_id() for photo in version_j.get_photos()]
                photos_j = [id for id in photos_j if self.photo_id_has_single_face(id)]

                pairs = list(itertools.product(photos_i, photos_j))
                rng.shuffle(pairs)
                truncated = round(len(pairs) / subsampling_ratios[key])
                if truncated == 0:
                    # We could raise the lower bound from 1 but this seems fine/fair.
                    truncated = 1
                pairs = pairs[:truncated]
                subsampled_photo_pairs.update(pairs)
        return subsampled_photo_pairs

    def run_providers_compare(self, groups, subsampling_seed=None):
        """
        Calls `self._run_provider_compare()` for each provider.
        """
        if len(self.resolved_faces_per_photo) == 0:
            self._resolve_detected_faces()

        if subsampling_seed:
            subsampling_set = self._gen_subsampled_photos(subsampling_seed, groups)
        else:
            subsampling_set = None

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the provider's operations and mark each future with its URL
            futures = [
                executor.submit(self._run_provider_compare,provider, subsampling_set=subsampling_set, pbar_position=i)
                for i, provider in enumerate(self.providers)
            ]
            for f in concurrent.futures.as_completed(futures):
                self.database._flush_results()

    def allow_pair_with_subsampling(self, subsampling_dict, face1, face2):
        if subsampling_dict is None:
            return True
        return (face1.photo_id, face2.photo_id) in subsampling_dict or (
            face2.photo_id,
            face1.photo_id,
        ) in subsampling_dict

    def _run_provider_compare(self, provider, subsampling_set=None, pbar_position=None):
        """
        Calls `provider.compare_faces()` for pairs of faces.

        If `subsampling_seed` is specified, then sample over photos
        only for the diff-id comparisons (to roughly ensure there is
        an equal number of same-id and diff-id).
        Otherwise, use all images.
        """

        faces = []
        for idx, face in enumerate(provider.get_all_detected_faces()):
            if not self.photo_id_has_single_face(face.photo_id):
                # print(f"skipped photo {face.photo_id} since source photo has 2+ faces")
                continue
            faces.append(face)

        # remove this assertion once we're sure there are no bugs
        valid_photo_ids = [f.photo_id for f in faces]
        for a, b in subsampling_set:
            assert a in valid_photo_ids
            assert b in valid_photo_ids

        with tqdm(position=pbar_position, leave=True, total=len(faces)**2 + len(subsampling_set),
                  desc=f"Comparing faces with {provider.get_name()}") as pbar:
            for f1, f2 in itertools.product(faces, faces):
                if f1.person_id == f2.person_id and f1.face_id != f2.face_id:
                    provider.compare_faces(f1, f2)
                pbar.update(1)

            self.database._flush_results()

            # this is a cumbersome way to map photo_id to face_id, improve later
            detected_single_faces_flat = {}
            for det_faces in provider.detected_faces.values():
                if len(det_faces) > 1:
                    continue
                for det_face in det_faces:
                    detected_single_faces_flat[det_face.photo_id] = det_face

            for photo_id_1, photo_id_2 in subsampling_set:
                if photo_id_1 in detected_single_faces_flat and photo_id_2 in detected_single_faces_flat:
                    face1 = detected_single_faces_flat[photo_id_1]
                    face2 = detected_single_faces_flat[photo_id_2]
                else:
                    pbar.update(1)
                    continue
                provider.compare_faces(face1, face2)
                pbar.update(1)

        self.database._flush_results()

        print(f'{provider_to_label(provider.provider_enum)} is done!')
        pbar.close()

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
        safe_diff_id_comparisons = {}
        safe_same_id_comparisons = {}
        for key, value in comparison_dict.items():
            (face1_id, person1_id), (face2_id, person2_id) = key
            if face1_id in safe_ids and face2_id in safe_ids:
                safe_comparisons[key] = value
                if person1_id != person2_id:
                    safe_diff_id_comparisons[key] = value
                else:
                    safe_same_id_comparisons[key] = value

        # This is the subset of comparisons where one of the faces does not match the seed person.
        faulty_same_id_comparisons = {}
        faulty_diff_id_comparisons = {}
        for key, value in comparison_dict.items():
            if key not in safe_comparisons:
                (_, person1_id), (_, person2_id) = key
                if person1_id == person2_id:
                    faulty_same_id_comparisons[key] = value
                else:
                    faulty_diff_id_comparisons[key] = value

        comparisons = {
                "safe_diff_id_comparisons" : safe_diff_id_comparisons,
                "safe_same_id_comparisons" : safe_same_id_comparisons,
                "faulty_diff_id_comparisons" : faulty_diff_id_comparisons,
                "faulty_same_id_comparisons" : faulty_same_id_comparisons,
            }
        return comparisons 

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

            if person1_id == person2_id or (
                photo1_id, photo2_id) in subsampling_dict or (
                photo2_id, photo1_id,) in subsampling_dict:
                subsampled_comparisons[key] = value

        return subsampled_comparisons

    def get_detected_faces(self):
        provider_to_detected_faces = {}
        for provider in self.providers:
            face_id_to_face = {}
            for faces in provider.detected_faces.values():
                for face in faces:
                    face_id_to_face[face.face_id] = face
            provider_to_detected_faces[provider.provider_enum] = face_id_to_face
        return provider_to_detected_faces

    def deduplicate(self, min_similarity_threshold=0.95):
        print("Deduplicating...")
        if len(self.resolved_faces_per_photo) == 0:
            self._resolve_detected_faces()

        # COMPARE USING MobileNet
        model = MobileNet(input_shape=(224, 224, 3), include_top=False, pooling='avg')

        for person in self.database.people:
            for padd in [200, 300, 100]:
                # Create a temp folder
                os.mkdir('temp')
                for version in person.versions:
                    for photo in version.get_photos():
                        photo_id = photo.get_photo_id()
                        if photo_id not in self.resolved_faces_per_photo:
                            continue
                        resolved_faces = self.resolved_faces_per_photo[photo_id]

                        if not self.photo_id_has_single_face(photo_id):
                            continue
                        
                        img_path = ImageScraper.get_image_filename(
                            self.database.get_photo_dir(), photo_id
                        )

                        try:
                            im = Image.open(img_path)
                            if im.mode != 'RGB':
                                # convert to RGBA first to avoid warning
                                # we ignore alpha channel if available
                                im = im.convert('RGBA').convert('RGB')
                        except:
                            continue

                        left = 0
                        upper = 0
                        right = 999999
                        lower = 999999
                        for face in resolved_faces[0].faces:
                            (leftA, upperA, rightA, lowerA) = face.bounding_box.bbox
                            left = max(leftA, left)
                            upper = max(upperA, upper)
                            right = min(rightA, right)
                            lower = min(lowerA, lower)

                        x_padding, y_padding = (padd, padd)
                        width, height = (right - left, lower - upper)
                        x_padding *= (width / 2) / 100
                        y_padding *= (height / 2) / 100
                        x_padding, y_padding = (int(x_padding), int(y_padding))

                        (orig_left, orig_upper, orig_right, orig_lower) = (left, upper, right, lower)

                        left = max(0, orig_left - x_padding)
                        upper = max(0, orig_upper - y_padding)
                        right = min(im.width, orig_right + x_padding)
                        lower = min(im.height, orig_lower + y_padding)

                        cropped_image = im.crop((left, upper, right, lower))

                        cropped_image.save(f"temp{os.sep}{photo_id}.jpg")

                if not os.listdir('temp'):
                    shutil.rmtree("temp")
                    # print("No faces detected for ", person.name)
                    continue

                data_generator = DataGenerator(
                    image_dir="temp",
                    batch_size=64,
                    target_size=(224, 224),
                )

                feat_vec = model.predict_generator(data_generator, len(data_generator))
                
                filenames = [i for i in data_generator.valid_image_files]

                encoding_map = {j: feat_vec[i, :] for i, j in enumerate(filenames)}
                

                # get all image ids
                # we rely on dictionaries preserving insertion order in Python >=3.6
                image_ids = np.array([*encoding_map.keys()])

                # put image encodings into feature matrix
                features = np.array([*encoding_map.values()])

                cosine_scores = cosine_similarity(features)

                np.fill_diagonal(
                    cosine_scores, 2.0
                )  # allows to filter diagonal in results, 2 is a placeholder value

                results = {}
                for i, j in enumerate(cosine_scores):
                    duplicates_bool = (j >= min_similarity_threshold) & (j < 2)
                    duplicates = list(image_ids[duplicates_bool])
                    # print(image_ids[i], duplicates)
                    results[image_ids[i]] = duplicates

                # Construct an undirected graph where edges
                # represent duplicate images per difPy.
                G = nx.Graph()
                for orig, dups in results.items():
                    for dup in dups:
                        G.add_edge(orig, dup)

                # Compute the connected components and select the highest
                # quality image per component to keep in the dataset.
                orig_to_dups = {}
                cc = nx.connected_components(G)
                for c in cc:
                    dups = set(c)
                    # Method of using st_size for quality
                    orig = max(c, key=lambda f: os.stat(str(f)).st_size)
                    dups = set([i.name for i in dups])
                    dups.remove(orig.name)
                    orig_to_dups[orig.name] = dups
                    # print(orig.name, dups)

                photos_to_remove = set()
                for k, v in orig_to_dups.items():
                    k = k.split('.jpg')[0]
                    v = [int(num.split('.jpg')[0]) for num in v]
                    if k not in photos_to_remove:
                        photos_to_remove.update(v)

                # Remove from Benchmark and each provider
                for photo_id in photos_to_remove:
                    self.resolved_faces_per_photo[photo_id] = []
                    for provider in self.providers:
                        provider.detected_faces.pop(photo_id, None)
                # Remove from the database
                self.database._remove_detections(photos_to_remove)

                # Remove the temp folder.
                shutil.rmtree("temp")