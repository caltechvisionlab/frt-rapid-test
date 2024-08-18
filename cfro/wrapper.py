import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import time
import functools
import pandas as pd
import pyngrok
import os
import configparser
import yaml

from .analyzer.data_interface import AnalysisDataInterface
from .analyzer.pdf_report import main as pdf_report
from .analyzer.gt_estimation import main as gt_estimation
from .analyzer.acc_and_bias_plots import main as accuracy_and_bias_plots
from .database import Database
from .provider import ProviderType, provider_to_label, label_to_provider

DEFAULT_DATABASE_PATH = "cfro_data"
DEFAULT_NAMES_PATH = "names.csv"
DEFAULT_PROVIDERS_PATH = "services.yaml"
DEFAULT_SCRAPER_PATH = "../scrapers.yaml"

RESULTS_DIR = "results"
FACE_PANELS = "face_panels"

# To create the PDF
matplotlib.use('Agg')


def timer(func):
    """
    Print the runtime of the decorated function.
    Source: https://realpython.com/primer-on-python-decorators/
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        funcname = func.__name__
        if not funcname in ["compare_faces", "label_detected_faces", "analyze"]:
            print(f"Started {func.__name__!r} (✓ → success, . → failure)")
        else:
            print(f"Started {func.__name__!r}")
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"\nFinished {func.__name__!r} in {run_time:.4f} secs\n")
        return value

    return wrapper_timer


def set_constants_config(func):
    """
    Passes the user-specified config to the tester.
    """

    @functools.wraps(func)
    def wrapper_func(*args, **kwargs):
        self = args[0]
        self.benchmark = self.database.get_benchmark(0)
        self.benchmark.set_constants_config(self.face_cluster_config)
        return func(*args, **kwargs)

    return wrapper_func


class CFRO:
    """
    This is a benchmark for face comparison cloud APIs, e.g., AWS or Azure.
    Upload a list of names and cloud providers, and the benchmark will scrape up to
    100 faces for each name and use this dataset to test each provider for bias
    (along the dimensions of race/gender). No manual annotations are required,
    although you can provide annotations to validate our unsupervised results
    (by setting ``label_faces = True``).

    A benchmark can run as a local Python process or interactively on a Jupyter/Colab
    notebook. Since the labeler tool (to collect manual face annotations) spawns a web server,
    you need to set ``ngrok = True`` if you are running on Colab.

    There are two files you must create to run the benchmark:

    1. names.csv

    Below is the format to input names (with race/gender information).
    At least 2 names must be entered per race/gender (or none)::

        name,race,gender
        Name 1,Race 1,Gender 1
        Name 2,Race 2,Gender 2
        Name 3,Race 3,Gender 3

    Example::

        name,race,gender
        Joseph Biden,White,Male
        Donald Trump,White,Male
        Elizabeth Warren,White,Female
        Hillary Clinton,White,Female
        Barack Obama,Black,Male
        Cory Booker,Black,Male
        Ketanji Brown Jackson,Black,Female
        Kamala Harris,Black,Female

    If you wish to use local photos of some person instead of images scraped
    from news articles, add the `upload_path` column to `names.csv`.
    Note that local photo uploads do not yet undergo duplicate detection.::

        name,race,gender,upload_path
        Joseph Biden,White,Male,photos/biden
        Donald Trump,White,Male,
        Elizabeth Warren,White,Female,
        Hillary Clinton,White,Female,
        Barack Obama,Black,Male,photos/obama
        Cory Booker,Black,Male,
        Ketanji Brown Jackson,Black,Female,
        Kamala Harris,Black,Female,photos/harris

    2. provider.csv

    Below is the format to input providers (only include rows for providers you wish to benchmark)::

        provider,username,password
        Azure,<endpoint>,<key>
        AWS,<access key ID>,<access key secret>
        Face++,<key>,<secret>

    Also, there is an optional file you can provide to tune the package constants:

    3. config.ini

    If provided, this file would override the default constants for the EM algorithm and/or face grouping algorithm.::

        [UNSUPERVISED_EM_CONSTANTS]
        EM_PRIOR_P_MATCH_GIVEN_SAME_SEED_IDENTITY_AND_DOWNLOADED = 0.6
        ...

    For example, this prior would be 0.6 instead of the default 0.5, using the config above.

    :param names_path:                  path to txt file with input names to scrape (see ``names.txt`` above)
    :param providers_path:              path to csv file with provider credentials (see ``providers.csv`` above)
    :param database_path:               path to database directory for dataset/cached API output, created if it does not yet exist
    :param label_faces:                 enables a labeler tool to annotate the dataset to verify unsupervised results
    :param restore:                     enables use of any existing database at ``database_path``
    :param add_timestamp:               appends a timestamp to ``database_path`` (be careful if ``restore == True``)
    :param max_number_of_photos:        applies an upper bound on the number of photos scraped for the dataset
    :param ngrok:                       enables ngrok to generate a public url for the labeling server (required for Google colab)
    :param ngrok_token:                 ngrok access token (if you choose to register at https://ngrok.com/)
    :param port:                        local port to run the labeling server on, if ``label_faces == True``
    :param subsampling_seed:            if provided, this is the random seed to subsample diff-id comparisons
    :param dataset_label:               optional label included in the output figures
    :param remove_duplicate_images:     enables a scan to remove duplicates from the scraped datasets
    :param scrape_from_past_n_days:     filters out articles older than n days to improve dataset recency
    :param apply_majority_vote:         if True, applies majority vote for for ID estimation, otherwise uses per-provider
    :param use_ramdisk_for_photos:      if True, photos will be stored non-persistenly in memory
    """

    def __init__(
            self,
            names_path=DEFAULT_NAMES_PATH,
            scraper_path=DEFAULT_SCRAPER_PATH,
            providers_path=DEFAULT_PROVIDERS_PATH,
            database_path=DEFAULT_DATABASE_PATH,
            image_source="google_news",
            label_faces=True,
            restore=True,
            add_timestamp=False,
            max_number_of_photos=None,
            ngrok=False,
            ngrok_token=None,
            port=5000,
            subsampling_seed=None,
            dataset_label=None,
            remove_duplicate_images=True,
            scrape_from_past_n_days=None,
            apply_majority_vote=True,
            use_ramdisk_for_photos=False,
    ):
        self.names_path = names_path
        self.scraper_path = scraper_path
        self.providers_path = providers_path
        self.database_path = database_path
        self.label_faces = label_faces
        self.restore = restore
        self.max_number_of_photos = max_number_of_photos
        self.ngrok = ngrok
        self.ngrok_token = ngrok_token
        self.port = port
        self.subsampling_seed = subsampling_seed
        self.dataset_label = dataset_label
        self.remove_duplicate_images = remove_duplicate_images
        self.scrape_articles_from_past_n_days = scrape_from_past_n_days
        self.apply_majority_vote = apply_majority_vote
        self.use_ramdisk_for_photos = use_ramdisk_for_photos

        assert image_source in ["google_news", "google_images"], "Invalid image source. " \
                                                                 "Must be one of 'google_news' or 'google_images'"
        self.image_source = image_source

        if self.ngrok_token:
            # We could move this somewhere so it doesn't run all the time.
            pyngrok.ngrok.set_auth_token(self.ngrok_token)

        if add_timestamp:
            self.database_path += " " + str(datetime.now())
        self.database = Database(self.database_path, use_ramdisk_for_photos=use_ramdisk_for_photos)

        # Read in from names path to scrape for each name.
        (
            self.person_names,
            self.person_paths,
            self.groups_with_name,
            self.names_with_attributes,
        ) = self._parse_input_names_and_paths()

        # Read in from scraper path to access Google Images API credentials.
        self.scraper_credentials = self._parse_input_scraper_credentials()

        # Read in from providers path to access tokens for each API we will test.
        self.providers_to_credentials = self._parse_input_providers()
        self.provider_enums = list(self.providers_to_credentials.keys())

        self.face_cluster_config = self._prepare_constants_config()

        if (not self.database.is_fresh_database) and restore:
            # Try to load an existing benchmark if one exists.
            # A benchmark is created after `load_photos` runs to completion, so it is
            # possible to reach this point for a database with no benchmarks.
            try:
                self.benchmark = self.database.get_benchmark(0)
            except:
                print(
                    "Error: Unable to restore the 1st benchmark from the existing database...try creating a fresh database folder."
                )
                exit(1)
            self.restored = True
        else:
            # Otherwise, create a new database from scratch.
            self.person_ids = []
            for person_name, person_attributes in self.names_with_attributes:
                person_id = self.database.add_person(person_name, **person_attributes)
                self.person_ids.append(person_id)
            for provider_enum, credentials in self.providers_to_credentials.items():
                self.database.update_provider_credentials(provider_enum, credentials)
            self.restored = False

        # Process the race/gender metadata per person.
        self.groups_with_id = {}
        self.union_of_groups_with_id = {}
        for category, groups_per_category in self.groups_with_name.items():
            if category not in self.groups_with_id:
                self.groups_with_id[category] = {}

            for group_type, names in groups_per_category.items():
                assert (
                        len(names) > 1
                ), "At least 2 names per race/gender category are required."
                ids = set()
                for person_name in names:
                    ids.add(self.database._get_person_id(person_name))
                self.groups_with_id[category][group_type] = ids
                self.union_of_groups_with_id[group_type] = ids

    def release_ramdisks(self):
        if self.use_ramdisk_for_photos:
            self.database.release_ramdisks()

    def _prepare_constants_config(self):
        config = configparser.ConfigParser()

        # First, read in the default config.
        root = os.path.abspath(os.path.dirname(__file__))
        default_path = os.path.join(root, "default_config.ini")
        config.read(default_path)

        face_cluster_config_dict = {
            k.upper(): eval(v)
            for k, v in dict(config["FACE_GROUPING_CONSTANTS"]).items()
        }

        return face_cluster_config_dict

    def _parse_input_names_and_paths(self):
        # Returns list of names and list of upload paths.
        names = []
        paths = []
        groups = {
            "race": {},
            "gender": {},
            "race_gender": {},
        }
        names_with_attributes = []

        csv = pd.read_csv(self.names_path)
        for i in range(len(csv)):
            row = csv.iloc[i]
            name = row["name"].strip()
            names.append(name)

            race = row["race"].strip()
            gender = row["gender"].strip()

            if "upload_path" in row and type(row["upload_path"]) is str:
                paths.append(row["upload_path"].strip())
            else:
                paths.append(None)

            #            if race not in groups["race"]:
            #                groups["race"][race] = set()
            #            groups["race"][race].add(name)

            #            if gender not in groups["gender"]:
            #                groups["gender"][gender] = set()
            #            groups["gender"][gender].add(name)

            race_gender = f"{race} {gender}"
            if race_gender not in groups["race_gender"]:
                groups["race_gender"][race_gender] = set()
            groups["race_gender"][race_gender].add(name)

            names_with_attributes.append((name, {"race": race, "gender": gender}))

        return names, paths, groups, names_with_attributes

    def _parse_input_scraper_credentials(self):
        """
        Returns map from enum to its credentials.
        """
        return yaml.safe_load(open(self.scraper_path, "r"))

    def _parse_input_providers(self):
        """
        Returns map from enum to its credentials.
        """
        providers = yaml.safe_load(open(self.providers_path, "r"))
        provider_to_credentials = {}
        for key, vals in providers.items():
            provider_to_credentials[eval(f"ProviderType.{key}")] = {
                "endpoint": vals["credentials"]["username"],
                "key": vals["credentials"]["password"],
            }
        return provider_to_credentials

    def run(self):
        """
        Runs the end-to-end pipeline:

        1. ``load_photos``
        2. ``detect_faces``
        3. ``label_detected_faces`` (if enabled)
        4. ``compare_faces``
        5. ``analyze``
        """
        self.load_photos()
        self.detect_faces()
        self.compare_faces()
        if self.label_faces:
            self.label_detected_faces()
        self.analyze()
        self.release_ramdisks()

    @timer
    def load_photos(self):
        """
        This stage scrapes a dataset of images for each identity from ``names_path``.

        To accomplish this, we use Google Images or Google News API to look up recent news articles for each identity,
        then download photos from these news articles.

        The source urls for the dataset are loaded into the database's ``photo.csv`` file.
        The photos are downloaded into the database's ``photo/`` subdirectory, with filenames
        numbered from ``000000.jpg`` to ``999999.jpg``.
        """
        if self.restored:
            raise NotImplementedError("Can't download new images yet.")

        print(f"Using image source: {self.image_source}")

        self.version_ids = []
        for i, person_id in enumerate(self.person_ids):
            upload_path = self.person_paths[i]
            if upload_path is None:
                version_id = self.database.add_version(
                    person_id,
                    download_source=self.image_source,
                    max_number_of_photos=self.max_number_of_photos,
                    remove_duplicate_images=self.remove_duplicate_images,
                    scrape_articles_from_past_n_days=self.scrape_articles_from_past_n_days,
                    scraper_credentials=self.scraper_credentials,
                )
            else:
                version_id = self.database.add_version(
                    person_id,
                    upload_path=upload_path,
                    max_number_of_photos=self.max_number_of_photos,
                    remove_duplicate_images=self.remove_duplicate_images,
                )
            self.version_ids.append(version_id)

        # Minimum number N of photos downloaded per person
        N = 10

        for i, (person_id, version_id) in enumerate(zip(self.person_ids, self.version_ids)):
            version = self.database.get_person(person_id).get_version(version_id)
            if version.get_num_photos() < N:
                del self.person_ids[i], self.version_ids[i]

        self.dataset_id = self.database.add_dataset(self.person_ids, self.version_ids)
        self.benchmark_id = self.database.add_benchmark(
            self.dataset_id, self.provider_enums
        )
        self.benchmark = self.database.get_benchmark(self.benchmark_id)

    @set_constants_config
    @timer
    def detect_faces(self):
        """
        This stage runs cloud Face Detect APIs (each provider from ``providers_path``) over each image scraped.
        Detections are persisted periodically using the database's ``detections.csv`` file (to protect against crashes).

        Once this is complete, we overlay face detections from all providers (for each image).
        We keep photos with only one face and exactly one face detection per provider.
        All other photos are discarded.
        """
        self.benchmark.run_providers_detect()
        self.benchmark.deduplicate()

    @set_constants_config
    @timer
    def label_detected_faces(self, ngrok=None, port=None):
        """
        This stage spawns a web server to annotate scraped faces for each identity (as match or non-match).
        Labels are loaded into the database's ``annotation.csv`` file.

        :param ngrok: overrides the value `self.ngrok` if desired
        :param port: overrides the value `self.port` if desired
        """
        if ngrok is None:
            ngrok = self.ngrok
        if port is None:
            port = self.port
        self.benchmark.label_detected_faces(ngrok=ngrok, port=port)

    @set_constants_config
    @timer
    def compare_faces(self):
        """
        This stage runs cloud Face Compare APIs (each provider from ``providers_path``) over pairs of faces.
        Comparisons are persisted periodically using the database's ``result.csv`` file (to protect against crashes).

        All possible within-seed-id comparisons are made (i.e., every pair of faces scraped for the same identity).
        We run an equal number of between-seed-id comparisons (faces collected for different identities), using random sampling.
        """
        self.benchmark.run_providers_compare(
            groups=self.union_of_groups_with_id, subsampling_seed=self.subsampling_seed,
        )

    @set_constants_config
    @timer
    def analyze(self, create_pdf_report=True, use_annotations=True, fmr_fnmr_error_range=(0.0, 1.0)):

        service_info = yaml.safe_load(open(self.providers_path, "r"))
        data_interface = AnalysisDataInterface(self.database_path, service_info)
        if create_pdf_report:
            pdf_report(data_interface, service_info, out_dir=self.database_path)
        estimation_results = gt_estimation(data_interface,
                                           service_info,
                                           out_dir=self.database_path,
                                           return_annotations=use_annotations,
                                           majority_vote_gt=self.apply_majority_vote
                                           )
        accuracy_and_bias_plots(estimation_results,
                                data_interface,
                                service_info,
                                out_dir=self.database_path,
                                error_range=fmr_fnmr_error_range
                                )
        print(f"Done. Plots can be found in: {os.path.join(self.database_path, 'plots')}")
        return

    @set_constants_config
    def resolve_faces(self):
        """
        This runs a demo that shows the bounding box groups for
        each input photo.
        """
        pdf = PdfPages(f"{self.database_path}{os.sep}bboxes.pdf")
        self.benchmark._resolve_detected_faces(run_test_demo=True, pdf=pdf)
        pdf.close()
