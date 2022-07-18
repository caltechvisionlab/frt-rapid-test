from matplotlib import pyplot as plt
from datetime import datetime
import time
import functools
import matplotlib as mpl
import pandas as pd
import pyngrok
import os
import configparser

from .database import Database
from .scraper import ImageSource
from .provider import ProviderType, provider_to_label, label_to_provider
from .visualizer import *

DEFAULT_DATABASE_PATH = "cfro_data"
DEFAULT_NAMES_PATH = "names.csv"
DEFAULT_PROVIDERS_PATH = "providers.csv"

RESULTS_DIR = "results"
FACE_PANELS = "face_panels"
EMBEDDINGS = "embeddings"
PERF_PLOTS = "perf_plots"
DB_STATS = "dataset_stats"
EM_OUTPUT = "unsupervised"


def timer(func):
    """
    Print the runtime of the decorated function.
    Source: https://realpython.com/primer-on-python-decorators/
    """

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        print(f"Started {func.__name__!r}")
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs\n")
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
    :param constants_config_path:       path to .ini configuration file to override default constants (see ``config.ini`` above)
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
    """

    def __init__(
        self,
        names_path=DEFAULT_NAMES_PATH,
        providers_path=DEFAULT_PROVIDERS_PATH,
        database_path=DEFAULT_DATABASE_PATH,
        constants_config_path=None,
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
    ):
        self.names_path = names_path
        self.providers_path = providers_path
        self.database_path = database_path
        self.constants_config_path = constants_config_path
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

        if self.ngrok_token:
            # We could move this somewhere so it doesn't run all the time.
            pyngrok.ngrok.set_auth_token(self.ngrok_token)

        if add_timestamp:
            self.database_path += " " + str(datetime.now())
        self.database = Database(self.database_path)

        # Read in from names path to scrape for each name.
        (
            self.person_names,
            self.person_paths,
            self.groups_with_name,
        ) = self._parse_input_names_and_paths()

        # Read in from credentials path to access tokens for each API we will test.
        self.providers_to_credentials = self._parse_input_providers()
        self.provider_enums = list(self.providers_to_credentials.keys())

        self.face_cluster_config, self.EM_config = self._prepare_constants_config()

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
            for person_name in self.person_names:
                person_id = self.database.add_person(person_name)
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

    def _prepare_constants_config(self):
        config = configparser.ConfigParser()

        # First, read in the default config.
        root = os.path.abspath(os.path.dirname(__file__))
        default_path = os.path.join(root, "default_config.ini")
        config.read(default_path)

        if self.constants_config_path is not None:
            # If any overrides are provided, read them (with higher priority).
            config.read(self.constants_config_path)

        # Process the config into a dictionary format.
        EM_config_dict = {
            k.upper(): eval(v)
            for k, v in dict(config["UNSUPERVISED_EM_CONSTANTS"]).items()
        }
        face_cluster_config_dict = {
            k.upper(): eval(v)
            for k, v in dict(config["FACE_GROUPING_CONSTANTS"]).items()
        }

        return face_cluster_config_dict, EM_config_dict

    def _parse_input_names_and_paths(self):
        # Returns list of names and list of upload paths.
        names = []
        paths = []
        groups = {
            "race": {},
            "gender": {},
            "race_gender": {},
        }

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

            if race not in groups["race"]:
                groups["race"][race] = set()
            groups["race"][race].add(name)

            if gender not in groups["gender"]:
                groups["gender"][gender] = set()
            groups["gender"][gender].add(name)

            race_gender = f"{race} {gender}"
            if race_gender not in groups["race_gender"]:
                groups["race_gender"][race_gender] = set()
            groups["race_gender"][race_gender].add(name)

        return names, paths, groups

    def _parse_input_providers(self):
        """
        Returns map from enum to its credentials.
        """
        provider_to_credentials = {}
        csv = pd.read_csv(self.providers_path)
        for i in range(len(csv)):
            row = csv.iloc[i]
            provider_id = row["provider"]
            if type(provider_id) is not str:
                provider_enum = ProviderType(provider_id)
            else:
                provider_enum = label_to_provider(provider_id)
            provider_to_credentials[provider_enum] = {
                "endpoint": row["username"],
                "key": row["password"],
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
        if self.label_faces:
            self.label_detected_faces()
        self.compare_faces()
        self.analyze()

    @timer
    def load_photos(self):
        """
        This stage scrapes a dataset of images for each identity from ``names_path``.

        To accomplish this, we use Google News API to look up recent news articles for each identity,
        then download photos from these news articles.

        The source urls for the dataset are loaded into the database's ``photo.csv`` file.
        The photos are downloaded into the database's ``photo/`` subdirectory, with filenames
        numbered from ``000000.jpg`` to ``999999.jpg``.
        """
        if self.restored:
            raise NotImplementedError("Can't download new images yet.")
        self.version_ids = []
        for i, person_id in enumerate(self.person_ids):
            upload_path = self.person_paths[i]
            if upload_path is None:
                version_id = self.database.add_version(
                    person_id,
                    download_source=ImageSource.GOOGLE_NEWS,
                    max_number_of_photos=self.max_number_of_photos,
                    remove_duplicate_images=self.remove_duplicate_images,
                    scrape_articles_from_past_n_days=self.scrape_articles_from_past_n_days,
                )
            else:
                version_id = self.database.add_version(
                    person_id,
                    upload_path=upload_path,
                    max_number_of_photos=self.max_number_of_photos,
                    remove_duplicate_images=self.remove_duplicate_images,
                )
            self.version_ids.append(version_id)

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
        Detections are persisted periodically using the database's ``face.csv`` file (to protect against crashes).

        Once this is complete, we overlay face detections from all providers (for each image).
        We keep photos with only one face and exactly one face detection per provider.
        All other photos are discarded.
        """
        self.benchmark.run_providers_detect()

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
            subsampling_seed=self.subsampling_seed,
        )

    def _get_output_name(self, provider_enum, plot_name, subdir="", ext=".pdf"):
        return f'{self.database_path}{os.sep}{RESULTS_DIR}{os.sep}{subdir}{os.sep if subdir else ""}{plot_name}_{provider_enum.value}{ext}'

    @set_constants_config
    @timer
    def analyze(self):
        """
        This stage processes the Compare output to generate the following graphics:

        Unsupervised:

        1. Match/non-match EM estimate distributions vs real distributions
        2. FMR/FNMR EM estimate vs real FMR/FNMR
        3. FMR/FNMR EM estimate vs real FMR/FNMR with (FMR, FNMR) scatter points at various thresholds
        4. FMR/FNMR EM bootstrapped estimate vs real FMR/FNMR

        Supervised:

        1. Performance curves (e.g., FNMR-FMR, Precision-Recall, etc.) - saved to the ``results/perf_plots/`` folder.
        2. MDS of faces scraped per identity (using confidences from the providers) - saved to the ``results/embeddings/`` folder.
        3. Extreme faces per identity (faces sorted by average confidence for match or non-match comparisons) - saved to the ``results/face_panels/`` folder.
        4. Database statistics - saved to the ``results/dataset_stats/`` folder.
        """
        # TODO - further organize the output and/or improve the visualizations.
        root = f"{self.database_path}{os.sep}{RESULTS_DIR}"
        for path in [
            root,
            f"{root}{os.sep}{DB_STATS}",
            f"{root}{os.sep}{FACE_PANELS}",
            f"{root}{os.sep}{EMBEDDINGS}",
            f"{root}{os.sep}{PERF_PLOTS}",
            f"{root}{os.sep}{EM_OUTPUT}",
        ]:
            if not os.path.exists(path):
                os.mkdir(path)

        # We record which identities were scraped vs uploaded, so we can adjust the
        # EM same-id comparison priors accordingly.
        person_id_to_was_uploaded = {
            self.database._get_person_id(name): (path is not None)
            for name, path in zip(self.person_names, self.person_paths)
        }

        results, dataset_stats = self.benchmark.run_providers_evaluate(
            self.EM_config,
            subsampling_seed=self.subsampling_seed,
            groups=self.union_of_groups_with_id,
            person_id_to_was_uploaded=person_id_to_was_uploaded,
        )

        # Here, we will use the unsupervised results and supervised results.
        for provider_enum, provider_results in results.items():
            provider_label = provider_to_label(provider_enum)

            supervised_results = provider_results["supervised"]
            unsupervised_results = provider_results["unsupervised"]

            # TODO - include multi-group comparison plots like we do for the supervised results
            # on line 622 as of this commit. For example, we would like to have plots that
            # compare races, compare genders, compare race x gender categories for bias.

            for group_name, group_unsupervised_results in unsupervised_results.items():
                unsup_root = f"{root}{os.sep}{EM_OUTPUT}"
                # TODO - once we have multiple providers running, we need to include
                # the provider information in the filename. Otherwise, each provider
                # will overwrite the same filename here.
                subdir = f'{unsup_root}{os.sep}{str(group_name).replace(" ", "_")}'
                if not os.path.exists(subdir):
                    os.mkdir(subdir)

                if supervised_results is None:
                    group_supervised_results = None
                    true_match_confidences = None
                    true_nonmatch_confidences = None
                else:
                    group_supervised_results = supervised_results["metrics"][group_name]
                    true_match_confidences = group_supervised_results[
                        "true_match_confidences"
                    ]
                    true_nonmatch_confidences = group_supervised_results[
                        "no_match_confidences"
                    ]

                # Loop over the EM algo input param specs here.
                for estimate_nonmatch_distribution_each_iteration in [True, False]:

                    # This is the true unsupervised output.
                    kernel_output = group_unsupervised_results["point"][
                        estimate_nonmatch_distribution_each_iteration
                    ]
                    bootstrapped_output = group_unsupervised_results["bootstrapped"][
                        estimate_nonmatch_distribution_each_iteration
                    ]

                    generate_unsupervised_plots(
                        true_match_confidences,
                        true_nonmatch_confidences,
                        kernel_output,
                        bootstrapped_output,
                        subdir,
                        estimate_nonmatch_distribution_each_iteration=estimate_nonmatch_distribution_each_iteration,
                    )

        # From this point onward, all results are supervised.
        results = {
            provider_enum: provider_results["supervised"]
            for provider_enum, provider_results in results.items()
        }

        # This generates the database statistics folder files *without* race/gender groups reflected.
        show_dataset_statistics(
            dataset_stats,
            self.database.people,
            f"{self.database_path}{os.sep}{RESULTS_DIR}{os.sep}{DB_STATS}{os.sep}",
        )
        # This generates the database statistics folder files *with* categories for race/gender groups.
        show_dataset_statistics(
            dataset_stats,
            self.database.people,
            f"{self.database_path}{os.sep}{RESULTS_DIR}{os.sep}{DB_STATS}{os.sep}",
            groups=self.union_of_groups_with_id,
            disjoint_groups=list(self.groups_with_id["race_gender"].keys()),
        )

        provider_faces = self.benchmark.get_detected_faces()

        # The following plots are all saved to the PERF subdirectory.
        # TODO - we could generate a folder per person or per provider to organize the output.

        # Plot supervised FMR vs FNMR (and related curves) for all providers on one graphs!!!
        axes, figs = prepare_empty_plots_v2(4)
        for provider_enum, provider_results in results.items():
            try:
                provider_aggregate = provider_results["metrics"]["aggregate"]
            except:
                # If there aren't any supervised results, we can exit here.
                exit(0)
            provider_label = provider_to_label(provider_enum)
            plot_data_v2(
                provider_aggregate,
                None,
                axes=axes,
                tag=provider_label,
                dataset_label=self.dataset_label,
                provider_name=None,
            )
        plot_data_v2_consolidate(
            axes,
            figs,
            f"{self.database_path}{os.sep}{RESULTS_DIR}{os.sep}{PERF_PLOTS}{os.sep}all_providers_aggregate_plot_v2",
        )

        # If there are multiple providers being tested, display
        # all of the aggregate FMR-FNMR curves in a single figure.
        # This is the same as above but just uses another plotting library.
        if len(results) > 1:
            plot_many_fmr_fnmr(
                results,
                f"{self.database_path}{os.sep}{RESULTS_DIR}{os.sep}{PERF_PLOTS}{os.sep}fmr_fnmr_combined.pdf",
            )

        # Plot supervised FMR vs FNMR (and related curves) for all providers, per race and/or gender group.
        if self.union_of_groups_with_id is not None:
            for group in self.union_of_groups_with_id:
                # Plot these (e.g., FMR vs FNMR) for all providers on one graphs!!!
                axes, figs = prepare_empty_plots_v2(4)
                for provider_enum, provider_results in results.items():
                    provider_aggregate = provider_results["metrics"][group]
                    provider_label = provider_to_label(provider_enum)
                    plot_data_v2(
                        provider_aggregate,
                        None,
                        axes=axes,
                        tag=provider_label,
                        dataset_label=self.dataset_label,
                        provider_name=None,
                    )
                plot_data_v2_consolidate(
                    axes,
                    figs,
                    f"{self.database_path}{os.sep}{RESULTS_DIR}{os.sep}{PERF_PLOTS}{os.sep}all_providers_{group}_plot_v2",
                )

        # These are individual-provider plots, i.e., only one provider per figure.
        for provider_enum, provider_results in results.items():
            provider_label = provider_to_label(provider_enum)

            # Note: race/gender groups are also processed here and lumped in with "people".
            # This could be cleaned up, but for now, wherever a comment says "each person" it
            # means "each person's individual results and each race/gender group's aggregate results".
            for key, metrics in provider_results["metrics"].items():
                # Plot the true match and true non-match confidences collected for each person
                # on a separate graph.
                plot_match_nonmatch_confidences(
                    provider_enum,
                    metrics,
                    self._get_output_name(
                        provider_enum,
                        f"person{key}_match_nonmatch_confidences",
                        subdir=PERF_PLOTS,
                    ),
                )

                # Plot a histogram of all confidences collected for each person on a separate graph.
                plot_all_confidences(
                    provider_enum,
                    metrics,
                    self._get_output_name(
                        provider_enum, f"person{key}_all_confidences", subdir=PERF_PLOTS
                    ),
                )

                # Plot supervised FMR vs FNMR (and related curves) for each person on a separate graph.
                # This is the same as below except using a separate plotting library.
                plot_fmr_fnmr(
                    provider_enum,
                    metrics,
                    self._get_output_name(
                        provider_enum, f"person{key}_fmr_fnmr", subdir=PERF_PLOTS
                    ),
                )

                # Plot supervised FMR vs FNMR (and related curves) for each person on a separate graph.
                plot_data_v2(
                    metrics,
                    f"{self.database_path}{os.sep}{RESULTS_DIR}{os.sep}{PERF_PLOTS}{os.sep}person{key}_plot_v2_"
                    + str(provider_enum.value),
                    dataset_label=self.dataset_label,
                    provider_name=provider_label,
                )

            # Plot supervised FMR vs FNMR (and related curves) for every person on the same graph.
            # This also includes the aggregate curves.
            axes, figs = prepare_empty_plots_v2(4)
            num_items_in_legend = len(
                [
                    key
                    for key in provider_results["metrics"]
                    if key == "aggregate" or type(key) is not str
                ]
            )
            # If there are many people, we need to include more unique legend colors and resize the legend.
            huge_legend = num_items_in_legend > 15
            if huge_legend:
                prop_cycle_old = mpl.rcParams["axes.prop_cycle"]
                mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
                    color=list(plt.cm.tab20.colors) + list(plt.cm.tab20.colors)
                ) + mpl.cycler(markersize=[6] * 20 + [3] * 20)
            for key, metrics in provider_results["metrics"].items():
                # TODO - should the aggregate be included? (seems fine)
                if key == "aggregate":
                    person_name = "Aggregate"
                elif type(key) is str:
                    # Do not include groups in this plot.
                    # They are the only other key with str type,
                    # as individual people have int keys (person ids).
                    continue
                else:
                    person_name = self.database.people[key].name

                plot_data_v2(
                    metrics,
                    None,
                    axes=axes,
                    tag=person_name,
                    dataset_label=self.dataset_label,
                    provider_name=provider_label,
                )
            plot_data_v2_consolidate(
                axes,
                figs,
                f"{self.database_path}{os.sep}{RESULTS_DIR}{os.sep}{PERF_PLOTS}{os.sep}all_people_plot_v2_"
                + str(provider_enum.value),
                huge_legend=huge_legend,
            )
            # Restore the legend to its old properties.
            if huge_legend:
                mpl.rcParams["axes.prop_cycle"] = prop_cycle_old

            # Plot supervised FMR vs FNMR (and related curves) for each category of the race, gender,
            # and race-gender attributes. For example, one plot will have Male vs Female on the same figure.
            for label, groups in self.groups_with_id.items():
                # Plot supervised FMR vs FNMR (and related curves) for every sub-group on the same graph.
                axes, figs = prepare_empty_plots_v2(4)
                for group in groups:
                    metrics = provider_results["metrics"][group]
                    plot_data_v2(
                        metrics,
                        None,
                        axes=axes,
                        tag=group,
                        dataset_label=self.dataset_label,
                        provider_name=provider_label,
                    )
                plot_data_v2_consolidate(
                    axes,
                    figs,
                    f"{self.database_path}{os.sep}{RESULTS_DIR}{os.sep}{PERF_PLOTS}{os.sep}groups_{label}_plot_v2_"
                    + str(provider_enum.value),
                )

            # Plot supervised FMR vs FNMR (and related curves) for all people in a given race/gender category.
            for label, groups in self.union_of_groups_with_id.items():
                # Plot supervised FMR vs FNMR (and related curves) for every sub-group on the same graph.
                axes, figs = prepare_empty_plots_v2(4)
                for group in groups:
                    metrics = provider_results["metrics"][group]
                    plot_data_v2(
                        metrics,
                        None,
                        axes=axes,
                        tag=group,
                        dataset_label=self.dataset_label,
                        provider_name=provider_label,
                    )
                plot_data_v2_consolidate(
                    axes,
                    figs,
                    f"{self.database_path}{os.sep}{RESULTS_DIR}{os.sep}{PERF_PLOTS}{os.sep}groups_{label}_plot_v2_"
                    + str(provider_enum.value),
                )

            # TODO - extend the face panel HTML to handle multiple providers more effectively
            names_ids = list(
                zip(
                    self.benchmark.dataset.people_names,
                    [
                        person_id
                        for person_id, _ in self.benchmark.dataset.get_person_and_version_ids()
                    ],
                )
            )
            create_extreme_faces_landing_page(
                provider_enum,
                names_ids,
                self._get_output_name(
                    provider_enum, f"index", ext=".html", subdir=FACE_PANELS
                ),
            )

            faces = self.database._load_faces()
            photo_id_to_url = self.database.photo_id_to_url

            for person, extrema in provider_results["extrema"].items():
                if person == "aggregate":
                    continue

                # This produces an HTML panel that shows the faces with the
                # highest/lowest average confidence values over all comparisons.
                for extrema_type in ["confusing", "representative"]:
                    for summary_stat in ["avg", "median"]:
                        show_extreme_faces(
                            self.database,
                            faces,
                            self.database.people,
                            photo_id_to_url,
                            provider_enum,
                            extrema["same_id"],
                            True,  # same-id
                            self._get_output_name(
                                provider_enum,
                                f"{extrema_type}_{summary_stat}_same_id_{person}",
                                ext=".html",
                                subdir=FACE_PANELS,
                            ),
                            summary_stat,
                            extrema_type,
                            False,  # multiple_ids
                        )

                        show_extreme_faces(
                            self.database,
                            faces,
                            self.database.people,
                            photo_id_to_url,
                            provider_enum,
                            extrema["diff_id"],
                            False,  # diff-id
                            self._get_output_name(
                                provider_enum,
                                f"{extrema_type}_{summary_stat}_diff_id_{person}",
                                ext=".html",
                                subdir=FACE_PANELS,
                            ),
                            summary_stat,
                            extrema_type,
                            False,  # multiple_ids
                        )

                    show_extreme_face_pairs(
                        self.database,
                        faces,
                        self.database.people,
                        photo_id_to_url,
                        provider_enum,
                        extrema["same_id"],
                        True,  # same-id
                        self._get_output_name(
                            provider_enum,
                            f"{extrema_type}_pairs_same_id_{person}",
                            ext=".html",
                            subdir=FACE_PANELS,
                        ),
                        extrema_type,
                        False,  # multiple_ids
                    )

                    show_extreme_face_pairs(
                        self.database,
                        faces,
                        self.database.people,
                        photo_id_to_url,
                        provider_enum,
                        extrema["diff_id"],
                        False,  # same-id
                        self._get_output_name(
                            provider_enum,
                            f"{extrema_type}_pairs_diff_id_{person}",
                            ext=".html",
                            subdir=FACE_PANELS,
                        ),
                        extrema_type,
                        True,  # multiple_ids
                    )

            # This produces a 2d embedding for each face annotated as a correct match
            # for its seed person, with 2d distances ~ face API similarity output scores.
            for person, matrix in provider_results["matrix"].items():
                if person == "all":
                    continue
                show_faces_2d_embedding(
                    self.database,
                    matrix,
                    provider_results["face_to_person"],
                    provider_faces[provider_enum],
                    out=self._get_output_name(
                        provider_enum,
                        f"face_embedding_REPLACE_p{person}",
                        subdir=EMBEDDINGS,
                    ),
                )

            # We skipped the aggregate data before, so here we perform the extreme face analysis
            # over all people/faces.
            for extrema_type in ["confusing", "representative"]:
                label = (
                    extrema_type if extrema_type == "confusing" else "distinguishable"
                )
                for summary_stat in ["avg", "median"]:
                    show_extreme_faces(
                        self.database,
                        faces,
                        self.database.people,
                        photo_id_to_url,
                        provider_enum,
                        provider_results["extrema"]["aggregate"]["diff_id"],
                        False,  # same-id
                        self._get_output_name(
                            provider_enum,
                            f"{label}_{summary_stat}_diff_id",
                            ext=".html",
                            subdir=FACE_PANELS,
                        ),
                        summary_stat,
                        extrema_type,
                        True,  # multiple_ids
                    )

                    show_extreme_faces(
                        self.database,
                        faces,
                        self.database.people,
                        photo_id_to_url,
                        provider_enum,
                        provider_results["extrema"]["aggregate"]["same_id"],
                        True,  # same-id
                        self._get_output_name(
                            provider_enum,
                            f"{extrema_type}_{summary_stat}_same_id",
                            ext=".html",
                            subdir=FACE_PANELS,
                        ),
                        summary_stat,
                        extrema_type,
                        True,  # multiple_ids
                    )

                show_extreme_face_pairs(
                    self.database,
                    faces,
                    self.database.people,
                    photo_id_to_url,
                    provider_enum,
                    provider_results["extrema"]["aggregate"]["diff_id"],
                    False,  # diff-id
                    self._get_output_name(
                        provider_enum,
                        f"{label}_pairs_diff_id",
                        ext=".html",
                        subdir=FACE_PANELS,
                    ),
                    extrema_type,
                    True,  # multiple_ids
                )

                show_extreme_face_pairs(
                    self.database,
                    faces,
                    self.database.people,
                    photo_id_to_url,
                    provider_enum,
                    provider_results["extrema"]["aggregate"]["same_id"],
                    True,  # same-id
                    self._get_output_name(
                        provider_enum,
                        f"{extrema_type}_pairs_same_id",
                        ext=".html",
                        subdir=FACE_PANELS,
                    ),
                    extrema_type,
                    True,  # multiple_ids
                )

        return results

    @set_constants_config
    def resolve_faces(self):
        """
        This runs a demo that shows the bounding box groups for
        each input photo.
        """
        self.benchmark._resolve_detected_faces(run_test_demo=True)
