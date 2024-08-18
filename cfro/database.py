import subprocess
from PIL import Image, ImageOps
from datetime import datetime
import pandas as pd
import shutil
import os

from .scraper import ImageScraper
from .dataset import Dataset, Version, Person, Photo
from .tester import Benchmark
from .face import DetectedFace, BoundingBox
from .labeler import MatchType
from .provider import ProviderType

# Mutex / Semaphore for the threads to access the database
from threading import Lock


# Default file/dir names
PERSON_CSV = "person_meta"
PHOTO_CSV = "photo_meta"
DETECTION_CSV = "detections"
BENCHMARK_CSV = "benchmark"
PROVIDER_CSV = "provider"
DATASET_CSV = "dataset"
VERSION_CSV = "version"
ANNOTATION_CSV = "annotation"
MATCHES_CSV = "matches"
PHOTO_DIR = "photo"
DUPLICATE_PHOTO_DIR = "duplicate_photo"
VERIFY_DATABASE_FILE = "database.info"

RAMDISK_SIZE_PHOTOS = 1024 * 4  # 4GB
RAMDISK_SIZE_DUP_PHOTOS = 1024 * 4  # 4GB


class Database:
    """
    A database saves/loads benchmark data in these csv files:

    **person.csv:** names of people, e.g.,

        Ethan Mann, Ignacio Serna, Pietro Perona

    **photo.csv:** scraped photo filenames/source urls, e.g.,

        0001.jpg -> a.com/img.jpg

        0002.jpg -> b.com/img.jpg

    **version.csv:** versions of photo batches over time, e.g.,

        Ethan Mann v1 -> [0001.jpg, 0002.jpg, ...]

        Ethan Mann v2 -> [0007.jpg, 0008.jpg, ...]

    **dataset.csv:** list of versions per person to test, e.g.,

        Dataset 1 -> [Ethan Mann v1, Ignacio Serna v1, Pietro Perona v1]

        Dataset 2 -> [Ethan Mann v2, Ignacio Serna v2, Pietro Perona v2]

    **provider.csv:** provider API auth/credentials dictionary, e.g.,

        Provider 1 -> ('endpoint': endpoint, 'secret': secret)

        Provider 2 -> ('username': username, 'key': key)

    **benchmark.csv:** dataset and list of providers to test, e.g.,

        Benchmark 1 -> (Dataset 1, [Provider 1, Provider 2])

    **detections.csv:** face detections, e.g.,

        Face 1 -> Provider 1, 0001.jpg, bounding box 1

        Face 2 -> Provider 1, 0001.jpg, bounding box 2

        Face 3 -> Provider 1, 0002.jpg, bounding box 3

    **annotation.csv:** face annotations, e.g.,

        Face 1 -> right id (Ethan Mann)

        Face 2 -> right id (Ethan Mann)

        Face 3 -> wrong id (not Ethan Mann),

    **results.csv:** face comparison scores, e.g.,

        Provider 1, Face 1, Face 2 -> 0.97531

        Provider 1, Face 1, Face 3 -> 0.13579
    """

    # Locks / mutex for thread operations in the database
    lock = Lock()
    detection_lock = Lock()
    comparison_lock = Lock()

    def __init__(self, path, use_ramdisk_for_photos=False):
        """
        Loads an existing database from `path` if one exists.
        Otherwise, creates a new database at `path`.
        """
        self.path = path

        # Lists to store data from person.csv, dataset.csv, and benchmark.csv.
        self.people = []
        self.datasets = []
        self.benchmarks = []

        # Map from provider enum to dictionary of auth info for that provider.
        self.credentials = {}

        # When we run the cloud providers, we intermediately save
        # detections to the detection cache and comparisons to the results cache.
        # These caches are eventually saved to the filesystem.
        # This method saves time and minimizes data loss in the event of a crash.
        self.results_cache = {
            "benchmark_id": [],
            "bbox0_id": [],
            "bbox1_id": [],
            "confidence": [],
        }
        self.detections_cache = {
            "bbox_id": [],
            "photo_id": [],
            "person_id": [],
            "provider_id": [],
            "benchmark_id": [],
            "bbox_coords": [],
            "metadata": [],
        }

        # This optimization tracks the number of photos and detected
        # faces in the database so we do not need an expensive IO
        # operation to compute the number of photos/faces stored.
        self.num_photos = 0
        self.num_faces = 0

        # This optimization maps photo filename ids to their source url
        # so we can easily obtain the source url for a given photo.
        self.photo_id_to_url = {}

        exists = os.path.exists(path)
        if not exists:
            # Create a directory for the database at path,
            # along with any parent directories that do
            # not yet exist.
            os.makedirs(path)

        # All databases have a special verification file, which we
        # check for when we initialize a Database(). If this
        # file exists at the specified path, then we can load
        # the existing database.
        if VERIFY_DATABASE_FILE not in os.listdir(path):
            self._create_empty_database(use_ramdisk_for_photos=use_ramdisk_for_photos)
            self.is_fresh_database = True
        else:
            self._load_database()
            self.is_fresh_database = False

    def _get_subpath(self, subpath: str, csv: bool = True):
        """
        Returns the absolute path, given a subpath within the
        database. For example, if the database root is 'home/database'
        then the input 'images' would output 'home/database/images'

        Args:
            subpath (str): subpath within the database
            csv (bool): flag to append ".csv" to the output path
        """
        return self.path + os.sep + subpath + (".csv" if csv else "")

    def _append_df_to_db(self, csv_prefix, df, header=False):
        """
        Appends the rows in df to the db's file corresponding
        to csv_prefix. Adds a header based on param header,
        which should only be true when the db is initialized.
        """
        subpath = self._get_subpath(csv_prefix)
        df.to_csv(subpath, mode="a", header=header, index=False)

    def _overwrite_df_to_db(self, csv_prefix, df):
        """
        Overwrites the existing db contents with df.
        Always includes the df header.
        Otherwise, behaves the same as `_append_df_to_db`.

        WARNING: this will erase the file `<csv_prefix>.csv`
        """
        subpath = self._get_subpath(csv_prefix)
        df.to_csv(subpath, header=True, index=False)

    def _create_empty_database(self, use_ramdisk_for_photos=False):
        """
        Creates empty CSV files (with headers).
        Creates empty folder for photos/faces/results.
        Also creates a basic database verification file.
        """
        # Database verification file.
        with open(self._get_subpath(VERIFY_DATABASE_FILE, csv=False), "w+") as f:
            f.write("This file exists to verify a database exists here.")

        # Empty CSV's with headers.
        new_df = pd.DataFrame(
            {
                "person_id": [],
                "name": [],
                "race": [],
                "gender": [],
            }  # we might want to make this dynamic for any kind of meta data
        )
        self._append_df_to_db(PERSON_CSV, new_df, header=True)

        new_df = pd.DataFrame(
            {
                "person_id": [],
                "version_id": [],
                "time_collected": [],
                "photo_ids": [],
            }
        )
        self._append_df_to_db(VERSION_CSV, new_df, header=True)

        new_df = pd.DataFrame(
            {
                "photo_id": [],
                "source_url": [],
                "query_person_id": [],
            }
        )
        self._append_df_to_db(PHOTO_CSV, new_df, header=True)

        new_df = pd.DataFrame(
            {
                "dataset_id": [],
                "person_ids": [],
                "version_ids": [],
            }
        )
        self._append_df_to_db(DATASET_CSV, new_df, header=True)

        provider_ids = [e.value for e in ProviderType]
        new_df = pd.DataFrame(
            {
                "provider_id": provider_ids,
                "credentials": [str({}) for _ in provider_ids],
            }
        )
        self._append_df_to_db(PROVIDER_CSV, new_df, header=True)

        new_df = pd.DataFrame(
            {
                "benchmark_id": [],
                "provider_ids": [],
                "dataset_id": [],
            }
        )
        self._append_df_to_db(BENCHMARK_CSV, new_df, header=True)

        new_df = pd.DataFrame(
            {
                "bbox_id": [],
                "photo_id": [],
                "person_id": [],
                "provider_id": [],
                "benchmark_id": [],
                "bbox_coords": [],
                "metadata": [],
            }
        )
        self._append_df_to_db(DETECTION_CSV, new_df, header=True)

        new_df = pd.DataFrame(
            {
                "bbox_id": [],
                "match_type": [],
            }
        )
        self._append_df_to_db(ANNOTATION_CSV, new_df, header=True)

        new_df = pd.DataFrame(
            {
                "benchmark_id": [],
                "bbox0_id": [],
                "bbox1_id": [],
                "confidence": [],
            }
        )
        self._append_df_to_db(MATCHES_CSV, new_df, header=True)

        # Empty folder for photos (no longer for faces or benchmark output).
        if use_ramdisk_for_photos:
            self.init_ramdisk(self._get_subpath(PHOTO_DIR, csv=False), size=RAMDISK_SIZE_PHOTOS)
            self.init_ramdisk(self._get_subpath(DUPLICATE_PHOTO_DIR, csv=False), size=RAMDISK_SIZE_DUP_PHOTOS)
        else:
            os.mkdir(self._get_subpath(PHOTO_DIR, csv=False))
            os.mkdir(self._get_subpath(DUPLICATE_PHOTO_DIR, csv=False))


    def init_ramdisk(self, mount_path: str, size: int = 1024):
        """
        Initializes the RAM disk for storing photos.
        Exits script if creation fails.
        """
        try:
            print("Ramdisk creation requires sudo. Please enter your password.")
            os.makedirs(mount_path, exist_ok=True)
            subprocess.run(['sudo', '-S', 'mount', '-t', 'tmpfs', '-o', f'size={size}M', 'tmpfs', mount_path], check=True)
            print(f"RAM disk created at: {mount_path}")
        except Exception as e:
            print(f"Could not create non-persistent ramdisks for photos. This feature was only tested for Linux.")
            exit(1)

    def release_ramdisks(self):
        for ramdisk_path in [
            self._get_subpath(PHOTO_DIR, csv=False),
            self._get_subpath(DUPLICATE_PHOTO_DIR, csv=False)
        ]:

            subprocess.run(['sudo', '-S', 'umount', ramdisk_path], check=True)
            print(f"Removing RAM disk at: {ramdisk_path}")


    def _load_faces(self):
        """
        This is copied from _load_database. It could be refactored later.
        """
        faces = {}

        face_csv = pd.read_csv(self._get_subpath(DETECTION_CSV))
        num_faces = len(face_csv)
        for i in range(num_faces):
            # For each face, load the bounding box into a `BoundingBox`
            # and create the `DetectedFace` data structure.
            face_row = face_csv.iloc[i]
            bbox_id = face_row["bbox_id"]
            benchmark_id = face_row["benchmark_id"]
            person_id = face_row["person_id"]
            photo_id = face_row["photo_id"]
            provider_id = face_row["provider_id"]
            provider_enum = ProviderType(provider_id)
            metadata = face_row["metadata"]
            bounding_box_args = self._cast_str_to_ints(face_row["bounding_box"])
            bounding_box = BoundingBox(*bounding_box_args)
            face = DetectedFace(
                photo_id,
                person_id,
                bounding_box,
                provider_enum,
                benchmark_id,
                metadata=metadata,
            )
            face.set_face_id(bbox_id)
            faces[bbox_id] = face
        return faces

    def _load_database(self):
        """
        Reads in the data tables from `self.path` and initializes
        the wrappers for Person, Version, Tester, etc.
        """
        # Read in from person.csv to load self.people
        people_csv = pd.read_csv(self._get_subpath(PERSON_CSV))
        for person_id, person_name in enumerate(people_csv["name"]):
            self.people.append(Person(person_name, person_id))

        # Read in from photo.csv to load each Version
        # in the next step.
        photo_csv = pd.read_csv(self._get_subpath(PHOTO_CSV))
        photo_csv.set_index("photo_id", inplace=True, drop=False)
        self.num_photos = len(photo_csv)

        version_csv = pd.read_csv(self._get_subpath(VERSION_CSV))
        for i in range(len(version_csv)):
            # Read in from version.csv for each Person
            version_row = version_csv.iloc[i]
            person_id = version_row["person_id"]
            time_collected = version_row["time_collected"]
            person = self.people[person_id]
            curr_version = Version(person, time_collected)

            # Process the photo ids in this version and add
            # a Photo to the version for each photo id.
            try:
                photo_ids = self._cast_str_to_ints(version_row["photo_ids"])
            except:
                photo_ids = []
            for photo_id in photo_ids:
                photo_row = photo_csv.loc[photo_id]
                assert photo_row["photo_id"] == photo_id
                source_url = photo_row["source_url"]
                curr_version.add_photo(Photo(photo_id, source_url))
                self.photo_id_to_url[photo_id] = source_url

            person.add_version(curr_version)

        dataset_csv = pd.read_csv(self._get_subpath(DATASET_CSV))
        for i in range(len(dataset_csv)):
            # Load the versions for each dataset.
            dataset_row = dataset_csv.iloc[i]
            person_ids = self._cast_str_to_ints(dataset_row["person_ids"])
            version_ids = self._cast_str_to_ints(dataset_row["version_ids"])
            versions = []
            for person_id, version_id in zip(person_ids, version_ids):
                person = self.get_person(person_id)
                version = person.get_version(version_id)
                versions.append(version)
            self.datasets.append(Dataset(versions))

        provider_csv = pd.read_csv(self._get_subpath(PROVIDER_CSV))
        for i in range(len(provider_csv)):
            # Read in the user-inputted credentials for each provider.
            provider_row = provider_csv.iloc[i]
            provider_id = provider_row["provider_id"]
            provider_enum = ProviderType(provider_id)
            provider_credentials = eval(provider_row["credentials"])
            self.credentials[provider_enum] = provider_credentials

        benchmark_csv = pd.read_csv(self._get_subpath(BENCHMARK_CSV))
        for i in range(len(benchmark_csv)):
            # For each existing benchmark, read in its providers
            # but delay reading in face data/results in this loop.
            benchmark_row = benchmark_csv.iloc[i]
            benchmark_id = benchmark_row["benchmark_id"]
            provider_ids = self._cast_str_to_ints(benchmark_row["provider_ids"])
            provider_enums = [ProviderType(i) for i in provider_ids]
            dataset_id = benchmark_row["dataset_id"]
            dataset = self.datasets[dataset_id]
            benchmark = Benchmark(
                self, dataset, provider_enums, self.credentials, benchmark_id
            )
            self.benchmarks.append(benchmark)

        face_id_to_annotation = {}
        annotation_csv = pd.read_csv(self._get_subpath(ANNOTATION_CSV))
        for i in range(len((annotation_csv))):
            # Load each annotation into a lookup table by face id.
            annotation_row = annotation_csv.iloc[i]
            bbox_id = annotation_row["bbox_id"]
            match_type = annotation_row["match_type"]
            match_enum = MatchType(match_type)
            face_id_to_annotation[bbox_id] = match_enum

        bbox_id_to_provider = {}
        bbox_id_to_person_id = {}
        face_csv = pd.read_csv(self._get_subpath(DETECTION_CSV))
        num_faces = len(face_csv)
        self.num_faces = num_faces
        for i in range(num_faces):
            # For each face, load the bounding box into a `BoundingBox`
            # and create the `DetectedFace` data structure.
            face_row = face_csv.iloc[i]
            bbox_id = face_row["bbox_id"]
            benchmark_id = face_row["benchmark_id"]
            person_id = face_row["person_id"]
            photo_id = face_row["photo_id"]
            provider_id = face_row["provider_id"]
            provider_enum = ProviderType(provider_id)
            metadata = face_row["metadata"]
            bounding_box_args = self._cast_str_to_ints(face_row["bbox_coords"])
            bounding_box = BoundingBox(*bounding_box_args)
            face = DetectedFace(
                photo_id,
                person_id,
                bounding_box,
                provider_enum,
                benchmark_id,
                metadata=metadata,
            )
            # Also, update the table mapping face ids to person ids
            bbox_id_to_person_id[bbox_id] = person_id
            # Then apply the annotations, where applicable.
            face.set_face_id(bbox_id)
            if bbox_id in face_id_to_annotation:
                face.annotate(face_id_to_annotation[bbox_id])
            # Finally, add this face to the appropriate provider,
            # under the appropriate photo id.
            for provider in self.benchmarks[benchmark_id].providers:
                if provider.provider_enum == provider_enum:
                    provider.detected_faces[photo_id] = provider.detected_faces.get(
                        photo_id, []
                    ) + [face]
                    # Also, update the table mapping face ids to providers
                    bbox_id_to_provider[bbox_id] = provider
                    break

        result_csv = pd.read_csv(self._get_subpath(MATCHES_CSV))
        num_results = len(result_csv)
        for i in range(num_results):
            # For each comparison, load the data into the appropriate provider.
            result_row = result_csv.iloc[i]
            benchmark_id = result_row["benchmark_id"]
            try:
                face1_id = int(result_row["bbox0_id"])
                face2_id = int(result_row["bbox1_id"])
            except:
                continue
            confidence = result_row["confidence"]
            provider = bbox_id_to_provider[face1_id]
            same_provider = bbox_id_to_provider[face2_id]
            assert provider == same_provider
            person1_id = bbox_id_to_person_id[face1_id]
            person2_id = bbox_id_to_person_id[face2_id]
            key = ((face1_id, person1_id), (face2_id, person2_id))
            provider.comparisons[key] = confidence

    def _get_person_id(self, person_name):
        """
        Args:
            person_name (str): name of the person in the db

        Returns:
            person_id (int) if found, or -1 if not found
        """
        for idx, person in enumerate(self.people):
            if person_name == person.get_name():
                return idx
        return -1

    def _add_photos(self, photos, person_id):
        """
        Adds a list of photos to the db.
        Unlike self._add_photo, this is a single append op.
        """
        new_ids = []
        source_urls = []

        for photo in photos:
            new_id = self.num_photos
            self.num_photos += 1
            self.photo_id_to_url[new_id] = photo.source_url

            new_ids.append(new_id)
            source_urls.append(photo.source_url)

        new_df = pd.DataFrame(
            {
                "photo_id": new_ids,
                "source_url": source_urls,
                "query_person_id": [person_id] * len(new_ids),
            }
        )
        self._append_df_to_db(PHOTO_CSV, new_df)

    def _add_photo(self, photo, person_id):
        """
        Adds a photo to the db.
        Note that source_url is None if the photo is a local upload.
        """
        new_id = self.num_photos
        self.num_photos += 1
        self.photo_id_to_url[new_id] = photo.source_url

        new_df = pd.DataFrame(
            {
                "photo_id": [new_id],
                "source_url": [photo.source_url],
                "query_person_id": [person_id],
            }
        )
        self._append_df_to_db(PHOTO_CSV, new_df)

    def _get_image_path(self, photo=None, photo_id=None):
        photo_dir = self._get_subpath(PHOTO_DIR, csv=False)
        return ImageScraper.get_image_filename(
            photo_dir, photo.get_photo_id() if photo is not None else photo_id
        )

    def _add_version_from_upload(
        self,
        person_id,
        upload_path,
        max_number_of_photos=None,
        remove_duplicate_images=False,
    ):
        """
        Pull all image files from `upload_path` into the database.
        Specifically, image files with the .jpg, .jpeg, or .png format.

        Returns a list of Photo.
        """
        print(f"Transfering photos from {upload_path} to the dataset.")
        if remove_duplicate_images:
            print("Note: local photos will not be checked for duplicates.")

        photos = []
        curr_photo_id = self.get_num_photos()
        photo_dir = self._get_subpath(PHOTO_DIR, csv=False)
        extensions = [".jpg", ".jpeg", ".png"]
        i = 0
        for filename in os.listdir(upload_path):
            if max_number_of_photos is not None and i >= max_number_of_photos:
                break
            if any(filename.lower().endswith(e) for e in extensions):
                i += 1
                src_filename = upload_path + os.sep + filename
                dest_filename = ImageScraper.get_image_filename(
                    photo_dir, curr_photo_id
                )
                shutil.copy(src_filename, dest_filename)

                # https://stackoverflow.com/questions/13872331/rotating-an-image-with-orientation-specified-in-exif-using-python-without-pil-in
                curr_image = Image.open(dest_filename)
                curr_image = ImageOps.exif_transpose(curr_image)
                try:
                    curr_image.save(dest_filename, format=curr_image.format)
                except:
                    rgb_curr_image = curr_image.convert("RGB")
                    rgb_curr_image.save(dest_filename, format=rgb_curr_image.format)

                photos.append(Photo(curr_photo_id, None))
                curr_photo_id += 1
        return photos

    def _add_version_from_download(
        self,
        person_id,
        download_source,
        max_number_of_photos=None,
        remove_duplicate_images=False,
        scrape_articles_from_past_n_days=None,
        scraper_credentials=None,
    ):
        """
        Curate photos from a source. The download source options
        are specified in the scraper.ImageSource enum.

        Returns a list of Photo.
        """
        person = self.get_person(person_id)
        person_name = person.get_name()
        min_photo_id = self.get_num_photos()
        photo_dir = self._get_subpath(PHOTO_DIR, csv=False)
        duplicate_photo_dir = self._get_subpath(DUPLICATE_PHOTO_DIR, csv=False)
        photos = ImageScraper.download_images(
            person_name,
            photo_dir,
            download_source,
            min_photo_id,
            max_number_of_photos=max_number_of_photos,
            duplicate_photo_dir=duplicate_photo_dir if remove_duplicate_images else None,
            scrape_articles_from_past_n_days=scrape_articles_from_past_n_days,
            scraper_credentials=scraper_credentials
        )
        return photos

    def _add_detected_face(self, detected_face):
        """
        Returns an id for a DetectedFace. Updates the csv and DetectedFace.
        """
        with self.detection_lock:
            new_bbox_id = self.num_faces
            self.num_faces += 1
            detected_face.set_face_id(new_bbox_id)

            self.detections_cache["bbox_id"].append(new_bbox_id)
            self.detections_cache["photo_id"].append(detected_face.photo_id)
            self.detections_cache["person_id"].append(detected_face.person_id)
            self.detections_cache["provider_id"].append(detected_face.provider.value)
            self.detections_cache["benchmark_id"].append(detected_face.benchmark_id)
            self.detections_cache["bbox_coords"].append(
                    self._cast_ints_to_str(detected_face.bounding_box.bbox)
                )
            self.detections_cache["metadata"].append(detected_face.metadata)
        return new_bbox_id

    def _save_face_annotation(self, face):
        """
        Saves the annotation for a :class:`DetectedFace` to the database.
        This assumes that the face has already been annotated.
        """
        assert face.has_annotation, "The input face must be annotated."
        new_df = pd.DataFrame(
            {
                "face_id": [face.face_id],
                "match_type": [face.annotation.value],
            }
        )
        self._append_df_to_db(ANNOTATION_CSV, new_df)

    def _remove_detections(self, photos):
        subpath = self._get_subpath(DETECTION_CSV)
        df = pd.read_csv(subpath)
        for photo_id in photos:
            df.drop(df.index[(df["photo_id"] == photo_id)],axis=0,inplace=True)
        # We would have to reload the database
        # df['face_id'] = np.arange(df.shape[0])
        self._overwrite_df_to_db(DETECTION_CSV, df)

    def _flush_detections(self):
        """
        Writes the pending set of detected faces to csv.
        Then resets the cache.
        """
        with self.detection_lock:
            new_df = pd.DataFrame(
                {
                    "bbox_id": self.detections_cache["bbox_id"],
                    "photo_id": self.detections_cache["photo_id"],
                    "person_id": self.detections_cache["person_id"],
                    "provider_id": self.detections_cache["provider_id"],
                    "benchmark_id": self.detections_cache["benchmark_id"],
                    "bbox_coords": self.detections_cache["bbox_coords"],
                    "metadata": self.detections_cache["metadata"],
                }
            )
            self._append_df_to_db(DETECTION_CSV, new_df)

            self.detections_cache = {
                "bbox_id": [],
                "photo_id": [],
                "person_id": [],
                "provider_id": [],
                "benchmark_id": [],
                "bbox_coords": [],
                "metadata": [],
            }

    def _flush_results(self):
        """
        Saves all pending results to the backing data structure.
        """
        with self.comparison_lock:
            new_df = pd.DataFrame(
                {
                    "benchmark_id": self.results_cache["benchmark_id"],
                    "bbox0_id": self.results_cache["bbox0_id"],
                    "bbox1_id": self.results_cache["bbox1_id"],
                    "confidence": self.results_cache["confidence"],
                }
            )
            self._append_df_to_db(MATCHES_CSV, new_df)

            self.results_cache = {
                "benchmark_id": [],
                "bbox0_id": [],
                "bbox1_id": [],
                "confidence": [],
            }

    def _add_result(self, benchmark_id, bbox0_id, bbox1_id, confidence):
        """
        Saves a comparison result to the backing data structure.
        """
        with self.comparison_lock:
            self.results_cache["benchmark_id"].append(benchmark_id)
            self.results_cache["bbox0_id"].append(bbox0_id)
            self.results_cache["bbox1_id"].append(bbox1_id)
            self.results_cache["confidence"].append(confidence)

    def _delete_database(self, force=False):
        """
        This is actually a public method, but should *not* be called
        unless you are absolutely sure you want to clear the db.
        """
        if not force:
            confirm = input(f"Are you sure you want to delete the db at `{self.path}`?")
            if "y" not in confirm:
                print("Delete aborted.")
                return
        shutil.rmtree(self.path)
        print("Database deleted.")

    def get_photos_faces(self):
        """
        Needed to make the conversion from face_id to photo_id
        """
        photo_to_face = {}
        face_to_photo = {}

        face_csv = pd.read_csv(self._get_subpath(DETECTION_CSV))
        num_faces = len(face_csv)
        for i in range(num_faces):
            face_row = face_csv.iloc[i]
            face_id = face_row["face_id"]
            photo_id = face_row["photo_id"]

            photo = photo_to_face.get(photo_id, None)
            if photo:
                photo.add(face_id)
                photo_to_face[photo_id] = photo
            else:
                photo_to_face[photo_id] = {face_id}

            face_to_photo[face_id] = photo_id

        return photo_to_face, face_to_photo

    def get_person_id(self, person_name):
        """
        Args:
            person_name (str): name of the person in the db

        Returns:
            person_id (int) if found in the db

        Raises:
            KeyError: if the person_name is not found in the db
        """
        person_id = self._get_person_id(person_name)
        if person_id == -1:
            raise KeyError("This person is not in the db.")
        return person_id

    def add_person(self, person_name, **person_kwargs):
        """
        Adds a person to the database.

        Args:
            person_name (str): name of the person to add

        Returns:
            person_id (int): unique id of the person in the db

        Raises:
            KeyError: if this person is already in the db
        """
        person_id = self._get_person_id(person_name)
        if person_id == -1:
            # Create a new Person for the interface.
            person_id = len(self.people)
            self.people.append(Person(person_name, person_id))

            kwarg_dict = {key: [value] for key, value in person_kwargs.items()}

            # Append the Person's data to the filesystem db.
            new_df = pd.DataFrame(
                {
                    "person_id": [person_id],
                    "name": [person_name],
                    **kwarg_dict
                }
            )
            self._append_df_to_db(PERSON_CSV, new_df)

            return person_id

        raise KeyError("This person is already in the db.")

    def get_person(self, person_id):
        """
        Looks up a person based on their person id.

        Args:
            person_id (int): id of the person in the db

        Returns:
            Person corresponding to this id

        Raises:
            KeyError: if the person_id is invalid
        """
        if 0 <= person_id < len(self.people):
            return self.people[person_id]
        raise KeyError("This is an invalid person id.")

    def add_version(
        self,
        person_id,
        upload_path=None,
        download_source=None,
        max_number_of_photos=None,
        remove_duplicate_images=False,
        scrape_articles_from_past_n_days=None,
        scraper_credentials=None,
    ):
        """
        Add a version to a person indexed at person_id.

        Args:
            person_id (int): id of the person in the db
            upload_path (None | str): path to the source of images
            download_source (None | str):
            max_number_of_photos (None | int)

        Returns:
            version_id (int): version number for this person

        Raises:
            TypeError: if upload_path and download_source are
                       both provided or if neither are provided
            Exception: if no photos are obtained for the version
        """
        use_upload = download_source is None
        use_download = upload_path is None

        person = self.get_person(person_id)

        ## Don't want to create a new version
        # if len(person.versions) != 0:
        #     return person.get_last_version_id()

        if use_upload == use_download:
            raise TypeError("Upload and download cannot both be specified.")
        elif use_download:
            photos = self._add_version_from_download(
                person_id,
                download_source,
                max_number_of_photos=max_number_of_photos,
                remove_duplicate_images=remove_duplicate_images,
                scrape_articles_from_past_n_days=scrape_articles_from_past_n_days,
                scraper_credentials=scraper_credentials,
            )
            # if len(photos) == 0:
            #     raise Exception("No images were downloaded. Try again.")
        elif use_upload:
            photos = self._add_version_from_upload(
                person_id,
                upload_path,
                max_number_of_photos=max_number_of_photos,
                remove_duplicate_images=remove_duplicate_images,
            )
            if len(photos) == 0:
                raise Exception("No images were uploaded. Try again.")

        time_collected = str(datetime.now())
        new_version = Version(person, time_collected)
        photo_ids = []
        for photo in photos:
            new_version.add_photo(photo)
            photo_ids.append(photo.get_photo_id())

        # Process the photos in a batch.
        self._add_photos(photos, person_id)

        photo_ids = self._cast_ints_to_str(photo_ids)
        version_id = person.add_version(new_version)

        # Append the Version's data to the filesystem db.
        new_df = pd.DataFrame(
            {
                "person_id": [person_id],
                "version_id": [version_id],
                "time_collected": [time_collected],
                "photo_ids": [photo_ids],
            }
        )
        self._append_df_to_db(VERSION_CSV, new_df)

        return version_id

    def get_num_photos(self):
        """
        Return the number of photos spanned by all
        :class:`Version`'s in the database.
        """
        return self.num_photos

    def get_dataset(self, dataset_id):
        """
        Returns the :class:`Dataset` with the given dataset_id.
        """
        return self.datasets[dataset_id]

    def add_dataset(self, person_ids, version_ids):
        """
        Creates a dataset to use for benchmarking.
        Returns the dataset_id.
        """
        dataset_id = len(self.datasets)

        versions = []
        for person_id, version_id in zip(person_ids, version_ids):
            versions.append(self.get_person(person_id).get_version(version_id))
        self.datasets.append(Dataset(versions))

        # Append the Dataset's data to the filesystem db.
        new_df = pd.DataFrame(
            {
                "dataset_id": [dataset_id],
                "person_ids": [self._cast_ints_to_str(person_ids)],
                "version_ids": [self._cast_ints_to_str(version_ids)],
            }
        )
        self._append_df_to_db(DATASET_CSV, new_df)

        return dataset_id

    def add_dataset_with_latest_versions(self, person_ids):
        """
        Same as `add_dataset` but this uses the latest
        version for each person.
        """
        version_ids = []
        for person_id in person_ids:
            version_ids.append(self.get_person(person_id).get_last_version_id())

        return self.add_dataset(person_ids, version_ids)

    def add_benchmark(self, dataset_id, provider_enums):
        """
        Creates a new :class:`Benchmark` to benchmark the input
        dataset and cloud providers.

        Returns a benchmark_id, which can be used to
        access the benchmark via `get_benchmark(benchmark_id)`.
        """
        benchmark_id = len(self.benchmarks)

        # Append the Benchmark's data to the filesystem db.
        provider_ids = self._cast_ints_to_str(e.value for e in provider_enums)
        new_df = pd.DataFrame(
            {
                "benchmark_id": [benchmark_id],
                "provider_ids": [provider_ids],
                "dataset_id": [dataset_id],
            }
        )
        self._append_df_to_db(BENCHMARK_CSV, new_df)

        benchmark = Benchmark(
            self,
            self.get_dataset(dataset_id),
            provider_enums,
            self.credentials,
            benchmark_id,
        )
        self.benchmarks.append(benchmark)
        return benchmark_id

    def get_benchmark(self, benchmark_id):
        """
        Returns the :class:`Benchmark` with the given benchmark_id.
        """
        return self.benchmarks[benchmark_id]

    def update_provider_credentials(self, provider_enum, new_credentials):
        """
        Providers need secret information like auth and/or endpoint link.

        Overwrites the credentials in the db for this provider.

        If new providers are implemented locally, this function will
        update `providers.csv` for the new provider, as long as the ProviderType
        enum is updated in the source code.
        """
        self.credentials[provider_enum] = new_credentials
        new_df = pd.DataFrame(
            {
                "provider_id": [e.value for e in ProviderType],
                "credentials": [str(self.credentials.get(e, {})) for e in ProviderType],
            }
        )
        self._overwrite_df_to_db(PROVIDER_CSV, new_df)

    def get_photo_dir(self):
        return self.path + os.sep + PHOTO_DIR

    def _cast_ints_to_str(_, ids):
        """
        Packs an array of ids into a string using the
        format "1,2,3,4,..."
        """
        return ",".join(str(id) for id in ids)

    def _cast_str_to_ints(_, str_ids):
        """
        Unpacks a string of format "1,2,3,4,..." into
        an array [1,2,3,4,...]
        """
        return [int(id) for id in str(str_ids).split(",")]
