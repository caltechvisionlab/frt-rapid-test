from PIL import Image
import io
import math


class Person:
    """
    Represents a (seed) person stored in our dataset.
    """

    def __init__(self, name, person_id):
        self.name = name
        self.person_id = person_id
        self.versions = []
        # Later we could add other background attributes, e.g., age

    def get_name(self):
        return self.name

    def get_version(self, version_id):
        return self.versions[version_id]

    def get_version_id(self, version):
        for version_id in range(len(self.versions)):
            if self.versions[version_id].time_collected == version.time_collected:
                return version_id
        raise Exception("This version is not associated with the person.")

    def add_version(self, version):
        """
        Returns the version id of the version added.
        """
        version_id = len(self.versions)
        self.versions.append(version)
        return version_id

    def get_num_versions(self):
        """
        Return the number of versions stored.
        """
        return len(self.versions)

    def get_last_version_id(self):
        version_id = self.get_num_versions() - 1
        if version_id < 0:
            raise Exception(f"This person ({self.get_name()}) has no versions.")
        return version_id


class Version:
    """
    Represents a version of a :class:`Person` stored in our dataset.
    """

    def __init__(self, person, time_collected):
        self.person = person
        self.time_collected = time_collected
        self.photos = []

    def add_photo(self, photo):
        self.photos.append(photo)

    def get_photos(self):
        return self.photos

    def get_num_photos(self):
        return len(self.photos)


class Dataset:
    """
    Represents a collection of :class:`Version`'s, which form a test
    dataset to evaluate Facial Comparison models.

    For now, this is just a wrapper for a list of :class:`Version`'s.
    """

    def __init__(self, versions):
        """
        Initializes a Dataset for a list of Versions.
        """
        self.versions = versions

        # Assert that no person is duplicated
        self.people_names = [v.person.get_name() for v in versions]
        assert len(self.people_names) == len(set(self.people_names))

    def get_person_and_version_ids(self):
        """
        Obtains the person id and version id for eac
        Version in the dataset.
        """
        person_version_ids = []
        for version in self.versions:
            person = version.person
            pair = (person.person_id, person.get_version_id(version))
            person_version_ids.append(pair)
        return person_version_ids


class Photo:
    """
    Represents a single photo stored in a :class:`Version`.
    """

    def __init__(self, photo_id, source_url):
        """
        source_url is None <=> the photo was uploaded locally
        """
        self.photo_id = photo_id
        self.source_url = source_url

    def get_photo_id(self):
        return self.photo_id

    def _needs_to_be_hosted(url):
        return type(url) is not str and (url is None or math.isnan(url))

    def load_as_bytes(self, database):
        filename = database._get_image_path(photo=self)
        image = Image.open(filename)

        # Side effect
        self.size = image.size

        stream = io.BytesIO()
        # https://docs.aws.amazon.com/rekognition/latest/dg/images-orientation.html
        if "exif" in image.info:
            # Note: why does this not fail even though it takes image.format
            # instead of hard-coding JPEG???
            exif = image.info["exif"]
            image.save(stream, format=image.format, exif=exif)
        else:
            image.save(stream, format=image.format)
        image_binary = stream.getvalue()
        return image_binary

    def load_as_stream(self, database):
        filename = database._get_image_path(photo=self)
        return open(filename, "r+b")
