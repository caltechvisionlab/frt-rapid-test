# Microsoft API imports
import time
import sys
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face import FaceClient

# Amazon API import
import boto3

from enum import Enum
import random
import numpy as np
import traceback
import requests
import base64

from .face import BoundingBox, DetectedFace
from .dataset import Photo


class ProviderType(Enum):
    # Effectively random noise output
    TEST_PROVIDER = 0
    # Realistic random (normal distributions of confidence for match/nonmatch)
    TEST_PROVIDER_V2 = 1
    # Various industry providers
    CLOUDMERSIVE = 2 # not used
    MICROSOFT_AZURE = 3 # not used
    AMAZONAWS = 4
    TUYA = 5  # not used
    FACEPLUSPLUS = 6
    MXFACE = 7  # not used
    CLEARVIEW = 8
    


PROVIDER_TO_LABEL = {
    ProviderType.TEST_PROVIDER: "Pytest",
    ProviderType.MICROSOFT_AZURE: "Azure",
    ProviderType.AMAZONAWS: "Amazon Rekognition",
    ProviderType.FACEPLUSPLUS: "Face++",
    ProviderType.MXFACE: "MXFace",
    ProviderType.CLEARVIEW: "Clearview",
    ProviderType.CLOUDMERSIVE: "Cloudmersive",
}


def provider_to_label(provider_enum):
    return PROVIDER_TO_LABEL[provider_enum]


def label_to_provider(label):
    for k, v in PROVIDER_TO_LABEL.items():
        if v == label:
            return k
    return None


def gen_provider_class(provider_enum):
    """
    Maps from a provider enum encoding to its Python class.
    """
    enum_to_class_map = {
        ProviderType.TEST_PROVIDER: TestProvider,
        ProviderType.TEST_PROVIDER_V2: RealisticTestProvider,
        ProviderType.MICROSOFT_AZURE: MicrosoftAzure,
        ProviderType.AMAZONAWS: AmazonAWS,
        ProviderType.FACEPLUSPLUS: FacePlusPlus,
        ProviderType.CLEARVIEW: Clearview,
        ProviderType.CLOUDMERSIVE: Cloudmersive,
    }
    return enum_to_class_map[provider_enum]


class Provider:
    """
    This is a general class to represent the high-level functionality
    of a cloud provider. We will inherit from this class whenever
    we implement a specific cloud API.
    """

    def __init__(self, database, provider_enum, credentials):
        self.database = database
        self.provider_enum = provider_enum
        self.credentials = credentials

        # Map from photo_id -> list of `DetectedFaces` (from faces.csv)
        self.detected_faces = {}

        # Map from (face id, face id) to comparison outcome (from results.csv)
        self.comparisons = {}

    def get_all_detected_faces(self):
        # https://stackabuse.com/python-how-to-flatten-list-of-lists/
        return (
            face for photo_faces in self.detected_faces.values() for face in photo_faces
        )

    def get_num_detected_faces(self):
        count = 0
        for photo_faces in self.detected_faces.values():
            count += len(photo_faces)
        return count

    def get_detected_face_sameid_counts(self, photo_id_has_single_face=None, group=None):
        person_to_face_count = {}
        for face in self.get_all_detected_faces():
            if not photo_id_has_single_face(face.photo_id):
                continue

            person_id = face.person_id
            if group:
                if person_id not in group: continue
            if person_id not in person_to_face_count:
                person_to_face_count[person_id] = 0
            person_to_face_count[person_id] += 1
        return list(person_to_face_count.values())

    def get_num_sameid(self, **kwargs):
        counts = self.get_detected_face_sameid_counts(**kwargs)
        same_id = sum(max(v * (v - 1) // 2, 0) for v in counts)
        return same_id

    def get_num_diffid(self, **kwargs):
        counts = self.get_detected_face_sameid_counts(**kwargs)
        total = 0
        for i, e1 in enumerate(counts):
            for j, e2 in enumerate(counts):
                if i < j:
                    total += e1 * e2
        return total

    def get_name(self):
        """
        Returns the ProviderType enum string name.
        """
        return str(self.provider_enum).split(".")[1]

    def detect_faces(self, benchmark_id, person_id, photo, skip=False):
        """
        Detects faces in a photo for a single cloud provider.

        Args:
            photo (:class:`Photo`): the image to process.

        Returns:
            A list of :class:`DetectedFace`'s for each bounding
                box detected in the photo.
        """
        # Results are cached, so this function makes <= 1 cloud API
        # call (the child class's implementation of _detect_faces).
        key = photo.get_photo_id()
        if not key in self.detected_faces:
            if skip:
                self.detected_faces[key] = []
            else:
                try:
                    faces = self._detect_faces(benchmark_id, person_id, photo)
                except Exception as err:
                    #print(
                    #    f"Exception in provider _detect_faces for photo {key}:",
                    #    file=sys.stderr,
                    #)
                    #traceback.print_tb(err.__traceback__)
                    return
                self.detected_faces[key] = faces
                for face in faces:
                    self.database._add_detected_face(face)
        return self.detected_faces[key]

    def compare_faces(self, detected_face1, detected_face2):
        # Results are cached, so this function makes <= 1 cloud API
        # call (the child class's implementation of _compare_faces).

        # Technically, the person_id is redundant information
        # but it will make it easier for us to run comparisons later on
        # just using the self.comparisons
        key = (
            (detected_face1.face_id, detected_face1.person_id),
            (detected_face2.face_id, detected_face2.person_id),
        )
        
        if detected_face2.face_id % 10 == 0:
        #    print(f"Comparison ({detected_face1.face_id},{detected_face2.face_id}) {provider_to_label(self.provider_enum)}")
            self.database._flush_results()

        # If the flip of the faces was computed, don't compute again
        # or even store the new result.
        flipped_key = key[::-1]
        if flipped_key in self.comparisons:
            return self.comparisons[flipped_key]

        if not key in self.comparisons:
            try:
                confidence = self._compare_faces(detected_face1, detected_face2)
            except Exception as err:
                print(
                    f"Exception in provider _compare_faces for {detected_face1.face_id} and {detected_face2.face_id}:",
                    file=sys.stderr,
                )
                print(err, file=sys.stderr)
                traceback.print_tb(err.__traceback__)
                # print(".", end="", flush=True)
                return
            self.comparisons[key] = confidence
            benchmark_id = detected_face1.benchmark_id
#            print("Debería añadirse a la base de datos", file=sys.stderr)
            self.database._add_result(benchmark_id, key[0][0], key[1][0], confidence)
        #print("✓", end="", flush=True)
        return self.comparisons[key]

    def _compare_faces(self, detected_face1, detected_face2):
        raise NotImplementedError("This is a provider-specific feature.")

    def _detect_faces(self, benchmark_id, person_id, photo):
        raise NotImplementedError("This is a provider-specific feature.")


class TestProvider(Provider):
    def __init__(self, database, credentials):
        super().__init__(database, ProviderType.TEST_PROVIDER, credentials)

    def _compare_faces(self, detected_face1, detected_face2):
        return eval(f"0.{detected_face1.face_id}{detected_face2.face_id}")

    def _detect_faces(self, benchmark_id, person_id, photo):
        return [
            DetectedFace(
                photo.get_photo_id(),
                person_id,
                BoundingBox(
                    50 + random.randint(0, 10),
                    50 + random.randint(0, 10),
                    50 + random.randint(20, 30),
                    50 + random.randint(20, 30),
                ),
                self.provider_enum,
                benchmark_id,
                metadata="This is metadata.",
            )
        ]


class RealisticTestProvider(TestProvider):
    def __init__(self, database, credentials):
        Provider.__init__(self, database, ProviderType.TEST_PROVIDER_V2, credentials)

    def _compare_faces(self, detected_face1, detected_face2):
        if detected_face1.person_id == detected_face2.person_id:
            # Return from a normal probability distribution such
            # that two same-id faces have high confidence of being the same.
            mean = 0.8
            sd = 0.2
        else:
            # Return from a normal probability distribution such
            # that two diff-id faces have low confidence of being the same.
            mean = 0.1
            sd = 0.2

        # Truncate the distribution to fit within [0,1]
        draw = None
        while draw is None or draw < 0 or draw > 1:
            draw = np.random.normal(mean, sd)
        return draw


class MicrosoftAzure(Provider):
    def __init__(self, database, credentials):
        super().__init__(database, ProviderType.MICROSOFT_AZURE, credentials)

        self.client = FaceClient(
            self.credentials["endpoint"],
            CognitiveServicesCredentials(self.credentials["key"]),
        )

        response = requests.get(
            url = self.credentials["endpoint"] + 'face/v1.0/largefacelists',
            headers = {
                    # Request headers
                    'Content-Type': 'application/json',
                    'Ocp-Apim-Subscription-Key': self.credentials["key"],
                },
        )

        print(" First   ", response.status_code)
        print(" First   ", response.content)

        if len(response.json()) == 0:
            response = requests.put(
                url = self.credentials["endpoint"] + 'face/v1.0/largefacelists/cfrto',
                headers = {
                        # Request headers
                        'Content-Type': 'application/json',
                        'Ocp-Apim-Subscription-Key': self.credentials["key"],
                    },
                params = {
                        # Request parameters
                        'name': 'first',
                    },
            )

            print(" Second   ", response.status_code)
            print(" Second   ", response.content)

    def _compare_faces(self, detected_face1, detected_face2):
        microsoft_id1 = detected_face1.metadata
        microsoft_id2 = detected_face2.metadata
        return self.client.face.verify_face_to_face(
            microsoft_id1, microsoft_id2
        ).confidence

    def _detect_faces(self, benchmark_id, person_id, photo):
        response = requests.post(
            url = self.credentials["endpoint"] + 'face/v1.0/largefacelists/cfrto/persistedfaces',
            headers = {
                    # Request headers
                    'Content-Type': 'application/octet-stream',
                    'Ocp-Apim-Subscription-Key': self.credentials["key"],
                },
            params = {
                    # Request parameters
                    'returnFaceId': 'false',
                    'detectionModel': 'detection_03',
                },
            data = photo.load_as_stream(self.database).read(),
        )

        print(" Detect Face  ", response.status_code, response.content)

        exit(1)

        microsoft_faces = response.json()

        return [
            DetectedFace(
                photo.get_photo_id(),
                person_id,
                BoundingBox(
                    microsoft_face["faceRectangle"]["left"],
                    microsoft_face["faceRectangle"]["top"],
                    microsoft_face["faceRectangle"]["width"]
                    + microsoft_face["faceRectangle"]["left"],
                    microsoft_face["faceRectangle"]["height"]
                    + microsoft_face["faceRectangle"]["top"],
                ),
                self.provider_enum,
                benchmark_id,
                # metadata=microsoft_face.face_id,
            )
            for microsoft_face in microsoft_faces
        ]


class AmazonAWS(Provider):
    def __init__(self, database, credentials):
        super().__init__(database, ProviderType.AMAZONAWS, credentials)

        self.client = boto3.client(
            "rekognition",
            region_name="us-west-1",
            aws_access_key_id=self.credentials["endpoint"],
            aws_secret_access_key=self.credentials["key"],
        )

    def _detect_faces(self, benchmark_id, person_id, photo):
        """
        Sources:
        * https://docs.aws.amazon.com/rekognition/latest/dg/API_DetectFaces.html
        * https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/rekognition.html#Rekognition.Client.detect_faces
        * https://aws.amazon.com/blogs/machine-learning/build-your-own-face-recognition-service-using-amazon-rekognition/
        * https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html
        """
        response = self.client.detect_faces(
            Image={"Bytes": photo.load_as_bytes(self.database)}
        )

        detected_faces = []

        # Get image diameters.
        image_width = photo.size[0]
        image_height = photo.size[1]

        # Crop face from image.
        for face in response["FaceDetails"]:
            box = face["BoundingBox"]

            left = box["Left"] * image_width
            top = box["Top"] * image_height
            right = (box["Left"] + box["Width"]) * image_width
            bottom = (box["Top"] + box["Height"]) * image_height
            detected_faces.append(
                DetectedFace(
                    photo.get_photo_id(),
                    person_id,
                    BoundingBox(int(left), int(top), int(right), int(bottom)),
                    self.provider_enum,
                    benchmark_id,
                    metadata=",".join(
                        str(i)
                        for i in [box["Left"], box["Top"], box["Width"], box["Height"]]
                    ),
                )
            )

        return detected_faces

    def _compare_faces(self, detected_face1, detected_face2):
        # https://docs.aws.amazon.com/rekognition/latest/dg/faces-comparefaces.html
        response = self.client.compare_faces(
            SimilarityThreshold=0,
            SourceImage={
                "Bytes": Photo(detected_face1.photo_id, None).load_as_bytes(
                    self.database
                )
            },
            TargetImage={
                "Bytes": Photo(detected_face2.photo_id, None).load_as_bytes(
                    self.database
                )
            },
        )

        err_message = f"Multiple ({len(response['FaceMatches'])}) faces detected in {detected_face1.photo_id} vs {detected_face2.photo_id}, but only 1 face is present in each image..."
        assert len(response["FaceMatches"]) <= 1, err_message

        # For now, also assume that no pair of faces will ever return exactly 0
        err_message = f"No faces detected in {detected_face1.photo_id} vs {detected_face2.photo_id}, but 1 face is present in each image..."
        assert len(response["FaceMatches"]) >= 1, err_message

        return response["FaceMatches"][0]["Similarity"] / 100


class FacePlusPlus(Provider):
    def __init__(self, database, credentials):
        super().__init__(database, ProviderType.FACEPLUSPLUS, credentials)

    def _detect_faces(self, benchmark_id, person_id, photo):
        response = requests.post(
            "https://api-us.faceplusplus.com/facepp/v3/detect",
            params=dict(
                api_key=self.credentials["endpoint"],
                api_secret=self.credentials["key"],
            ),
            files=dict(
                image_file=photo.load_as_bytes(self.database),
            ),
        )

        while True:
            response = response.json()

            if "error_message" in response and "CONCURRENCY_LIMIT_EXCEEDED" in response["error_message"]:
                time.sleep(1)
            else:
                assert "faces" in response
                break

        detected_faces = []
        for face in response["faces"]:
            box = face["face_rectangle"]

            left = box["left"]
            top = box["top"]
            right = box["left"] + box["width"]
            bottom = box["top"] + box["height"]

            detected_faces.append(
                DetectedFace(
                    photo.get_photo_id(),
                    person_id,
                    BoundingBox(int(left), int(top), int(right), int(bottom)),
                    self.provider_enum,
                    benchmark_id,
                    metadata=face["face_token"],
                )
            )

        return detected_faces

    def _compare_faces(self, detected_face1, detected_face2):
        # # When more than 72h have passed face_token is lost
        # face_token1 = detected_face1.metadata
        # face_token2 = detected_face2.metadata

        # response = requests.post(
        #     "https://api-us.faceplusplus.com/facepp/v3/compare",
        #     params=dict(
        #         api_key=self.credentials["endpoint"],
        #         api_secret=self.credentials["key"],
        #         face_token1=face_token1,
        #         face_token2=face_token2,
        #     ),
        # )

        while True:
            response = requests.post(
                "https://api-us.faceplusplus.com/facepp/v3/compare",
                params=dict(
                    api_key=self.credentials["endpoint"],
                    api_secret=self.credentials["key"],
                ),
                files=dict(
                    image_file1=Photo(detected_face1.photo_id, None).load_as_bytes(self.database),
                    image_file2=Photo(detected_face2.photo_id, None).load_as_bytes(self.database),
                ),
            ).json()

            if "error_message" in response and "CONCURRENCY_LIMIT_EXCEEDED" in response["error_message"]:
                time.sleep(1)
            else:
                assert "confidence" in response
                break

        confidence = response["confidence"]
        return confidence / 100
        
        
class Clearview(Provider):
    def __init__(self, database, credentials):
        super().__init__(database, ProviderType.CLEARVIEW, credentials)

    def _detect_faces(self, benchmark_id, person_id, photo):
    
        response = requests.post(
            url = "https://developers.clearview.ai/v1/detect_image",
            headers = { "Authorization": "Bearer " + self.credentials["key"] },
            data = {
                'all_faces': True,
                'image': 'data:image/jpeg;base64,' + base64.b64encode(photo.load_as_bytes(self.database)).decode("ascii"),
            }
        )
        response = response.json()
        
        detected_faces = []
        for item in response["data"]["items"]:
            if "face" != item["kind"]:
                continue
                
            bbox = item["bounding_box"]

            left = bbox[0]
            top = bbox[1]
            right = bbox[2] + bbox[0]
            bottom = bbox[3] + bbox[1]

            detected_faces.append(
                DetectedFace(
                    photo.get_photo_id(),
                    person_id,
                    BoundingBox(left, top, right, bottom),
                    self.provider_enum,
                    benchmark_id,
                )
            )

        return detected_faces

    def _compare_faces(self, detected_face1, detected_face2):
        face1 = detected_face1.crop(self.database, skip_padding=True, return_base64=True)
        face2 = detected_face2.crop(self.database, skip_padding=True, return_base64=True)
        
        response = requests.post(
            url="https://developers.clearview.ai/v1/compare_images",
            headers = { "Authorization": "Bearer " + self.credentials["key"] },
            data = {
                'image_a': 'data:image/jpeg;base64,' + face1, 
                'image_b': 'data:image/jpeg;base64,' + face2,
            }
        )

        if "data" not in response.json():
            print(response.json())
        elif "similarity" not in response.json()["data"]:
            print(response.json()["data"])
        
        return response.json()["data"]["similarity"]


class Cloudmersive(Provider):
    def __init__(self, database, credentials):
        super().__init__(database, ProviderType.CLOUDMERSIVE, credentials)

    def _detect_faces(self, benchmark_id, person_id, photo):
    
        response = requests.post(
            url = "https://api.cloudmersive.com/image/face/locate",
            headers = { "Content-Type": "multipart/form-data", "Apikey" : self.credentials["key"]},
            form = {
                'imageFile': 'data:image/jpeg;base64,' + base64.b64encode(photo.load_as_bytes(self.database)).decode("ascii"),
            }
        )

        response = requests.post(
            "https://api-us.faceplusplus.com/facepp/v3/detect",
            params=dict(
                api_key=self.credentials["endpoint"],
                api_secret=self.credentials["key"],
            ),
            files=dict(
                image_file=photo.load_as_bytes(self.database),
            ),
        )

        response = response.json()
        
        detected_faces = []
        for item in response["data"]["items"]:
            if "face" != item["kind"]:
                continue
                
            bbox = item["bounding_box"]

            left = bbox[0]
            top = bbox[1]
            right = bbox[2] + bbox[0]
            bottom = bbox[3] + bbox[1]

            detected_faces.append(
                DetectedFace(
                    photo.get_photo_id(),
                    person_id,
                    BoundingBox(left, top, right, bottom),
                    self.provider_enum,
                    benchmark_id,
                )
            )

        return detected_faces

    def _compare_faces(self, detected_face1, detected_face2):
        face1 = detected_face1.crop(self.database, skip_padding=True, return_base64=True)
        face2 = detected_face2.crop(self.database, skip_padding=True, return_base64=True)
        
        response = requests.post(
            url="https://developers.clearview.ai/v1/compare_images",
            headers = { "Authorization": "Bearer " + self.credentials["key"] },
            data = {
                'image_a': 'data:image/jpeg;base64,' + face1, 
                'image_b': 'data:image/jpeg;base64,' + face2,
            }
        )
        
        return response.json()["data"]["similarity"]

class MXFace(Provider):
    def __init__(self, database, credentials):
        super().__init__(database, ProviderType.MXFACE, credentials)

    def _detect_faces(self, benchmark_id, person_id, photo):
    
        response = requests.post(
            url = "https://faceapi.mxface.ai/api/v2/face/detect",
            headers = { "Subscriptionkey" : self.credentials["key"] },
            data = {
                'encoded_image': base64.b64encode(photo.load_as_bytes(self.database)).decode("ascii"),
            }
            # params
        )
        response = response.json()
        
        detected_faces = []
        for face in response["faces"]:                
            bbox = face["faceRectangle"]

            left = bbox["x"]
            top = bbox["y"]
            right = bbox["x"] + bbox["width"]
            bottom = bbox["y"] + bbox["height"]

            detected_faces.append(
                DetectedFace(
                    photo.get_photo_id(),
                    person_id,
                    BoundingBox(left, top, right, bottom),
                    self.provider_enum,
                    benchmark_id,
                )
            )

        return detected_faces

    def _compare_faces(self, detected_face1, detected_face2):
        face1 = detected_face1.crop(self.database, skip_padding=True, return_base64=True)
        face2 = detected_face2.crop(self.database, skip_padding=True, return_base64=True)
        
        response = requests.post(
            url="https://faceapi.mxface.ai/api/v2/face/verify",
            headers = { "Subscriptionkey" : self.credentials["key"] },
            data = {
                'encoded_image1': face1, 
                'encoded_image2': face2,
            }
        )
        
        return response.json()["confidence"]
