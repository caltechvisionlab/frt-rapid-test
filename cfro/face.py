from PIL import Image
import io
import base64

from .labeler import MatchType

# Default padding for crops is 0, for now.
X_PADDING = 0
Y_PADDING = 0


class BoundingBox:
    """
    Simple bounding box representation. Stores the
    (left,upper) and (right,lower) corners.

    Units are pixels (e.g., 800 px), so the bounding box
    is absolute (not relative).

    Note that the y-axis points downward for photos, so
    (left,upper) would visually appear as (left,lower).
    """

    def __init__(self, left, upper, right, lower):
        self.bbox = (left, upper, right, lower)

    def compute_intersection_over_union(self, other):
        """
        Computes IoU metric, explained at
        https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
        """
        leftA, upperA, rightA, lowerA = self.bbox
        leftB, upperB, rightB, lowerB = other.bbox

        left = max(leftA, leftB)
        upper = max(upperA, upperB)
        right = min(rightA, rightB)
        lower = min(lowerA, lowerB)

        if left >= right or upper >= lower:
            return 0
        intersection = BoundingBox(left, upper, right, lower)
        intersection_area = intersection.get_area()
        iou = intersection_area / (
            self.get_area() + other.get_area() - intersection_area
        )
        return iou

    def add_padding(
        self, max_width, max_height, x_padding, y_padding, percentage=False
    ):
        """
        Returns a new bounding box with padding.

        If percentage is True, then x_padding and y_padding are
        percent increases in the x and y directions (e.g., 50%).

        If percentage is False, then x_padding and y_padding will be
        applied to each line segment of the bounding box.
        """
        if percentage:
            width, height = self.get_width_height()
            x_padding *= (width / 2) / 100
            y_padding *= (height / 2) / 100
            x_padding, y_padding = (int(x_padding), int(y_padding))

        (orig_left, orig_upper, orig_right, orig_lower) = self.bbox

        left = max(0, orig_left - x_padding)
        upper = max(0, orig_upper - y_padding)
        right = min(max_width, orig_right + x_padding)
        lower = min(max_height, orig_lower + y_padding)
        return BoundingBox(left, upper, right, lower)

    def get_width_height(self):
        """
        Returns the box's (width, height).
        """
        (left, upper, right, lower) = self.bbox
        return (right - left, lower - upper)

    def get_area(self):
        w, h = self.get_width_height()
        return w * h

    def get_top_left_width_height(self):
        """
        Returns another representation of a bounding box:
        (left, upper) corner and the box's (width, height).
        """
        (left, upper, _, _) = self.bbox
        width, height = self.get_width_height()
        return [left, upper, width, height]


class DetectedFace:
    """
    Represents a face detected by a single :class:`Provider` for a :class:`Photo`.

    These are persistent.
    """

    def __init__(
        self, photo_id, person_id, bounding_box, provider, benchmark_id, metadata=None
    ):
        self.photo_id = photo_id
        self.person_id = person_id
        self.bounding_box = bounding_box
        self.provider = provider
        self.benchmark_id = benchmark_id
        self.metadata = metadata
        self.has_annotation = False

    def set_face_id(self, face_id):
        self.face_id = face_id


    def uncropped(self,
                  database,
                  out_filename=None,
                  return_stream=False,
                  return_base64=False,
                  return_image=False,
                  ):
        source_filename = database._get_image_path(photo_id=self.photo_id)

        im = Image.open(source_filename)

        if out_filename is not None:
            im.save(out_filename)
        else:
            # Source: https://aws.amazon.com/blogs/machine-learning/build-your-own-face-recognition-service-using-amazon-rekognition/
            stream = io.BytesIO()
            # Note: I removed 'format=cropped_image.format'

            # https://stackoverflow.com/questions/48248405/cannot-write-mode-rgba-as-jpeg/48248432
            failed = False
            for i in range(2):
                if i == 1 and not failed:
                    break
                try:
                    if "exif" in im.info:
                        exif = im.info["exif"]
                        # Does this reformat as JPEG if needed?
                        im.save(stream, format="JPEG", exif=exif)
                    else:
                        im.save(stream, format="JPEG")
                except Exception as e:
                    if i == 1:
                        raise e
                    failed = True
                    im = im.convert("RGB")

            if return_stream:
                return stream
            elif return_image:
                return im
            image_binary = stream.getvalue()
            if return_base64:
                return base64.b64encode(image_binary).decode("ascii")
            return image_binary


    def crop(
        self,
        database,
        custom_padding=None,
        skip_padding=False,
        out_filename=None,
        return_stream=False,
        return_base64=False,
        return_image=False,
    ):
        """
        Crops the `Photo` at the bounding box.
        Saves the cropped image file to `out_filename`
        or returns as a stream of bytes if out_filename is None.

        Arg `padding` is a triple (x_padding, y_padding, percentage)
        that specifies the padding constants in the x and y directions,
        and whether they are absolute deltas or percentages.
        """
        source_filename = database._get_image_path(photo_id=self.photo_id)

        if not skip_padding:
            if custom_padding is not None:
                x_padding, y_padding, percentage = custom_padding
            else:
                x_padding, y_padding, percentage = (X_PADDING, Y_PADDING, False)

        # https://www.geeksforgeeks.org/python-pil-image-crop-method/
        im = Image.open(source_filename)

        if not skip_padding:
            padded_bbox = self.bounding_box.add_padding(
                im.width, im.height, x_padding, y_padding, percentage=percentage
            )
            cropped_image = im.crop(padded_bbox.bbox)
        else:
            cropped_image = im.crop(self.bounding_box.bbox)

        # We can use this for debugging.
        # cropped_image.show()

        if out_filename is not None:
            cropped_image.save(out_filename)
        else:
            # Source: https://aws.amazon.com/blogs/machine-learning/build-your-own-face-recognition-service-using-amazon-rekognition/
            stream = io.BytesIO()
            # Note: I removed 'format=cropped_image.format'

            # https://stackoverflow.com/questions/48248405/cannot-write-mode-rgba-as-jpeg/48248432
            failed = False
            for i in range(2):
                if i == 1 and not failed:
                    break
                try:
                    if "exif" in cropped_image.info:
                        exif = cropped_image.info["exif"]
                        # Does this reformat as JPEG if needed?
                        cropped_image.save(stream, format="JPEG", exif=exif)
                    else:
                        cropped_image.save(stream, format="JPEG")
                except Exception as e:
                    if i == 1:
                        raise e
                    failed = True
                    cropped_image = cropped_image.convert("RGB")

            if return_stream:
                return stream
            elif return_image:
                return cropped_image
            image_binary = stream.getvalue()
            if return_base64:
                return base64.b64encode(image_binary).decode("ascii")
            return image_binary

    def annotate(self, annotation):
        """
        Annotate the face and set its annotation flag to true.
        """
        self.annotation = annotation
        self.has_annotation = True

    def is_annotated(self):
        """
        Return if a face has an annotation.
        """
        return self.has_annotation

    def is_match_with_seed_person(self):
        """
        Return if a face is a match with the seed identity for which
        the source image was downloaded.
        """
        return self.has_annotation and self.annotation == MatchType.RIGHT_ID
