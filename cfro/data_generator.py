import numpy as np
from PIL import Image
from pathlib import PurePath, Path
from typing import Tuple, List

from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.mobilenet import preprocess_input


class DataGenerator(Sequence):

    def __init__(
        self,
        image_dir: str,
        batch_size: int,
        target_size: Tuple[int, int],
    ) -> None:
        """Init DataGenerator object.
        """
        if isinstance(image_dir, str):
            self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.target_size = target_size

        self._get_image_files()
        self.indexes = np.arange(len(self.image_files))
        self.valid_image_files = self.image_files

    def _get_image_files(self) -> None:
        self.image_files = sorted(
            [
                i.absolute()
                for i in self.image_dir.glob('*')
                if not i.name.startswith('.')]
        )  # ignore hidden files

    def __len__(self) -> int:
        """Number of batches in the Sequence."""
        return int(np.ceil(len(self.image_files) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.array, np.array]:
        """Get batch at position `index`.
        """
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_samples = [self.image_files[i] for i in batch_indexes]
        X = self._data_generator(batch_samples)
        return X

    def _data_generator(
            self, image_files: List[PurePath]
    ) -> Tuple[np.array, np.array]:
        """Generate data from samples in specified batch."""
        #  initialize images and labels tensors for faster processing
        X = np.empty((len(image_files), *self.target_size, 3))

        invalid_image_idx = []
        for i, image_file in enumerate(image_files):
            try:
                image = Image.open(image_file)
                if image.mode != 'RGB':
                    # convert to RGBA first to avoid warning
                    # we ignore alpha channel if available
                    image = image.convert('RGBA').convert('RGB')

                if isinstance(image, np.ndarray):
                    image = image.astype('uint8')
                    image_pil = Image.fromarray(image)
                    image_pil = image_pil.resize(self.target_size, Image.LANCZOS)
                    img = np.array(image_pil).astype('uint8')
                elif isinstance(image, Image.Image):
                    image_pil = image.resize(self.target_size, Image.LANCZOS)
                    img = np.array(image_pil).astype('uint8')
                else:
                    img = None

            except Exception as e:
                img = None

            if img is not None:
                X[i, :] = img

            else:
                invalid_image_idx.append(i)
                self.valid_image_files = [_file for _file in self.valid_image_files if _file != image_file]

        if invalid_image_idx:
            X = np.delete(X, invalid_image_idx, axis=0)

        # apply basenet specific preprocessing
        # input is 4D numpy array of RGB values within [0, 255]
        X = preprocess_input(X)

        return X