"""A torch general object detection dataset."""


from pathlib import Path
from typing import Dict, Union, Callable, Tuple, List

from numpy.typing import NDArray
from torch.utils.data import Dataset

from utils.image_utils.image_functions import read_image
from utils.cvat_utils.cvat_datasets import CvatObjectDetectionDataset


class ObjectDetectionDataset(Dataset):
    """A torch general object detection dataset."""

    def __init__(
        self,
        cvat_dset_dir: Union[str, Path],
        class_to_index: Dict[str, int] = None,
        transforms: Callable = None
    ):
        """Initialize object.

        Parameters
        ----------
        cvat_dset_dir : Union[str, Path]
            A path to a cvat dataset directory.
        class_to_index : Dict[str, int], optional
            Class name to class index converter.
            If not provided then will be generated automatically.
        transforms : Callable, optional
            Dataset transforms.
            It's expected that "Albumentations" lib will be used.
            By default is None.
        """
        if isinstance(cvat_dset_dir, str):
            cvat_dset_dir = Path(cvat_dset_dir)
        self.dset_dir = cvat_dset_dir
        self.image_dir = cvat_dset_dir / 'images'
        self.cvat_dset = CvatObjectDetectionDataset(cvat_dset_dir)
        self.labels = self.cvat_dset.get_labels()
        if class_to_index is None:
            self.class_to_index = {
                label: i for i, label in enumerate(self.labels)}
        else:
            self.class_to_index = class_to_index
        self.index_to_class = {
            idx: name for name, idx in self.class_to_index.items()}
        self.transforms = transforms

    def __len__(self):
        return len(self.cvat_dset)

    def __getitem__(
        self, idx: int
    ) -> Tuple[NDArray, List[List[float]], List[float], str, Tuple[int, int]]:
        """Return a sample by its index.

        Parameters
        ----------
        idx : int
            The index of the sample.

        Returns
        -------
        Tuple[NDArray, List[List[float]], List[float], str, Tuple[int, int]]
            By default sample is a tuple that contains
            an image ndarray with shape `(h, w, c)`,
            a bounding boxes float list with shape `(n_obj, 4)`,
            a classes float list with shape `(n_obj,)`,
            an image name string.
            and a tuple with original image shape.
        """
        sample = self.cvat_dset[idx]
        img_name = sample['name']
        classes = sample['labels']
        bboxes = sample['bboxes']
        shape = sample['shape']
        classes = list(map(lambda label: float(self.class_to_index[label]),
                           classes))
        img_pth = self.image_dir / img_name
        image = read_image(img_pth)

        if self.transforms:
            image, bboxes, classes = self.apply_transforms(
                image, bboxes, classes)
        return image, bboxes, classes, img_name, shape
    
    def apply_transforms(
        self, image: NDArray, bboxes: List[List[float]], classes: List[float]
    ) -> Tuple[NDArray, List[List[float]], List[float]]:
        """Apply transformations.

        Parameters
        ----------
        image : NDArray
            An image array with shape `(h, w, c)`.
        bboxes : List[List[float]]
            List of bounding boxes with shape `(n_boxes, 4)`.
        classes : List[float]
            Classes labels with shape `(n_boxes,)`.
        """
        transformed = self.transforms(
            image=image, bboxes=bboxes, classes=classes)
        image = transformed['image']  # ArrayLike or Tensor
        bboxes = transformed['bboxes']  # list[list[float]]
        classes = transformed['classes']  # list[float]
        return image, bboxes, classes
