"""The base classes for custom datasets for object detection.

This classes provide a base API for every custom datasets with its formats.
The main goal of these classes is extracting data from any format and packing
it to CVAT datasets format.
"""


from pathlib import Path
import sys
from typing import List, Optional, Dict, Union
import shutil

from numpy.typing import NDArray

sys.path.append(str(Path(__file__).parents[3]))
from utils.image_utils.image_functions import read_image
from utils.torch_utils.torch_functions import draw_bounding_boxes
from utils.cvat_utils.cvat_functions import create_cvat_object_detection_xml


class BaseObjectDetectionAnnotation:
    """The base annotation class for object detection.

    It consists of 4 points of bounding box and a class label.
    """
    def __init__(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        label: str
    ) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.label = label


class BaseObjectDetectionSample:
    """The base sample class for object detection.

    It consists of a path to an image and a list of annotations.
    """
    def __init__(
        self, img_pth: Path, img_annots: List[BaseObjectDetectionAnnotation]
    ) -> None:
        self._img_pth = img_pth
        self._img_annots = img_annots

    def get_image_path(self) -> Path:
        """Get a source image's path.

        Returns
        -------
        Path
            The source image's path.
        """
        return self._img_pth

    def get_image(self) -> NDArray:
        """Get source image of this sample.

        Returns
        -------
        NDArray
            The source image of this sample.
        """
        return read_image(self._img_pth)
    
    def get_image_with_bboxes(self) -> NDArray:
        """Get this sample's image with showed bounding boxes.

        Returns
        -------
        NDArray
            The image with bounding boxes.
        """
        img = self.get_image()
        bboxes = list(map(
            lambda annot: (annot.x1, annot.y1, annot.x2, annot.y2),
            self._img_annots))
        labels = list(map(lambda annot: annot.label, self._img_annots))
        return draw_bounding_boxes(img, bboxes, labels)
    
    def get_annotations(self) -> List[BaseObjectDetectionAnnotation]:
        """Get annotations of this sample.

        Returns
        -------
        List[BaseAnnotation]
            The annotations of this sample.
        """
        return self._img_annots


class BaseObjectDetectionDataset:
    """The base dataset class for object detection.

    It consists of subsets that represented as base samples lists
    and of a all possible labels str list.
    """
    def __init__(self, dset_folder: Union[str, Path]) -> None:
        if isinstance(dset_folder, str):
            self.dset_folder = Path(dset_folder)
        else:
            self.dset_folder = dset_folder
        self._subsets: Dict[List[BaseObjectDetectionSample]] = {}
        self._labels: Optional[List[str]] = None

    def __getitem__(self, set_name: str) -> List[BaseObjectDetectionSample]:
        if set_name not in self._subsets:
            raise KeyError(f'Available sets: {list(self._subsets.keys())}.')
        else:
            return self._subsets[set_name]
        
    def get_subsets_names(self) -> List[str]:
        """Get names of all subsets.

        Returns
        -------
        List[str]
            A list containing the names of the subsets.
        """
        return list(self._subsets.keys())
        
    def get_labels_names(self) -> List[str]:
        """Get all labels names from this dataset.

        First call of this function may take some time.

        Returns
        -------
        List[str]
            List of labels names.
        """
        if self._labels is not None:
            return self._labels
        # Iterate over whole dataset and collect all different labels
        labels = set()
        for subset in self._subsets.values():
            for sample in subset:
                for annot in sample.get_annotations():
                    labels.add(annot.label)
        return list(labels)
        
    def save_as_cvat(
        self, save_pth: Path, verbose: bool = False, copy_images: bool = False
    ):
        """Save the dataset in CVAT format to a specified directory.

        Parameters
        ----------
        save_pth : Path
            A path to the save directory.
        verbose : bool, optional
            Whether to show progress of saving. By default is `False`.
        copy_images : bool, optional
            Whether to copy images from original dataset to new CVAT.
            By default is `False`.
        """
        if isinstance(save_pth, str):
            save_pth = Path(save_pth)
        labels = self.get_labels_names()
        for subset_name, subset in self._subsets.items():
            subset_dir = save_pth / subset_name
            
            images_pth = subset_dir / 'images'
            annot_pth = subset_dir / 'annotations.xml'
            images_pth.mkdir(parents=True, exist_ok=True)

            create_cvat_object_detection_xml(annot_pth, subset, subset_name,
                                             labels, verbose)
            if copy_images:
                for sample in subset:
                    src_pth = sample.get_image_path()
                    dst_pth = images_pth / src_pth.name
                    shutil.copy2(src_pth, dst_pth)
