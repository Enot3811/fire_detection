"""Merge several object detection CVAT datasets into one."""


import argparse
from pathlib import Path
import sys
from typing import List, Set
import shutil

sys.path.append(str(Path(__file__).parents[3]))
from utils.cvat_utils.cvat_datasets import CvatObjectDetectionDataset
from utils.data_utils.datasets import (
    BaseObjectDetectionAnnotation, BaseObjectDetectionSample)
from utils.cvat_utils.cvat_functions import create_cvat_object_detection_xml


class TempSample(BaseObjectDetectionSample):
    """Temporary sample class with additional path to save image."""
    def __init__(
        self,
        img_pth: Path,
        img_annots: List[BaseObjectDetectionAnnotation],
        new_pth: Path
    ):
        super().__init__(img_pth, img_annots)
        self.new_pth = new_pth
    
    def get_image_path(self, new_pth: bool = True) -> Path:
        if new_pth:
            return self.new_pth
        else:
            return super().get_image_path()


def main(**kwargs):
    datasets_dirs = kwargs['datasets_dirs']
    save_dir = kwargs['save_dir']

    union_images_pth = save_dir / 'images'
    union_annots_pth = save_dir / 'annotations.xml'

    # Collect all samples and class labels
    samples: List[TempSample] = []
    set_classes: Set[str] = set()
    sample_counter = 0
    for dset_dir in datasets_dirs:
        img_dir = dset_dir / 'images'
        dset = CvatObjectDetectionDataset(dset_dir)
        for sample in dset:
            img_name = sample['name']
            classes = sample['labels']
            bboxes = sample['bboxes']

            img_annots: List[BaseObjectDetectionAnnotation] = []
            for cls, bbox in zip(classes, bboxes):
                img_annots.append(BaseObjectDetectionAnnotation(
                    bbox[0], bbox[1], bbox[2], bbox[3], cls))
                set_classes.add(cls)

            img_pth = img_dir / img_name
            ext = img_name.split('.')[-1]
            new_img_pth = union_images_pth / f'{sample_counter}.{ext}'
            sample_counter += 1
            samples.append(TempSample(img_pth, img_annots, new_img_pth))

    # Create CVAT xml
    set_classes = list(set_classes)
    union_images_pth.mkdir(parents=True, exist_ok=True)
    create_cvat_object_detection_xml(
        union_annots_pth, samples, 'train', set_classes, verbose=True)
    # Copy images
    for sample in samples:
        src_pth = sample.get_image_path(new_pth=False)
        dst_pth = sample.get_image_path(new_pth=True)
        shutil.copy2(src_pth, dst_pth)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'datasets_dirs', type=Path, nargs='+',
        help='Paths to CVAT datasets to merge.')
    parser.add_argument(
        'save_dir', type=Path,
        help='A path to save the merged CVAT dataset.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    datasets_dirs = args.datasets_dirs
    save_dir = args.save_dir
    main(datasets_dirs=datasets_dirs,
         save_dir=save_dir)
