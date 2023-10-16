"""Object detection dataset in CVAT format."""


from typing import List, Dict, Union, Any, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET


class CvatObjectDetectionDataset:
    """Object detection dataset in CVAT format."""

    def __init__(self, dset_pth: Union[Path, str]) -> None:
        if isinstance(dset_pth, str):
            self.dset_pth = Path(dset_pth)
        else:
            self.dset_pth = dset_pth
        annots = ET.parse(self.dset_pth / 'annotations.xml').getroot()
        
        # Get labels
        self._label_to_color = {}
        self._labels = []
        labels_annots = annots.findall('meta/job/labels/label')
        for i, label_annot in enumerate(labels_annots):
            name = label_annot.find('name').text
            hex_color = label_annot.find('color').text
            color = [int(hex_color[j:j + 2], 16)
                     for j in range(1, len(hex_color), 2)]
            self._label_to_color[name] = color
            self._labels.append(name)

        # Get images
        self.samples: List[Dict[str, Any]] = []
        imgs_annots = annots.findall('image')
        for img_annots in imgs_annots:
            name = img_annots.get('name')
            shape = (int(img_annots.get('height')),
                     int(img_annots.get('width')))
            img_bboxes = img_annots.findall('box')
            img_labels: List[str] = []
            img_bboxes_pts: List[Tuple[float, float, float, float]] = []
            for bbox in img_bboxes:
                label = bbox.get('label')
                img_labels.append(label)
                x1 = float(bbox.get('xtl'))
                y1 = float(bbox.get('ytl'))
                x2 = float(bbox.get('xbr'))
                y2 = float(bbox.get('ybr'))
                img_bboxes_pts.append((x1, y1, x2, y2))
            self.samples.append({
                'name': name,
                'labels': img_labels,
                'bboxes': img_bboxes_pts,
                'shape': shape
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
    
    def get_labels(self) -> List[str]:
        """Get a list of dataset's labels.

        Returns
        -------
        List[str]
            The list of the labels.
        """
        return self._labels

    def get_labels_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Get labels with corresponding colors.

        Returns
        -------
        Dict[str, Tuple[int, int, int]]
            Dict that contains labels as keys and "color" as values.
        """
        return self._label_to_color
