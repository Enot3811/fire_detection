"""Helper-функции для RCNN."""

from typing import Tuple, List, Dict

import torch
from torch import FloatTensor, Tensor
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def rcnn_collate_fn(
    batch: List[Tuple[
        FloatTensor,
        List[Tuple[float, float, float, float]],
        List[float],
        str,
        Tuple[int, int]
    ]]
) -> Tuple[
        List[FloatTensor],
        List[Dict[str, Tensor]],
        List[Tuple[str]],
        List[Tuple[int, int]]]:
    """Привести пакет к формату torch FasterRCNN.

    Изображения должны быть в списке, при этом каждое -
    это ``FloatTensor[C, H, W]``, где H, W могут варьироваться.
    Каждое изображение отмасштабировано в диапазон от 0 до 1.
    Метки должны быть представлены в виде списка словарей,
    где "boxes" - это ``FloatTensor[N, 4]`` в формате "xyxy",
    а "labels" - это ``Int64Tensor[N]`` с метками классов.

    Метки классов обязательно должны содержать 0-й "background" класс.

    Parameters
    ----------
    batch : List[Tuple[FloatTensor,
                       List[Tuple[float, float, float, float]],
                       List[float],
                       str,
                       Tuple[int, int] ]]
        Пакет из ObjectDetectionDataset.

    Returns
    -------
    Tuple[List[FloatTensor],
          List[Dict[str, Tensor]],
          List[Tuple[str]],
          List[Tuple[int, int]]]
        Отформатированный пакет.
    """
    images, boxes, classes, image_names, image_sizes = zip(*batch)
    images = list(images)
    image_names = list(image_names)
    image_sizes = list(image_sizes)

    targets = [{'boxes': torch.stack(list(map(lambda box: torch.tensor(box),
                                              img_boxes))),
                'labels': torch.tensor(img_classes, dtype=torch.int64)}
               for img_boxes, img_classes in zip(boxes, classes)]
    return images, targets, image_names, image_sizes


def get_rcnn_model(num_classes: int) -> FasterRCNN:
    model = (torchvision.models.detection
             .fasterrcnn_resnet50_fpn_v2(weights='DEFAULT'))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
