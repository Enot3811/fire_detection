"""Helper-функции для RCNN."""

from typing import Tuple, List, Dict

import torch
from torch import FloatTensor, Tensor
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics import Metric


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


class RcnnLosses(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state('loss_classifier',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')
        self.add_state('loss_box_reg',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')
        self.add_state('loss_objectness',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')
        self.add_state('loss_rpn_box_reg',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')
        self.add_state('n_total',
                       default=torch.tensor(0, dtype=torch.float64),
                       dist_reduce_fx='sum')
    
    def update(self, losses_dict: Dict[str, FloatTensor]):
        self.loss_classifier += losses_dict['loss_classifier']
        self.loss_box_reg += losses_dict['loss_box_reg']
        self.loss_objectness += losses_dict['loss_objectness']
        self.loss_rpn_box_reg += losses_dict['loss_rpn_box_reg']
        self.n_total += 1

    def compute(self) -> Dict[str, FloatTensor]:
        self.loss_classifier /= self.n_total
        self.loss_box_reg /= self.n_total
        self.loss_objectness /= self.n_total
        self.loss_rpn_box_reg /= self.n_total
        total_loss = (self.loss_classifier + self.loss_box_reg +
                      self.loss_objectness + self.loss_rpn_box_reg)
        return {
            'total_loss': total_loss,
            'loss_classifier': self.loss_classifier,
            'loss_box_reg': self.loss_box_reg,
            'loss_objectness': self.loss_objectness,
            'loss_rpn_box_reg': self.loss_rpn_box_reg
        }
