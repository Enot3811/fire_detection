"""The modula that contains metrics calculating functions."""


from typing import List

import torch
from torch import FloatTensor
from torchvision.ops import box_iou


def calculate_iou(
    gt_boxes: FloatTensor, predicted_boxes: List[FloatTensor]
) -> FloatTensor:
    """Calculate IoU over predictions and ground truth bounding boxes.

    Parameters
    ----------
    gt_boxes : FloatTensor
        Ground truth bounding boxes with shape `(n_img, n_max_obj, 4)`.
    predicted_boxes : List[FloatTensor]
        Proposals list with length `n_img`
        and each element has shape `(n_props_per_img, 4)`.

    Returns
    -------
    FloatTensor
        Calculated IoU scalar value.
    """
    # Calculate n objects in gt boxes
    n_objs = (gt_boxes >= 0).any(dim=2).sum(dim=1)
    iou_list = []
    for i, n_obj in enumerate(n_objs):
        iou = box_iou(gt_boxes[i][:n_obj], predicted_boxes[i])
        # Check is there a prediction
        if iou.shape[0] != 0 and iou.shape[1] != 0:
            iou, _ = iou.max(dim=1)
            iou = torch.mean(iou)
        else:
            iou = torch.zeros((1,), dtype=torch.float32, device=iou.device)
        iou_list.append(iou)
    iou = sum(iou_list) / len(iou_list)
    return iou


def calculate_prediction_count_diff(
    gt_boxes: FloatTensor, predicted_boxes: List[FloatTensor]
) -> FloatTensor:
    """Calculate difference between predictions count and gt count.

    Parameters
    ----------
    gt_boxes : FloatTensor
        The ground truth bounding boxes.
    predicted_boxes : List[FloatTensor]
        The predicted bounding boxes.

    Returns
    -------
    FloatTensor
        Mean difference between counts of predicts and ground truth bboxes.
    """
    # TODO сделать описание получше
    gt_objs = (gt_boxes >= 0).any(dim=2).sum(dim=1, dtype=torch.float32)
    pred_objs = torch.tensor(list(map(len, predicted_boxes)),
                             dtype=torch.float32, device=gt_objs.device)
    diff = torch.mean(torch.abs(gt_objs - pred_objs))
    return diff
