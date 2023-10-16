"""A module containing some helpful functions working with Torch."""


from typing import List, Tuple, Union

from numpy.typing import NDArray
import cv2
from torch import Tensor, tensor
from torchvision.ops import box_convert


def image_tensor_to_numpy(tensor: Tensor) -> NDArray:
    """Convert an image or a batch of images from tensor to ndarray.

    Parameters
    ----------
    tensor : Tensor
        The tensor with shape `(h, w)`, `(c, h, w)` or `(b, c, h, w)`.

    Returns
    -------
    NDArray
        The array with shape `(h, w)`, `(h, w, c)` or `(b, h, w, c)`.
    """
    if len(tensor.shape) == 3:
        return tensor.detach().cpu().permute(1, 2, 0).numpy()
    elif len(tensor.shape) == 4:
        return tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
    elif len(tensor.shape) == 2:
        return tensor.detach().cpu().numpy()


def draw_bounding_boxes(
    image: NDArray,
    bboxes: List[List[Union[float, int]]],
    class_labels: List[Union[str, int, float]] = None,
    confidences: List[float] = None,
    bbox_format: str = 'xyxy',
    line_width: int = 1,
    color: Tuple[int, int, int] = (255, 255, 255),
    exclude_classes: List[Union[str, int, float]] = None
) -> NDArray:
    """Draw bounding boxes and corresponding labels on a given image.

    Parameters
    ----------
    image : NDArray
        The given image with shape `(h, w, c)`.
    bboxes : List[List[Union[float, int]]]
        The bounding boxes with shape `(n_boxes, 4)`.
    class_labels : List, optional
        Bounding boxes' labels. By default is None.
    exclude_classes : List[str, int, float]
        Classes which bounding boxes won't be showed. By default is None.
    confidences : List, optional
        Bounding boxes' confidences. By default is None.
    bbox_format : str, optional
        A bounding boxes' format. It should be one of "xyxy", "xywh" or
        "cxcywh". By default is 'xyxy'.
    line_width : int, optional
        A width of the bounding boxes' lines. By default is 1.
    color : Tuple[int, int, int], optional
        A color of the bounding boxes' lines. By default is `(255, 255, 255)`.

    Returns
    -------
    NDArray
        The image with drawn bounding boxes.

    Raises
    ------
    NotImplementedError
        Implemented only for "xyxy", "xywh" and "cxcywh"
        bounding boxes formats.
    """
    image = image.copy()
    if exclude_classes is None:
        exclude_classes = []

    # Convert to "xyxy"
    if bbox_format != 'xyxy':
        if bbox_format in ('xywh', 'cxcywh'):
            bboxes = box_convert(tensor(bboxes), bbox_format, 'xyxy').tolist()
        else:
            raise NotImplementedError(
                'Implemented only for "xyxy", "xywh" and "cxcywh"'
                'bounding boxes formats.')
    
    for i, bbox in enumerate(bboxes):
        # Check if exclude
        if class_labels is not None and class_labels[i] in exclude_classes:
            continue

        # Draw bbox
        bbox = list(map(int, bbox))
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2),
                      color=color, thickness=line_width)
        
        # Put text if needed
        if class_labels is not None:
            put_text = f'cls: {class_labels[i]} '
        else:
            put_text = ''
        if confidences is not None:
            put_text += 'conf: {:.2f}'.format(confidences[i])
        if put_text != '':
            cv2.putText(image, put_text, (x1, y1 - 2), 0, 0.3,
                        (255, 255, 255), 1)
    return image
