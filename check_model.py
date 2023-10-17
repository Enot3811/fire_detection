"""Проверка модели."""

import argparse
import json
from pathlib import Path

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.torch_utils.datasets import ObjectDetectionDataset
from rcnn_utils import rcnn_collate_fn, get_rcnn_model


def main(train_config: Path):
    # Read train config
    with open(train_config, 'r') as f:
        config_str = f.read()
    config = json.loads(config_str)

    # Prepare some stuff
    torch.manual_seed(config['random_seed'])

    if config['device'] == 'auto':
        device: torch.device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
    else:
        device: torch.device = torch.device(config['device'])

    # Get transforms
    train_transform = A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['classes'])
    )
    val_transform = A.Compose(
        [
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2()
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['classes'])
    )

    # Get datasets and loaders
    train_dir = Path(config['dataset']) / 'train'
    val_dir = Path(config['dataset']) / 'val'

    train_dset = ObjectDetectionDataset(train_dir,
                                        class_to_index=config['cls_to_id'],
                                        transforms=train_transform)
    val_dset = ObjectDetectionDataset(val_dir,
                                      class_to_index=config['cls_to_id'],
                                      transforms=val_transform)

    train_dloader = DataLoader(train_dset,
                               batch_size=config['batch_size'],
                               shuffle=config['shuffle_train'],
                               collate_fn=rcnn_collate_fn)
    val_dloader = DataLoader(val_dset,
                             batch_size=config['batch_size'],
                             shuffle=config['shuffle_val'],
                             collate_fn=rcnn_collate_fn)
    
    # Get the model
    model = get_rcnn_model(config['num_classes'])
    
    # Train step
    images, targets, img_names, img_sizes = next(iter(train_dloader))
    losses = model(images, targets)
    print(losses)

    # Val step
    model.eval()
    images, targets, img_names, img_sizes = next(iter(val_dloader))
    predictions = model(images)
    print(predictions)


def parse_args() -> argparse.Namespace:
    """Create & parse command-line args."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'train_config', type=Path,
        help='Путь к файлу с конфигом обучения.')
    args = parser.parse_args(['configs/train_1.json'])
    return args


if __name__ == '__main__':
    args = parse_args()
    train_config = args.train_config
    main(train_config)
