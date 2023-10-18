import argparse
import json
from pathlib import Path

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.detection import MeanAveragePrecision

from utils.torch_utils.datasets import ObjectDetectionDataset
from rcnn_utils import rcnn_collate_fn, get_rcnn_model, RcnnLosses


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
    cpu_device = torch.device('cpu')

    work_dir = Path(config['work_dir'])
    tensorboard_dir = work_dir / 'tensorboard'
    ckpt_dir = work_dir / 'ckpts'
    if not config['continue_training']:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Check and load checkpoint
    if config['continue_training']:
        checkpoint = torch.load(ckpt_dir / 'last_checkpoint.pth')
        model_params = checkpoint['model']
        optim_params = checkpoint['optimizer']
        lr_params = checkpoint['lr_scheduler']
        start_ep = checkpoint['epoch']
    else:
        model_params = None
        optim_params = None
        lr_params = None
        start_ep = 0

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

    # Get tensorboard
    log_writer = SummaryWriter(str(tensorboard_dir))

    # Get datasets and loaders
    train_dir = Path(config['dataset']) / 'train'
    val_dir = Path(config['dataset']) / 'val'

    train_dset = ObjectDetectionDataset(train_dir, transforms=train_transform)
    val_dset = ObjectDetectionDataset(val_dir, transforms=val_transform)
    # from torch.utils.data import random_split
    # train_dset, _ = random_split(train_dset, [0.02, 0.98])
    # val_dset, _ = random_split(val_dset, [0.02, 0.98])

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
    model.to(device=device)
    if model_params:
        model.load_state_dict(model_params)

    # Get an optimizer and scheduler
    optimizer = optim.Adam(model.parameters(),
                           lr=config['lr'],
                           weight_decay=config['weight_decay'])
    if optim_params:
        optimizer.load_state_dict(optim_params)
    
    if lr_params:
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=config['n_epoch'], eta_min=1e-6,
            last_epoch=start_ep - 1)
        lr_scheduler.load_state_dict(lr_params)
    else:
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=config['n_epoch'], eta_min=1e-6, last_epoch=-1)
    
    # Get the metrics
    train_losses_metric = RcnnLosses()
    train_losses_metric.to(device=device)
    map_metric = MeanAveragePrecision(extended_summary=True)
    map_metric.to(device=device)
    
    # Do training
    max_map = 0.0
    for epoch in range(start_ep, config['n_epoch']):

        print(f'Epoch {epoch + 1}')

        # Train
        model.train()
        for batch in tqdm(train_dloader, 'Train step'):
            images, targets, img_names, img_sizes = batch
            # To device
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()}
                       for t in targets]
            
            # Compute step
            losses = model(images, targets)
            total_loss = sum(loss for loss in losses.values())
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Log train metrics
            train_losses_metric.update(losses)

        # Validation
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_dloader, 'Val step'):
                images, targets, img_names, img_sizes = batch
                # To device
                images = list(image.to(device) for image in images)
                targets = [
                    {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in t.items()}
                    for t in targets]
                # Compute step
                predictions = model(images)

                # Log val metrics
                map_metric.update(predictions, targets)

        # Lr scheduler
        lr_scheduler.step()
        lr = lr_scheduler.get_last_lr()[0]

        # Log epoch metrics
        train_losses_dict = train_losses_metric.compute()
        train_losses_dict = {
            k: v.to(cpu_device)
            for k, v in train_losses_dict.items()}
        train_losses_metric.reset()
        log_writer.add_scalar('total_loss/train',
                              train_losses_dict['total_loss'],
                              epoch)
        log_writer.add_scalar('loss_classifier/train',
                              train_losses_dict['loss_classifier'],
                              epoch)
        log_writer.add_scalar('loss_box_reg/train',
                              train_losses_dict['loss_box_reg'],
                              epoch)
        log_writer.add_scalar('loss_objectness/train',
                              train_losses_dict['loss_objectness'],
                              epoch)
        log_writer.add_scalar('loss_rpn_box_reg/train',
                              train_losses_dict['loss_rpn_box_reg'],
                              epoch)
        map_dict = map_metric.compute()
        map_metric.reset()
        log_writer.add_scalar('map/val',
                              map_dict['map'].to(device=cpu_device),
                              epoch)
        log_writer.add_scalar('map_small/val',
                              map_dict['map_small'].to(device=cpu_device),
                              epoch)
        log_writer.add_scalar('map_medium/val',
                              map_dict['map_medium'].to(device=cpu_device),
                              epoch)
        log_writer.add_scalar('map_large/val',
                              map_dict['map_large'].to(device=cpu_device),
                              epoch)

        log_writer.add_scalar('Lr', lr, epoch)

        print('TrainLoss:', train_losses_dict['total_loss'].item())
        print('ValMap:', map_dict['map'].item())
        print('Lr:', lr)

        # Save model
        if max_map < map_dict['map']:
            torch.save(
                model.state_dict(), ckpt_dir / 'best_model.pt')
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch + 1
            },
            ckpt_dir / 'last_checkpoint.pth')

    log_writer.close()


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
