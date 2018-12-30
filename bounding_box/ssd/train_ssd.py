import argparse
import os
import logging
import sys
import itertools
import random
import numpy

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from ssd.model.var_ssd import MatchPrior, VarSSD
from ssd.utils.misc import str2bool, Timer, freeze_net_layers

from ssd.dataset.BrainIOIDataset import BrainIOIDataset, IOIDatasetETips
from ssd.transforms.preprocessing import TrainAugmentation, TestTransform

import ssd.transforms.transforms as tr

from ssd.model.multibox_loss import MultiboxLoss
from ssd.utils.metrics import StatCollector

from sampler import ImbalancedDatasetSampler

import importlib.util


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')


parser.add_argument('--dataset', help='Dataset directory path')
parser.add_argument('--gt', default='area_tips', help="Choose ground truth type 'area_tips', 'one_tip', 'two_tips'")
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")

parser.add_argument('--log', help='Path of log file. If log is set the logging will be activated else '
                                  'nothing will be logged.')
parser.add_argument('--num_channels', default=2, type=int,
                    help='Set number of input channels to use for network and data')
parser.add_argument('--border', default=20, type=float,
                    help='Set border attribute for ground truth. (area_tips uses absolute pixel value, for '
                         'the others it\'s the percentage of the distance between tips e.g 0.5 for 50% of the distance )')

parser.add_argument('--random_seed', default=456, type=float,
                    help='Initialize random generator with fixed value to reproduce results')
parser.add_argument('--prob_aug', default=0.5, type=float,
                    help='Probability for data augmentations')

parser.add_argument('--use_mean', default=False, type=str2bool,
                    help='Use normalizatian via mean and standard deviation')

parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')

parser.add_argument('--config',
                    help='Configuration file with priors and other needed values.')

args = parser.parse_args()

if args.random_seed:
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    numpy.random.seed(args.random_seed)

if args.config:
    spec = importlib.util.spec_from_file_location("module.config",args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
else:
    import ssd.config.mobilenetv1_ssd_config as config

DEVICE = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def load_checkpoint(model, optimizer, scheduler, filename):
    # Note: Input model & optimizer, scheduler should be pre-defined.  This routine only updates their states.
    checkpoint = torch.load(filename)
    start_epoch = checkpoint['epoch']
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, start_epoch, scheduler


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1, alpha=1.0):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    count = 0
    collector = StatCollector(300, config)
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        collector(confidence, locations, labels, boxes)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + alpha * classification_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

        count += 1

    avg_loss = running_loss / count
    avg_reg_loss = running_regression_loss / count
    avg_clf_loss = running_classification_loss / count
    logging.info(
        f"Epoch: {epoch}, Step: {i}, " +
        f"Average Loss: {avg_loss:.4f}, " +
        f"Average Regression Loss {avg_reg_loss:.4f}, " +
        f"Average Classification Loss: {avg_clf_loss:.4f}" +
        f"Precision: {collector.precision():.3f}" +
        f"Recall: {collector.recall():.3f}"
    )


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    collector = StatCollector(300, config)
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            collector(confidence, locations, labels, boxes)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num, collector.precision(), collector.recall()


if __name__ == '__main__':
    if args.log:
        logging.basicConfig(
            filename=args.log,
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s")

    timer = Timer()
    num_input = args.num_channels

    normalization = tr.Normalize(2 ** 12 - 1)
    if args.use_mean:
        normalization = tr.NormalizeMean(config.image_mean, config.image_std)

    train_transform = TrainAugmentation(config.image_size, normalization, background_color=int(config.mean), p=args.prob_aug)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, config.iou_threshold)

    test_transform = TestTransform(config.image_size, normalization)

    logging.info("Prepare training dataset.")
    if args.gt == 'area_tips':
        train_dataset = BrainIOIDataset(os.path.join(args.dataset, 'stimulation.csv'), args.dataset, border=args.border,
                                        num_channels=num_input, transform=train_transform,
                                        target_transform=target_transform)
        val_dataset = BrainIOIDataset(os.path.join(args.validation_dataset, 'stimulation.csv'), args.validation_dataset,
                                      num_channels=num_input, border=args.border, transform=test_transform,
                                      target_transform=target_transform)
    else:
        if args.gt == 'one_tip':
            use_all = False
        else:  # 'two_tips'
            use_all = True

        train_dataset = IOIDatasetETips(os.path.join(args.dataset, 'stimulation.csv'), args.dataset, use_all=use_all,
                                        border=args.border, num_channels=num_input, transform=train_transform,
                                        target_transform=target_transform)
        val_dataset = IOIDatasetETips(os.path.join(args.validation_dataset, 'stimulation.csv'), args.validation_dataset, use_all=use_all,
                                      border=args.border, num_channels=num_input, transform=test_transform,
                                      target_transform=target_transform)

    logging.info("Train dataset size: {}".format(len(train_dataset)))
    if args.balance_data:
        train_loader = DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  sampler=ImbalancedDatasetSampler(train_dataset))
    else:
        train_loader = DataLoader(train_dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True)

    logging.info("validation dataset size: {}".format(len(val_dataset)))
    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    num_classes = len(train_dataset.classes)
    logging.info("Build network.")
    net = VarSSD(num_classes, config=config, input_channels=num_input)
    min_loss = -10000.0
    last_epoch = -1

    if args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net.to(DEVICE)

    criterion = MultiboxLoss(iou_threshold=config.iou_threshold, neg_pos_ratio=config.neg_pos_ratio,
                             center_variance=config.center_variance, size_variance=config.size_variance, device=DEVICE,
                             weights=config.weights)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        model, optimizer, last_epoch, scheduler = load_checkpoint(net, optimizer, scheduler, args.resume)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        scheduler.step()
        logging.info(
            f"Epoch: {epoch}, " +
            f"learning rate: {scheduler.get_lr()[0]}, "
        )
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch, alpha=config.alpha)
        
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss, precision, recall = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}" +
                f"Validation Precision: {precision:.3f}" +
                f"Validation Recall: {recall:.3f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"ssd-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.train()
            if torch.cuda.device_count() > 1:
                state = {'epoch': epoch, 'state_dict': net.module.state_dict(),
                         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            else:
                state = {'epoch': epoch, 'state_dict': net.state_dict(),
                         'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
            torch.save(state, model_path)
            logging.info(f"Saved model {model_path}")
