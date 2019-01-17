import random
import argparse
import numpy as np
import os
import logging

from yolo.model.darknet import Darknet
from yolo.util.util import *
import yolo.transforms.preprocessing as pr
from yolo.dataset.BrainIOIDataset import BrainIOIDataset, IOIDatasetETips
import yolo.model.loss as ls
import yolo.model.FocalLoss as fl
from yolo.util.metrics import StatCollectorTrain
from ssd.utils.misc import str2bool, SavePointManager


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Dataset directory path')
parser.add_argument('--csv', default='stimulation.csv', help='Filename of annotation csv')
parser.add_argument('--gt', default='area_tips', help="Choose ground truth type 'area_tips', 'one_tip', 'two_tips'")
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--val_csv', default='stimulation.csv', help='Filename of annotation csv for valaidation')
parser.add_argument('--border', default=20, type=float,
                    help='Set border attribute for ground truth. (area_tips uses absolute pixel value, for '
                         'the others it\'s the percentage of the distance between tips e.g 0.5 for 50% of the distance )')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs')
parser.add_argument('-cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
parser.add_argument("--weights", dest = 'weightsfile', help ="weightsfile", type = str)
parser.add_argument('-resume', default=False, help='resume training flag')
parser.add_argument('--log', help='Path of log file. If log is set the logging will be activated else '
                                  'nothing will be logged.')
parser.add_argument('--use_focal_loss', default=False, type=str2bool,
                    help='Use focal loss for classification')


args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
rd_seed = 456
random.seed(rd_seed)
np.random.seed(rd_seed)
torch.manual_seed()
if cuda:
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)


def train(loader, model, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    i = 0
    net_info = model.net_info
    anchors = net_info['anchors']
    num_anchors = net_info['num_anchors']
    det_strides = [32, 16, 8]
    metrics = StatCollectorTrain(416, anchors, det_strides=det_strides, conf_threshold=0.5, nms_threshold=0.0,
                                 iou_threshold=0.5, device=device)
    for _, data in enumerate(loader):
        optimizer.zero_grad()
        output = model(data[0].to(device))
        ground_truth = data[1].to(device)

        metrics(output, ground_truth)
        loss = criterion(output, ground_truth)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        i += 1

    return running_loss / i, metrics.precision(), metrics.recall()


def test(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    i = 0
    net_info = model.net_info
    anchors = net_info['anchors']
    num_anchors = net_info['num_anchors']
    metrics = StatCollectorTrain(416, anchors, conf_threshold=0.5, nms_threshold=0.0, iou_threshold=0.5, device=device)
    for _, data in enumerate(loader):
        with torch.no_grad():
            output = model(data[0].to(device), is_test=True)
            ground_truth = data[1].to(device)
            metrics(output, ground_truth)
            loss = criterion(ground_truth, output)
            running_loss += loss.item()

        i += 1

    model.train()
    return running_loss / i, metrics.precision(), metrics.recall()


if __name__ == '__main__':
    if args.log:
        logging.basicConfig(
            filename=args.log,
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s")

    # Initialize model
    model = Darknet(args.cfg)
    if args.weightfile:
        model.load_weights(args.weightsfile, stop=74)
        logging.info("Loaded original darknet weights.")

    model.to(device)
    model.train()
    num_classes = 1

    net_options = model.net_info

    ##Parse the config file
    batch_size = net_options['batch']
    num_input = net_options['channels']
    momentum = net_options['momentum']
    decay = net_options['decay']  # Penalty for regularisation
    learning_rate = net_options['learning_rate']  # Initial learning rate

    train_transform = pr.TrainAugmentation(args.image_size, p=0.5)
    test_transform = pr.TestTransform(args.image_size)
    target_transform = pr.MatchAnchors(net_options['anchors'], net_options['num_anchors'], inp_dim=net_options['width'],
                                       strides=[32, 16, 8], num_classes=1)
    logging.info("Prepare training dataset.")
    if args.gt == 'area_tips':
        train_dataset = BrainIOIDataset(os.path.join(args.dataset, args.csv), args.dataset, border=args.border,
                                        num_channels=num_input, transform=train_transform,
                                        target_transform=target_transform)
        val_dataset = BrainIOIDataset(os.path.join(args.validation_dataset, args.val_csv), args.validation_dataset,
                                      num_channels=num_input, border=args.border, transform=test_transform,
                                      target_transform=target_transform)
    else:
        if args.gt == 'one_tip':
            use_all = False
        else:  # 'two_tips'
            use_all = True

        train_dataset = IOIDatasetETips(os.path.join(args.dataset, args.csv), args.dataset, use_all=use_all,
                                        border=args.border, num_channels=num_input, transform=train_transform,
                                        target_transform=target_transform)
        val_dataset = IOIDatasetETips(os.path.join(args.validation_dataset, args.val_csv), args.validation_dataset, use_all=use_all,
                                      border=args.border, num_channels=num_input, transform=test_transform,
                                      target_transform=target_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers, shuffle=False)

    criterion = ls.YOLOLoss(delta_coord=net_options['delta_coord'], delta_obj=net_options['delta_obj'],
                            delta_noobj=net_options['delta_noobj'], delta_cls=net_options['delta_cls'])
    if args.use_focal_loss:
        criterion = fl.FocalLoss(gamma=net_options['gamma'], alpha=net_options['alpha'])

    # Reload saved optimizer state
    start_epoch = 0
    best_loss = float('inf')

        # Set optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
    chpts = SavePointManager(args.checkpoint_folder, args.max_chpt)
    for epoch in range(args.num_epochs):
        scheduler.step()
        logging.info(
            f"Epoch: {epoch}, " +
            f"learning rate: {scheduler.get_lr()[0]}, "
        )
        loss, precision, recall = train(train_loader, model, criterion, optimizer)

        logging.info(
            f"Epoch: {epoch}, " +
            f"Loss: {loss:.4f}, " +
            f"Precision: {precision:.3f}, " +
            f"Recall: {recall:.3f}"
        )

        val_loss, precision, recall = test(val_loader, model, criterion,device)
        logging.info(
            f"Epoch: {epoch}, " +
            f"Validation Loss: {val_loss:.4f}, " +
            f"Validation Precision: {precision:.3f}, " +
            f"Validation Recall: {recall:.3f}"
        )
        model.train()
        if torch.cuda.device_count() > 1:
            state = {'epoch': epoch, 'state_dict': model.module.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}
        else:
            state = {'epoch': epoch, 'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}

        model_name = f"ssd-Epoch-{epoch}-Loss-{val_loss}.pth"
        if chpts.save(state, model_name, val_loss):
            logging.info(f"Saved model {model_name}")


