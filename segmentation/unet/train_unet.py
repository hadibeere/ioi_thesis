import argparse
import logging
import random
import numpy as np
import importlib
import torch
from torch.utils.data import DataLoader
import os
from unet.model.unet import UNet
from ssd.utils.misc import SavePointManager

from unet.dataset.BrainIOIDataset import IOIDatasetETips

parser = argparse.ArgumentParser(
    description='U-Net Training With Pytorch')
parser.add_argument('--dataset', help='Dataset directory path')
parser.add_argument('--csv', default='stimulation.csv', help='Filename of annotation csv')
parser.add_argument('--gt', default='area_tips', help="Choose ground truth type 'area_tips', 'one_tip', 'two_tips'")
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--val_csv', default='stimulation.csv', help='Filename of annotation csv for valaidation')
parser.add_argument('--log', help='Path of log file. If log is set the logging will be activated else '
                                  'nothing will be logged.')
parser.add_argument('--num_channels', default=2, type=int,
                    help='Set number of input channels to use for network and data')
parser.add_argument('--border', default=0.5, type=float,
                    help='Set border attribute for ground truth. (area_tips uses absolute pixel value, for '
                         'the others it\'s the percentage of the distance between tips e.g 0.5 for 50% of the distance )')

parser.add_argument('--random_seed', default=456, type=float,
                    help='Initialize random generator with fixed value to reproduce results')
parser.add_argument('--prob_aug', default=0.5, type=float,
                    help='Probability for data augmentations')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--max_chpt', default=10, type=int,
                    help='Set maximum number of checkpoints to keep')
parser.add_argument('--config',
                    help='Configuration file with priors and other needed values.')

args = parser.parse_args()

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
rd_seed = args.random_seed
random.seed(rd_seed)
np.random.seed(rd_seed)
torch.manual_seed(rd_seed)
if cuda:
    torch.cuda.manual_seed(rd_seed)
    torch.cuda.manual_seed_all(rd_seed)

if args.config:
    spec = importlib.util.spec_from_file_location("module.config",args.config)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
else:
    import unet.config.config as config


def pad_prediction(pred, mask):
    _,h,w = mask.shape
    _,_,h_p, w_p = pred.shape
    top = bottom = int((h - h_p) / 2)
    left = right = int((w - w_p) / 2)
    #(paddingLeft, paddingRight, paddingTop, paddingBottom)
    pad = torch.nn.ReplicationPad2d((left,right,top,bottom))
    return pad(pred)
    
def train(loader, net, optimizer, criterion, device):
    net.train()
    running_loss = 0.0
    count = 0
    for i, data in enumerate(loader):
        images, mask = data
        images = images.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        prediction = net(images)  # [N, 2, H, W]
        prediction = pad_prediction(prediction,mask)
        loss = criterion(prediction[:,0,:,:], mask.float().to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

    avg_loss = running_loss / count
    logging.info(
        f"Epoch: {epoch}, Step: {count}, " +
        f"Average Loss: {avg_loss:.4f}"
    )

def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    num = 0

    for _, data in enumerate(loader):
        images, mask = data
        images = images.to(device)
        mask = mask.to(device)
        num += 1

        with torch.no_grad():
            prediction = net(images)  # [N, 2, H, W]
            prediction = pad_prediction(prediction,mask)
            loss = criterion(prediction[:,0,:,:], mask.float().to(device))

        running_loss += loss.item()

    net.train()
    return running_loss / num


if __name__ == '__main__':
    if args.log:
        logging.basicConfig(
            filename=args.log,
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s")

    num_input = args.num_channels

    train_transform = config.TrainAugmentation(p=args.prob_aug)
    test_transform = config.TestTransform()

    logging.info("Prepare training dataset.")
    if args.gt == 'one_tip':
        use_all = False
    else:  # 'two_tips'
        use_all = True

    train_dataset = IOIDatasetETips(os.path.join(args.dataset, args.csv), args.dataset, use_all=use_all,
                                    border=args.border, num_channels=num_input, transform=train_transform)
    val_dataset = IOIDatasetETips(os.path.join(args.validation_dataset, args.val_csv), args.validation_dataset, use_all=use_all,
                                  border=args.border, num_channels=num_input, transform=test_transform)

    train_loader = DataLoader(train_dataset, args.batch_size,num_workers=args.num_workers,shuffle=True)

    logging.info("validation dataset size: {}".format(len(val_dataset)))
    val_loader = DataLoader(val_dataset, args.batch_size,num_workers=args.num_workers,shuffle=False)

    net = UNet(n_classes=1, padding=True, up_mode='upsample')
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net.to(device)
    optim = torch.optim.Adam(net.parameters(),lr=args.lr, weight_decay=1e-4)
    #weights = torch.Tensor([0.000003,1.0]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()#torch.nn.CrossEntropyLoss(weights)

    chpts = SavePointManager(args.checkpoint_folder, args.max_chpt)
    for epoch in range(0, args.epochs):
        #scheduler.step()
        train(train_loader, net, optim, criterion, device=device)
        val_loss = test(val_loader, net, criterion, device)
        logging.info(
            f"Epoch: {epoch}, " +
            f"Validation Loss: {val_loss:.4f}"
        )
        if torch.cuda.device_count() > 1:
            state = {'epoch': epoch, 'state_dict': net.module.state_dict(),
                     'optimizer': optim.state_dict()}#, 'scheduler': scheduler.state_dict()}
        else:
            state = {'epoch': epoch, 'state_dict': net.state_dict(),
                     'optimizer': optim.state_dict()}#, 'scheduler': scheduler.state_dict()}

        model_name = f"unet-Epoch-{epoch}-Loss-{val_loss}.pth"
        if chpts.save(state, model_name, val_loss):
            logging.info(f"Saved model {model_name}")