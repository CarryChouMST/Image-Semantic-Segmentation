import sys

sys.path.extend([r'D:\dl_project\dl_project_cnn\DataInput', r'D:\dl_project\dl_project_cnn\Models' \
                    , r'D:\dl_project\dl_project_cnn\Trainer', r'D:\dl_project\dl_project_cnn\Utils'])

from Trainer.train import *
from Trainer.self_train import self_train
from Trainer.test import *
from Trainer.bridge import *
from Models.ResUnet import *
from Models.SegNet import SegNet
from Models.DLinkNet34 import *
from Models.hrnetv2 import *
from Models.Unet import *
from Models.SegNet import *
from Models.FINALNET import FINALNET
from Models.FINALNET_base import FINALNET_base
from Models.FINALNET_base_h import FINALNET_base_h
from Models.deeplabv3p_Xception65 import deeplabv3plus_Xception65
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')

    return parser.parse_args()


if __name__ == '__main__':
    # config_path = r'F:\dl_project_cnn\config.ini'
    config_path = r'D:\dl_project\dl_project_cnn\config.ini'
    # net = deeplabv3_hrnetv2_32(num_classes=1, pretrained_backbone=False)
    # net = ResUnet(in_channels=3, num_classes=1, norm_layer='GN')
    # net = UNet(3, 1)
    # net = SegNet(3, 1)
    # net = DLinkNet34(num_classes=1, num_channels=3)

    net = deeplabv3plus_Xception65(num_classes=1)


    args = get_args()
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location='cuda:0')
        )
        logging.info(f'Model is loaded from {args.load}')

    mode = 'train'

    if mode == 'train':
        train(net=net, config_path=config_path)
    if mode == 'test':
        # loss, res_dict = test(net=net, config_path=config_path)
        # for gamma in np.arange(0, 1, 0.1):
        loss, res_dict = dilated_test(net=net, config_path=config_path, dilated_size=512, size=512, gamma=0.5, update=False)
        # print(f'loss = {loss}')
        print('----------------')
        print(f'gamma={0.5}')
        print(f'res_dict={res_dict}')
    if mode == 'bridge':
        bridge(net=net, config_path=config_path)
































