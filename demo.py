import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
import datetime

from models.models import Detector, Descriptor, RSKDD
from data.kittiloader import get_pointcloud

def parse_args():
    parser = argparse.ArgumentParser('RSKDD-Net')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_path', type=str, default='./pretrain/rskdd.pth')
    parser.add_argument('--save_dir', type=str, default='./demo/results')
    parser.add_argument('--nsample', type=int, default=512)
    parser.add_argument('--npoints', type=int, default=16384)
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--desc_dim', type=int, default=128)
    parser.add_argument('--dilation_ratio', type=float, default=2.0)
    parser.add_argument('--data_dir', type=str, default='./demo/pc')

    return parser.parse_args()

def demo(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    model = RSKDD(args)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    file_names = os.listdir(args.data_dir)

    kp_save_dir = os.path.join(args.save_dir, "keypoints")
    desc_save_dir = os.path.join(args.save_dir, "desc")
    if not os.path.exists(kp_save_dir):
        os.makedirs(kp_save_dir)
    if not os.path.exists(desc_save_dir):
        os.makedirs(desc_save_dir)

    for file_name in file_names:
        file_path = os.path.join(args.data_dir, file_name)
        kp_save_path = os.path.join(kp_save_dir, file_name)
        desc_save_path = os.path.join(desc_save_dir, file_name)

        pc, sn = get_pointcloud(file_path, args.npoints)
        feature = torch.cat((pc, sn), dim=-1)
        feature = feature.unsqueeze(0)
        feature = feature.cuda()

        startT = datetime.datetime.now()
        kp, sigmas, desc = model(feature)
        endT = datetime.datetime.now()
        computation_time = (endT - startT).microseconds

        kp_sigmas = torch.cat((kp, sigmas.unsqueeze(1)),dim=1)
        kp_sigmas = kp_sigmas.squeeze().cpu().detach().numpy().transpose()
        desc = desc.squeeze().cpu().detach().numpy().transpose()

        print(file_name, "processed", ' computation time: {} ms'.format(computation_time))

        np.savetxt(kp_save_path, kp_sigmas, fmt='%.04f')
        np.savetxt(desc_save_path, desc, fmt='%.04f')
    
    print("Done")

if __name__ == '__main__':
    args = parse_args()
    demo(args)