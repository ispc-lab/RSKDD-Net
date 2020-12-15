import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import os
from tqdm import tqdm
import datetime
import glob

from models.models import Detector, Descriptor, RSKDD
from data.kittiloader import get_pointcloud

def parse_args():
    parser = argparse.ArgumentParser('RSKDD-Net')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--data_dir', type=str, default='',help='dir of dataset')
    parser.add_argument('--test_seq', type=str, default='')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--nsample', type=int, default=1024)
    parser.add_argument('--npoints', type=int, default=16384)
    parser.add_argument('--k', type=int, default=128)
    parser.add_argument('--desc_dim', type=int, default=128)
    parser.add_argument('--dilation_ratio', type=float, default=2.0)
    parser.add_argument('--test_type', type=str, default='desc', help='det/desc')
    return parser.parse_args()

def test_detector(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = Detector(args)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    velodyne_dir = os.path.join(args.data_dir, 'sequences', args.test_seq, 'velodyne_txt')
    velodyne_names = os.listdir(velodyne_dir)
    velodyne_names = sorted(velodyne_names)

    for filename in velodyne_names:
        filepath = os.path.join(velodyne_dir, filename)
        kp_path = os.path.join(args.save_dir, "keypoints", filename)
        pc, sn = get_pointcloud(filepath, args.npoints)
        feature = torch.cat((pc, sn), dim=-1)
        feature = feature.unsqueeze(0)
        feature = feature.cuda()

        startT = datetime.datetime.now()
        kp, sigmas, _, _ = model(feature)
        endT = datetime.datetime.now()
        computation_time = (endT - startT).microseconds
        kp_sigmas = torch.cat((kp, sigmas.unsqueeze(1)),dim=1)
        kp_sigmas = kp_sigmas.squeeze().cpu().detach().numpy().transpose()

        np.savetxt(kp_path, kp_sigmas)

def test_descriptor(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = RSKDD(args)
    model = model.cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    velodyne_dir = os.path.join(args.data_dir, 'sequences', args.test_seq, 'velodyne_txt')
    velodyne_names = os.listdir(velodyne_dir)
    velodyne_names = sorted(velodyne_names)

    kp_save_dir = os.path.join(args.save_dir, args.test_seq, "keypoints")
    desc_save_dir = os.path.join(args.save_dir, args.test_seq, "desc")

    if not os.path.exists(kp_save_dir):
        os.makedirs(kp_save_dir)
    if not os.path.exists(desc_save_dir):
        os.makedirs(desc_save_dir)

    for filename in velodyne_names:
        filepath = os.path.join(velodyne_dir, filename)

        kp_path = os.path.join(kp_save_dir, "keypoints", filename)
        desc_path = os.path.join(desc_save_dir, "desc", filename)

        pc, sn = get_pointcloud(filepath, args.npoints)
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
        print(filename, computation_time)

        np.savetxt(kp_path, kp_sigmas)
        np.savetxt(desc_path, desc)

if __name__ == '__main__':
    args = parse_args()
    if args.test_type == 'det':
        test_detector(args)
    elif args.test_type == 'desc':
        test_descriptor(args)
    else:
        print("Invalid test type")