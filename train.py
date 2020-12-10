import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# We organize the parameters of this project using wandb
# If do not want to use this, please delete the code about wandb
import wandb

import argparse
from tqdm import tqdm
import os

from data.kittiloader import KittiDataset
from models.models import Detector, Descriptor, RSKDD
from models.losses import ChamferLoss, Point2PointLoss, Matching_loss

def parse_args():
    parser = argparse.ArgumentParser('RSKDD')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--data_dir', type=str, default='',help='dir of dataset')
    parser.add_argument('--seq', type=str, default='00', help='training sequence of Kitti dataset')
    parser.add_argument('--npoints', type=int, default=16384, help='number of input points')
    parser.add_argument('--k',type=int, default=128)
    parser.add_argument('--nsample', type=int, default=512)
    parser.add_argument('--desc_dim', type=int, default=128)
    parser.add_argument('--dilation_ratio', type=float, default=2.0)
    parser.add_argument('--pretrain_detector', type=str, default='', \
        help='path to pretrain model of detector')
    parser.add_argument('--alpha', type=float, default=1.0, \
        help='ratio between chamfer loss and point to point loss')
    parser.add_argument('--beta', type=float, default=1.0, \
        help='ratio between chamfer loss and point to matching loss')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--sigma_max', type=float, default=3.0, \
        help='predefined sigma upper bound')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt', help='path to save model')
    parser.add_argument('--train_type', type=str, default='det', help='det/desc')
    parser.add_argument('--use_wandb', action='store_true')
    return parser.parse_args()

def train_detector(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    trainset = KittiDataset(args.data_dir, args.seq, args.npoints)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, \
        shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    model = Detector(args)
    model = model.cuda()

    if args.use_wandb:
        wandb.watch(model)

    chamfer_criterion = ChamferLoss()
    point_criterion = Point2PointLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=args.lr, momentum=0.9)
    best_epoch_loss = float("inf")

    for epoch in range(args.epoch):
        torch.cuda.empty_cache()
        model.train()
        epoch_loss = 0
        epoch_chamfer_loss = 0
        epoch_point_loss = 0
        count = 0
        pbar = tqdm(enumerate(trainloader))
        for i, data in pbar:
            src_pc, src_sn, dst_pc, dst_sn, T = data
            src = torch.cat((src_pc, src_sn), dim=-1)
            dst = torch.cat((dst_pc, dst_sn), dim=-1)
            src = src.cuda()
            dst = dst.cuda()
            src_pc = src_pc.cuda()
            dst_pc = dst_pc.cuda()
            T = T.cuda()
            R = T[:,:3,:3].contiguous()
            t = T[:,:3,3].unsqueeze(1).contiguous()

            src_kp, src_sigma, _, _ = model(src)
            dst_kp, dst_sigma, _, _ = model(dst)
             
            src_kp_trans = (torch.matmul(R, src_kp).permute(0,2,1) + t).permute(0,2,1).contiguous()
            chamfer_loss = chamfer_criterion(src_kp_trans, dst_kp, src_sigma, dst_sigma)
            point_loss = point_criterion(src_kp, src_pc.permute(0,2,1).contiguous()) + point_criterion(dst_kp, dst_pc.permute(0,2,1).contiguous())
            loss = chamfer_loss + args.alpha * point_loss

            epoch_loss = epoch_loss + float(loss)
            epoch_chamfer_loss = epoch_chamfer_loss + float(chamfer_loss)
            epoch_point_loss = epoch_point_loss + float(point_loss)
            count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                pbar.set_description('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, i, len(trainloader), 100. * i/len(trainloader), loss.item()
                ))
        
        epoch_loss = epoch_loss/count
        epoch_chamfer_loss = epoch_chamfer_loss/count
        epoch_point_loss = epoch_point_loss/count
        print('Epoch {} finished. Loss: {:.3f} Chamfer loss: {:.3f} Point loss: {:.3f}'.\
            format(epoch+1, epoch_loss, epoch_chamfer_loss, epoch_point_loss))
        
        if args.use_wandb:
            wandb.log({"loss":epoch_loss, "chamfer loss":epoch_chamfer_loss, "point loss":epoch_point_loss})

        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if epoch_loss < best_epoch_loss:
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'best_detector.pth'))

def train_descriptor(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    trainset = KittiDataset(args.data_dir, args.seq, args.npoints)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, \
        shuffle=True, num_workers=args.num_workers, drop_last=True)
    
    model = RSKDD(args)
    model = model.cuda()
    model.detector.load_state_dict(torch.load(args.pretrain_detector))
    
    if args.use_wandb:
        wandb.watch(model)

    chamfer_criterion = ChamferLoss()
    point_criterion = Point2PointLoss()
    matching_criterion = Matching_loss(args)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr,momentum=0.9)
    best_epoch_loss = float("inf")

    for epoch in range(args.epoch):
        torch.cuda.empty_cache()
        model.train()
        epoch_loss = 0
        epoch_matching_loss = 0
        epoch_chamfer_loss = 0
        epoch_point_loss = 0
        count = 0
        pbar = tqdm(enumerate(trainloader))
        for i, data in pbar:
            src_pc, src_sn, dst_pc, dst_sn, T = data
            src = torch.cat((src_pc, src_sn),dim=-1)
            dst = torch.cat((dst_pc, dst_sn),dim=-1)
            src = src.cuda()
            dst = dst.cuda()
            src_pc = src_pc.cuda()
            dst_pc = dst_pc.cuda()
            T = T.cuda()
            R = T[:,:3,:3]
            t = T[:,:3,3]
            src_kp, src_sigmas, src_desc = model(src)
            dst_kp, dst_sigmas, dst_desc = model(dst)

            src_kp_trans = (torch.matmul(R, src_kp).permute(0,2,1)+t.unsqueeze(1)).permute(0,2,1)
            chamfer_loss = chamfer_criterion(src_kp_trans, dst_kp, src_sigmas, dst_sigmas)
            matching_loss = matching_criterion(src_kp_trans, src_sigmas, src_desc, dst_kp, dst_sigmas, dst_desc)
            point_loss = point_criterion(src_kp, src_pc.permute(0,2,1)) + point_criterion(dst_kp, dst_pc.permute(0,2,1))

            loss = chamfer_loss + args.beta * matching_loss + args.alpha * point_loss

            epoch_loss = epoch_loss + float(loss)
            epoch_matching_loss = epoch_matching_loss + float(matching_loss)
            epoch_chamfer_loss = epoch_chamfer_loss + float(chamfer_loss)
            epoch_point_loss = epoch_point_loss + float(point_loss)
            count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                pbar.set_description('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, i, len(trainloader), 100. * i/len(trainloader), loss.item()
                ))
        
        epoch_loss = epoch_loss/count
        epoch_matching_loss = epoch_matching_loss/count
        epoch_chamfer_loss = epoch_chamfer_loss/count
        epoch_point_loss = epoch_point_loss/count

        print('Epoch {} finished. Loss: {:.3f} Matching loss: {:.3f} Chamfer loss: {:.3f} Point loss: {:.3f}'.\
            format(epoch+1, epoch_loss, epoch_matching_loss, epoch_chamfer_loss, epoch_point_loss))

        if args.use_wandb:
            wandb.log({"loss":epoch_loss, "matching loss":epoch_matching_loss, "chamfer loss":epoch_chamfer_loss, "point loss":epoch_point_loss})

        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)

        if epoch_loss < best_epoch_loss:
            torch.save(model.state_dict(), args.ckpt_dir+'best_full.pth')

if __name__ == '__main__':
    args = parse_args()
    if args.use_wandb:
        wandb.init(config=args, project='RSKDD-Net', name=args.train_type)

    if args.train_type == 'det':
        train_detector(args)
    
    elif args.train_type == 'desc':
        train_descriptor(args)

    else:
        print("Invalid train_type (det or desc)")