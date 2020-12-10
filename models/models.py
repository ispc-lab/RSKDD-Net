import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import random_dilation_encoding

class Detector(nn.Module):
    '''
    Input:
        x: [B, N, 3+C]
    Output:
        keypoints: [B,3,M] 
        saliency_uncertainty: [B,M] 
        random_cluster: [B,4+C,M,k] 
        attentive_feature_map: [B, C_a, M,k]
    '''
    def __init__(self, args):
        super(Detector, self).__init__()
        self.ninput = args.npoints
        self.nsample = args.nsample
        self.k = args.k
        self.dilation_ratio = args.dilation_ratio

        self.C1 = 64
        self.C2 = 128
        self.C3 = 256
        self.in_channel = 8

        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channel, self.C1, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.C1, self.C2, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C2),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(self.C2, self.C3, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C3),
                                   nn.ReLU())
        
        self.mlp1 = nn.Sequential(nn.Conv1d(self.C3, self.C3, kernel_size=1),
                                  nn.BatchNorm1d(self.C3),
                                  nn.ReLU())
        self.mlp2 = nn.Sequential(nn.Conv1d(self.C3, self.C3, kernel_size=1),
                                  nn.BatchNorm1d(self.C3),
                                  nn.ReLU())
        self.mlp3 = nn.Sequential(nn.Conv1d(self.C3, 1, kernel_size=1))

        self.softplus = nn.Softplus()

    def forward(self, x):
        '''
        Input:
            x: [B,N,3+C]
        '''
        # random sample
        randIdx = torch.randperm(self.ninput)[:self.nsample]
        x_sample = x[:,randIdx,:]

        # random dilation cluster
        random_cluster, random_xyz = random_dilation_encoding(x_sample, x, self.k, self.dilation_ratio)

        # Attentive points aggregation
        embedding = self.conv3(self.conv2(self.conv1(random_cluster)))
        x1 = torch.max(embedding, dim=1, keepdim=True)[0]
        x1 = x1.squeeze(dim=1)
        attentive_weights = F.softmax(x1, dim=-1)

        score_xyz = attentive_weights.unsqueeze(1).repeat(1,3,1,1)
        xyz_scored = torch.mul(random_xyz.permute(0,3,1,2),score_xyz)
        keypoints = torch.sum(xyz_scored, dim=-1, keepdim=False)

        score_feature = attentive_weights.unsqueeze(1).repeat(1,self.C3,1,1)
        attentive_feature_map = torch.mul(embedding, score_feature)
        global_cluster_feature = torch.sum(attentive_feature_map, dim=-1, keepdim=False)
        saliency_uncertainty = self.mlp3(self.mlp2(self.mlp1(global_cluster_feature)))
        saliency_uncertainty = self.softplus(saliency_uncertainty) + 0.001
        saliency_uncertainty = saliency_uncertainty.squeeze(dim=1)

        return keypoints, saliency_uncertainty, random_cluster, attentive_feature_map

class Descriptor(nn.Module):
    '''
    Input:
        random_cluster: [B,4+C,M,k] 
        attentive_feature_map: [B, C_a, M,k]
    Output:
        desc: [B,C_f,M]
    '''
    def __init__(self, args):
        super(Descriptor, self).__init__()

        self.C1 = 64
        self.C2 = 128
        self.C3 = 128
        self.C_detector = 256

        self.desc_dim = args.desc_dim
        self.in_channel = 8
        self.k = args.k

        self.conv1 = nn.Sequential(nn.Conv2d(self.in_channel, self.C1, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.C1, self.C2, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C2),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(self.C2, self.C3, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C3),
                                   nn.ReLU())
        
        self.conv4 = nn.Sequential(nn.Conv2d(2*self.C3+self.C_detector, self.C2, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.C2),
                                   nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(self.C2, self.desc_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.desc_dim),
                                   nn.ReLU())
        
    def forward(self, random_cluster, attentive_feature_map):
        x1 = self.conv3(self.conv2(self.conv1(random_cluster)))
        x2 = torch.max(x1, dim=3, keepdim=True)[0]
        x2 = x2.repeat(1,1,1,self.k)
        x2 = torch.cat((x2, x1),dim=1) # [B,2*C3,N,k]
        x2 = torch.cat((x2, attentive_feature_map), dim=1)
        x2 = self.conv5(self.conv4(x2))
        desc = torch.max(x2, dim=3, keepdim=False)[0]
        return desc

class RSKDD(nn.Module):
    '''
    Input:
        x: point cloud [B,N,3+C]
    Output:
        keypoints: [B,3,M]
        sigmas: [B,M]
        desc: [B,C_d,M]
    '''
    def __init__(self, args):
        super(RSKDD, self).__init__()

        self.detector = Detector(args)
        self.descriptor = Descriptor(args)
    
    def forward(self, x):
        keypoints, sigmas, random_cluster, attentive_feature_map = self.detector(x)
        desc = self.descriptor(random_cluster, attentive_feature_map)

        return keypoints, sigmas, desc