import torch
import torch.nn as nn
import torch.nn.functional as F

class ChamferLoss(nn.Module):
    """
    Calculate probabilistic chamfer distance between keypoints1 and keypoints2
    Input:
        keypoints1: [B,3,M]
        keypoints2: [B,3,M]
        sigma1: [B,M]
        sigma2: [B,M]
    """
    def __init__(self):
        super(ChamferLoss, self).__init__()
    
    def forward(self, keypoints1, keypoints2, sigma1, sigma2):
        B, M = keypoints1.size()[0], keypoints1.size()[2]
        N = keypoints2.size()[2]

        keypoints1_expanded = keypoints1.unsqueeze(3).expand(B,3,M,N)
        keypoints2_expanded = keypoints2.unsqueeze(2).expand(B,3,M,N)

        # diff: [B, M, M]
        diff = torch.norm(keypoints1_expanded-keypoints2_expanded, dim=1, keepdim=False)

        if sigma1 is None or sigma2 is None:
            min_dist_forward, _ = torch.min(diff, dim=2, keepdim=False)
            forward_loss = min_dist_forward.mean()

            min_dist_backward, _ = torch.min(diff, dim=1, keepdim=False)
            backward_loss = min_dist_backward.mean()

            loss = forward_loss + backward_loss
        
        else:
            min_dist_forward, min_dist_forward_I = torch.min(diff, dim=2, keepdim=False)
            selected_sigma_2 = torch.gather(sigma2, dim=1, index=min_dist_forward_I)
            sigma_forward = (sigma1 + selected_sigma_2)/2
            forward_loss = (torch.log(sigma_forward)+min_dist_forward/sigma_forward).mean()

            min_dist_backward, min_dist_backward_I = torch.min(diff, dim=1, keepdim=False)
            selected_sigma_1 = torch.gather(sigma1, dim=1, index=min_dist_backward_I)
            sigma_backward = (sigma2 + selected_sigma_1)/2
            backward_loss = (torch.log(sigma_backward)+min_dist_backward/sigma_backward).mean()

            loss = forward_loss + backward_loss
        return loss

class Point2PointLoss(nn.Module):
    '''
    Calculate point-to-point loss between keypoints and pc
    Input:
        keypoints: [B,3,M]
        pc: [B,3,N]
    '''
    def __init__(self):
        super(Point2PointLoss, self).__init__()
    
    def forward(self, keypoints, pc):
        B, M = keypoints.size()[0], keypoints.size()[2]
        N = pc.size()[2]
        keypoints_expanded = keypoints.unsqueeze(3).expand(B,3,M,N)
        pc_expanded = pc.unsqueeze(2).expand(B,3,M,N)
        diff = torch.norm(keypoints_expanded-pc_expanded, dim=1, keepdim=False)
        min_dist, _ = torch.min(diff, dim=2, keepdim=False)
        return torch.mean(min_dist)

class Matching_loss(nn.Module):
    '''
    Calculate matching loss
    Input:
        src_kp: [B,3,M]
        src_sigma: [B,M]
        src_desc: [B,C,M]
        dst_kp: [B,3,M]
        dst_sigma: [B,M]
        dst_desc: [B,C,M]
    '''
    def __init__(self, args):
        super(Matching_loss, self).__init__()
        self.t = args.temperature
        self.sigma_max = args.sigma_max
    
    def forward(self, src_kp, src_sigma, src_desc, dst_kp, dst_sigma, dst_desc):

        src_desc = src_desc.unsqueeze(3) # [B,C,M,1]
        dst_desc = dst_desc.unsqueeze(2) # [B,C,1,M]

        desc_dists = torch.norm((src_desc - dst_desc), dim=1) # [B,M,M]
        desc_dists_inv = 1.0/desc_dists
        desc_dists_inv = desc_dists_inv/self.t

        score_src = F.softmax(desc_dists_inv, dim=2)
        score_dst = F.softmax(desc_dists_inv, dim=1).permute(0,2,1)

        src_kp = src_kp.permute(0,2,1)
        dst_kp = dst_kp.permute(0,2,1)

        src_kp_corres = torch.matmul(score_src, dst_kp)
        dst_kp_corres = torch.matmul(score_dst, src_kp)

        diff_forward = torch.norm((src_kp - src_kp_corres), dim=-1)
        diff_backward = torch.norm((dst_kp - dst_kp_corres), dim=-1)

        src_weights = torch.clamp(self.sigma_max - src_sigma, min=0.01)
        src_weights_mean = torch.mean(src_weights, dim=1, keepdim=True)
        src_weights = (src_weights/src_weights_mean).detach()

        dst_weights = torch.clamp(self.sigma_max - dst_sigma, min=0.01)
        dst_weights_mean = torch.mean(dst_weights, dim=1, keepdim=True)
        dst_weights = (dst_weights/dst_weights_mean).detach()

        loss_forward = (src_weights * diff_forward).mean()
        loss_backward = (dst_weights * diff_backward).mean()

        loss = loss_forward + loss_backward

        return loss