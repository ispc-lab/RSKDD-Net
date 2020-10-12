import torch
import numpy as np

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0,2,1))
    dist += torch.sum(src**2, -1).view(B, N, 1)
    dist += torch.sum(dst**2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input point cloud [B, N, C] tensor
        idx: point index [B, npoints] tensor
    output:
        indexed points: [B, npoints, C] tensor
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1]*(len(view_shape)-1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape) # [B, S]
    new_points = points[batch_indices, idx, :]
    return new_points

def query_knn_point(k, query, pc):
    """
    Input:
        k: number of neighbor points
        query: query points [B, S, 3]
        pc: point cloud [B, N, 3]
        points: point features
    Output:
        normed_knn_points [B, S, k, 3]
        knn_ids: index of knn points 
    """
    query = query.permute(0,2,1).unsqueeze(3)
    database = pc.permute(0,2,1).unsqueeze(2)
    norm = torch.norm(query-database, dim=1, keepdim=False)
    knn_d, knn_ids = torch.topk(norm, k=k, dim=2, largest=False, sorted=True)
    knn_points = index_points(pc, knn_ids)
    centroids = torch.mean(knn_points, dim=2)
    centroids = centroids.unsqueeze(2).repeat(1,1,k,1)
    normed_knn_points = knn_points - centroids
    return normed_knn_points, knn_ids

def random_dilation_encoding(x_sample, x, k, n):
    '''
    Input:
        x_sample: [B, nsample, C]
        x: [B, N, C]
        k: number of neighbors
        n: dilation ratio
    Output:
        dilation_group: random dilation cluster [B, 4+C, nsample, k]
        dilation_xyz: xyz of random dilation cluster [B, 3, nsample, k]
    '''
    xyz_sample = x_sample[:,:,:3]
    xyz = x[:,:,:3]
    feature = x[:,:,3:]
    _, knn_idx = query_knn_point(int(k*n), xyz_sample, xyz)
    rand_idx = torch.randperm(int(k*n))[:k]
    dilation_idx = knn_idx[:,:,rand_idx]
    dilation_xyz = index_points(xyz, dilation_idx)
    dilation_feature = index_points(feature, dilation_idx)
    xyz_expand = xyz_sample.unsqueeze(2).repeat(1,1,k,1)
    dilation_xyz_resi = dilation_xyz - xyz_expand
    dilation_xyz_dis = torch.norm(dilation_xyz_resi,dim=-1,keepdim=True)
    dilation_group = torch.cat((dilation_xyz_dis, dilation_xyz_resi),dim=-1)
    dilation_group = torch.cat((dilation_group, dilation_feature), dim=-1)
    dilation_group = dilation_group.permute(0,3,1,2)
    return dilation_group, dilation_xyz