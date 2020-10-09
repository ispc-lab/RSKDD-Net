function [R,t,src_inliers, dst_inliers] = estimateRt(src_kp,src_desc,dst_kp,dst_desc)
% src_pc: source keypoints, Nx3
% src_des: source descriptors, NxD
% dst_pc: target keypoints, Nx3
% dst_des: target descriptors, NxD
nsample = 3;
max_iter = 10000;
min_inlier_size = 100;

nsize = length(src_kp);

[src_corres_idx,src_corres_dists] = knnsearch(dst_desc,src_desc);
src_corres_kp = dst_kp(src_corres_idx,:);

iter = 0;
max_inlier_size = 0;

best_inlier_idx = 0;
dist_t = 1.0; % inlier threshold

N = 1;
p = 0.99;

while iter < max_iter && N > iter
    rand_idx = randi(nsize, nsample, 1);
    src_sample = src_kp(rand_idx,:);
    dst_sample = src_corres_kp(rand_idx,:);
    [R1, t1] = estimateRtSVD(src_sample, dst_sample);
    
    src_trans = (R1*src_kp' + t1)';
    
    resi = src_trans - src_corres_kp;
    resi = vecnorm(resi, 2, 2);
    inlier_idx = find(resi < dist_t);
    inlier_size = length(inlier_idx);
    if inlier_size > max_inlier_size
        inlier_ratio = inlier_size/nsize;
        pNoOutliers = 1 - inlier_ratio^nsample;
        pNoOutliers = max(eps, pNoOutliers);
        pNoOutliers = min(1-eps, pNoOutliers);
        N = log(1-p)/log(pNoOutliers);
        N = max(N,10);
        best_inlier_idx = inlier_idx;
    end
    iter = iter + 1;
end

src_inliers = src_kp(best_inlier_idx,:);
dst_inliers = src_corres_kp(best_inlier_idx,:);
[R,t] = estimateRtSVD(src_inliers, dst_inliers);