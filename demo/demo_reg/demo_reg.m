clear;
clc;

kp_dir = '../results/keypoints';
desc_dir = '../results/desc';
pc_dir = '../pc';

nkp = 256;

dataset = 'ford'; % kitti or ford

src_name = strcat(dataset, '_01.txt');
dst_name = strcat(dataset, '_02.txt');

src_kp_path = fullfile(kp_dir, src_name);
src_desc_path = fullfile(desc_dir, src_name);
src_pc_path = fullfile(pc_dir, src_name);

dst_kp_path = fullfile(kp_dir, dst_name);
dst_desc_path = fullfile(desc_dir, dst_name);
dst_pc_path = fullfile(pc_dir, dst_name);

src_pc = load(src_pc_path);
src_kp_sigmas = load(src_kp_path);
src_kp = src_kp_sigmas(:,1:3);
src_sigmas = src_kp_sigmas(:,4);
src_desc = load(src_desc_path);
[temp, src_idx] = sort(src_sigmas);
src_kp = src_kp(src_idx,:);
src_desc = src_desc(src_idx,:);
src_kp = src_kp(1:nkp,:);
src_desc = src_desc(1:nkp,:);

dst_pc = load(dst_pc_path);
dst_kp_sigmas = load(dst_kp_path);
dst_kp = dst_kp_sigmas(:,1:3);
dst_sigmas = dst_kp_sigmas(:,4);
dst_desc = load(dst_desc_path);
[temp, dst_idx] = sort(dst_sigmas);
dst_kp = dst_kp(dst_idx,:);
dst_desc = dst_desc(dst_idx,:);
dst_kp = dst_kp(1:nkp,:);
dst_desc = dst_desc(1:nkp,:);

[R, t, src_inliers, dst_inliers] = estimateRt(src_kp, src_desc, dst_kp, dst_desc);
plot_match(src_pc, src_kp, src_inliers, dst_pc, dst_kp, dst_inliers);