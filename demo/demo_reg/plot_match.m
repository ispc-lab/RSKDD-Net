function [] = plot_match(src_pc, src_keypoints, src_inliers, dst_pc, dst_keypoints, dst_inliers)
s1 = 2.0;
s2 = 20.0;
gap = 1.0;
minz_src = min(src_pc(:,3));
maxz_src = max(src_pc(:,3));
minz_dst = min(dst_pc(:,3));
maxz_dst = max(dst_pc(:,3));
dist = abs(minz_dst - maxz_src);
dst_pc(:,3) = dst_pc(:,3)+gap+dist;
dst_keypoints(:,3) = dst_keypoints(:,3)+gap+dist;
dst_inliers(:,3) = dst_inliers(:,3)+gap+dist;

scatter3(src_pc(:,1),src_pc(:,2),src_pc(:,3),s1,src_pc(:,4));
hold on;
scatter3(dst_pc(:,1),dst_pc(:,2),dst_pc(:,3),s1,dst_pc(:,4));
hold on;
scatter3(src_keypoints(:,1),src_keypoints(:,2),src_keypoints(:,3),s2,'r','filled');
hold on;
scatter3(dst_keypoints(:,1),dst_keypoints(:,2),dst_keypoints(:,3),s2,'r','filled');
hold on;

inlier_size = size(src_inliers);
inlier_size = inlier_size(1);
point_1 = src_inliers(1,:);
point_2 = dst_inliers(1,:);
line = cat(1,point_1,point_2);
plot3(line(:,1),line(:,2),line(:,3),'r*');
hold on;
for i = 1:inlier_size
    point_1 = src_inliers(i,:);
    point_2 = dst_inliers(i,:);
    line = cat(1,point_1,point_2);
    plot3(line(:,1),line(:,2),line(:,3),'r');
    hold on;
end
grid off;
axis off;
hold off;