function [R,t] = estimateRtSVD(src_sample,dst_sample)
reflect = [1,0,0;0,1,0;0,0,-1];
src_sample_mean = mean(src_sample, 1);
src_sample_decenter = src_sample - src_sample_mean;
dst_sample_mean = mean(dst_sample, 1);
dst_sample_decenter = dst_sample - dst_sample_mean;
W = src_sample_decenter' * dst_sample_decenter;
[u,s,v] = svd(W);
R = v*u';
detR = det(R);
if detR < 0
    v = v*reflect;
    R = v*u';
end
t = -R*src_sample_mean' + dst_sample_mean';