#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <iomanip>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/features/normal_3d.h>

typedef pcl::PointXYZ PointType;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudType;

pcl::PointCloud<pcl::PointXYZI>::Ptr readKittiBinData(std::string &in_file) {
    std::fstream input(in_file.c_str(), std::ios::in|std::ios::binary);
    if (!input.good()) {
        exit(EXIT_FAILURE);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);

    int i;
    for (i = 0; input.good() && !input.eof(); i++) {
        pcl::PointXYZI point;
        pcl::PointXYZI newpoint;
        float temp;
        input.read((char *) &point.x, 3* sizeof(float));
        input.read((char *) &temp, sizeof(float));
        newpoint.x = point.x;
        newpoint.y = point.y;
        newpoint.z = point.z;
        newpoint.intensity = temp;
        pc->push_back(newpoint);
    }
    input.close();
    return pc;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr cloudFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr pc) {
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZI>), cloud_f(new pcl::PointCloud<pcl::PointXYZI>);
    vg.setInputCloud(pc);
    vg.setLeafSize(0.1f, 0.1f, 0.1f);
    vg.filter(*cloud_filtered);

    return cloud_filtered;
}

void normalEstimation(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered, std::string out_path) {
    std::cout << "Pointcloud after filtered has: " << cloud_filtered->points.size() << "points" << std::endl;
    pcl::search::Search<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZI, pcl::Normal> normal_estimator;
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(cloud_filtered);
    normal_estimator.setKSearch(10);
    normal_estimator.compute(*normals);

    std::fstream fout;
    fout.open(out_path, std::ios::out);
    for (int i = 0; i < cloud_filtered->points.size(); i++) {
        pcl::PointXYZI point = cloud_filtered->points[i];
        pcl::Normal normal = normals->points[i];
        fout << point.x << " " << point.y << " " << point.z << " " << point.intensity << " "\
        << normal.normal_x << " " << normal.normal_y << " " << normal.normal_z << " " << normal.curvature << std::endl;
    }
    fout.close();
}

// We provides an example for KITTI odometry dataset preprocessing. 
int main(int argc, char **argv) {
    std::string root = ""; // Please set the root to your own dataset
    std::string sequence = ""; // Please set the process sequences
    std::string velodyne_dir = root + sequence + "/velodyne/";
    std::string save_dir = root + sequence + "/velodyne_txt/";

    if (opendir(save_dir.c_str())==NULL) {
        mkdir(save_dir.c_str(),0775);
    }
    
    DIR *dp;
    struct dirent *dirp;
    std::vector<std::string> fns;
    dp = opendir(velodyne_dir.c_str());
    while ((dirp=readdir(dp))!=NULL) {
        if (strcmp(dirp->d_name, ".") == 0  || strcmp(dirp->d_name, "..") == 0) {
            continue;
        }
        fns.push_back(dirp->d_name);
    }
    int max_number = fns.size();
    std::cout << "Processing " << max_number << " point clouds......" << std::endl;

    for (int i =0; i < max_number; i++) {
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << i;
        std::string file_name = ss.str();
        std::cout << file_name << ": ";
        std::string file_path = velodyne_dir + file_name + ".bin";
        std::string out_path = save_dir + file_name + ".txt";
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_f (new pcl::PointCloud<pcl::PointXYZI>);
        cloud = readKittiBinData(file_path);
        cloud_f = cloudFilter(cloud);
        normalEstimation(cloud_f, out_path);
    }

    std::cout << "Process finished. Processed point clouds have been saved in " << save_dir << std::endl;

    return 0;
}

