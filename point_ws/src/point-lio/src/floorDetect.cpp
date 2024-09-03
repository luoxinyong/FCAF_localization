/*
 * 
 *
 *  Created on: 2024.07.06
 *      Author: ZXD
 */

#include <ros/ros.h>
#include <chrono>
#include <iostream>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include "open3d/Open3D.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using namespace std;     
using namespace open3d;

// 1、准确识别天花板的点云，必须高于本身位置，且与地面平行(判断xyz的系数)
// 2、楼层识别，高度大于当前天花板的高度，则当前楼层建图完成，进行保存
// 3、高度不再大范围变化时，则换了楼层，求出高度数据，这里楼层与对应的点云保存至一个vector::pair中，一边点云一边坐标数据
// 4、当有楼层高度在中间或回到之前的楼层时，在对应位置插入，对坐标进行排序
// 5、建完图后，先对每一层楼内部进行一次帧之间的配准，最后对每个楼层求一个天花板平面，再进行一次楼层间的坐标变换

struct floorData{
	vector<int> cloud_idx;
	double floor_high;
};

vector<floorData> floor_vec;
//话题发布者
ros::Publisher pubCloud;

// set parameters
double normal_variance_threshold_deg, coplanarity_deg = 75, outlier_ratio = 0.75, min_plane_edge_length = 5.0, radius = 0.2, distance_threshold = 2;
int max_nn, nrNeighbors, ransac_n, num_iteration, min_num_points;
float voxel_size = 0.1;

bool first = true;
int init_num = 0;
double guess_high;
vector<double> high_buff;    // 在一层内所有的高度集合
vector<double> floor_high_buff = {-10000.0};   // 每层的平均高度
vector<pair<std::shared_ptr<geometry::PointCloud>, geometry_msgs::PoseStamped>> cloud_buff;
vector<pair<int, int>> idx_floor;   // 第一个参数是某一层的点云索引，第二个参数是楼层

void detect(const geometry_msgs::PoseStamped::ConstPtr& msg, double& high, std::shared_ptr<geometry::PointCloud> source)
{
	high = -1000;
	
	auto cloud_ptr = source->VoxelDownSample(voxel_size);

    auto t1 = std::chrono::high_resolution_clock::now();

    const geometry::KDTreeSearchParam &normals_search_param = geometry::KDTreeSearchParamKNN(nrNeighbors);

    cloud_ptr->EstimateNormals(normals_search_param);
    cloud_ptr->OrientNormalsTowardsCameraLocation();

    // std::cout << "EstimateNormals: " << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t1).count() << " seconds" << std::endl;

    
    const geometry::KDTreeSearchParam &search_param = geometry::KDTreeSearchParamHybrid(radius, max_nn);

    t1 = std::chrono::high_resolution_clock::now();
    const std::vector<std::shared_ptr<geometry::OrientedBoundingBox>> patches = cloud_ptr->DetectPlanarPatches(normal_variance_threshold_deg,
                                           coplanarity_deg, outlier_ratio,
                                           min_plane_edge_length,
                                           min_num_points, search_param);
										
    // std::cout << "DetectPlanarPatches: " << patches.size() << " in " << std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - t1).count() << " seconds" << std::endl;

	if(patches.size() > 0)
	{
		Eigen::Vector3d ground_normal(0,0,1);
		double threshold = std::cos(10.0 * 3.1415926 / 180.0);
		
		double max_area = 0;
		std::shared_ptr<open3d::geometry::OrientedBoundingBox> largest_vertical_plane = nullptr;
		
		for (const auto &plane : patches) 
		{
			Eigen::Vector3d plane_normal = plane->R_.col(2);
			plane_normal.normalize();

			double dot_product = std::fabs(plane_normal.dot(ground_normal));
			
			if(dot_product > threshold)
			{
				double area = plane->extent_(0) * plane->extent_(1);
				
				if(area > max_area)
				{
					max_area = area;
					largest_vertical_plane = plane;
				}
			}
		}
		
		if(largest_vertical_plane != nullptr)
		{
			std::vector<size_t> idxs = largest_vertical_plane->GetPointIndicesWithinBoundingBox(cloud_ptr->points_);

			int size = idxs.size();
			pcl::PointCloud<pcl::PointXYZINormal>::Ptr floorCloud(new pcl::PointCloud<pcl::PointXYZINormal>(size, 1));

			for (size_t i = 0; i < idxs.size(); ++i) 
			{
				floorCloud->points[i].x = cloud_ptr->points_[idxs[i]](0);
				floorCloud->points[i].y = cloud_ptr->points_[idxs[i]](1);
				floorCloud->points[i].z = cloud_ptr->points_[idxs[i]](2);
			}

			Eigen::Vector3d plane_normal = largest_vertical_plane->R_.col(2);
			Eigen::Vector3d plane_center = largest_vertical_plane->center_;

			double d = -plane_normal.dot(plane_center);

			if(abs(plane_normal.transpose().x()) < 0.1 && abs(plane_normal.transpose().y()) < 0.1 && plane_normal.transpose().z() > -1.01 && plane_normal.transpose().z() < -0.99)
			{
				// 天花板点云
				if(d > msg->pose.position.z)
				{
					sensor_msgs::PointCloud2 laserCloudmsg;
					pcl::toROSMsg(*floorCloud, laserCloudmsg);
					laserCloudmsg.header.stamp = ros::Time::now();
					laserCloudmsg.header.frame_id = "camera_init";
					pubCloud.publish(laserCloudmsg);
					// std::cout << plane_normal.transpose() << " . (x, y, z) + " << d << "= 0" << std::endl;
					high = d;
				}
			}
		}
	}
}

void floorInit(double high)
{
	static double last_high = 0;
	
	if(high != -1000)
	{
		if(first)
		{
			last_high = high;
			first = false;
		}
		else
		{
			if(abs(high - last_high) < 0.5)
			{
				init_num++;
				if(init_num > 1) 
				{
					cout << "first_high:" << high << endl;
					high_buff.push_back(high);
				}
			}
			else
				first = true;
		}
	} 
}

int last_idx = 0;

int whichFloor(double mean, int current_idx)
{
	int floor_num;
	floorData floor_data; 
	// 找到当前位置在哪一层
	if(mean > floor_vec.back().floor_high)
	{
		floor_num = floor_vec.size();
		floor_data.cloud_idx = {last_idx, current_idx};
		floor_data.floor_high = mean;
		floor_vec.push_back(floor_data);
	}
	else  // 此时位置在中间
	{
		for(int i = 0; i < floor_vec.size() - 1; i++)
		{
			if(mean > floor_vec[i].floor_high - 1 && mean < floor_vec[i+1].floor_high + 1)
			{
				floor_num = i + 1;
				floor_vec[floor_num].cloud_idx.push_back(last_idx);
				floor_vec[floor_num].cloud_idx.push_back(current_idx);
				// 之前没有经历过的楼层
				// if(abs(mean - floor_vec[i].floor_high) > guess_high && abs(mean - floor_vec[i+1].floor_high) > guess_high)
				// {
				// 	vector<double>::iterator it = floor_vec.begin().floor_high + floor_num;
				// 	floor_data.floor_high = mean;
				//  floor_data.cloud_idx.push_back(last_idx);
				//  floor_data.cloud_idx.push_back(current_idx);
				// 	floor_vec.insert(it, floor_data);
				// }
				break;
			}
		}
	}

	return floor_num;
}


void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
	double high;
	std::shared_ptr<geometry::PointCloud> cloud;
	geometry_msgs::PoseStamped pose = *msg;
	
	string all_points_dir(string(string(ROOT_DIR) + "PCD/for_detect") + string(".pcd"));
	std::shared_ptr<geometry::PointCloud> source = open3d::io::CreatePointCloudFromFile(all_points_dir);
	detect(msg, high, source);

	cloud_buff.push_back(make_pair(source, pose));

	if(high != -1000)
	{
		cout << "current high:" << high << endl;
		// 初始化
		if(init_num < 2)
		{
			floorInit(high);
		}
		else
		{
			if(abs(high - high_buff.back()) < 0.5)  // 0.5米的误差内则算是好的数据
				high_buff.push_back(high);
		}
	}
	
	if(high_buff.size() > 1)
	{
		double add_high = 0;
		for(int i = 0; i < high_buff.size(); i++)
			add_high += high_buff[i];
		
		double mean = add_high / high_buff.size();

		// 上下楼
		if((msg->pose.position.z - mean) > 0.05 || (mean - msg->pose.position.z) > guess_high)
		{
			int floor_num = whichFloor(mean, cloud_buff.size() - 1);
			cout << "Now Floor:" << floor_num  << endl;
			last_idx = cloud_buff.size() - 1;

			high_buff.clear();
			first = true;
			init_num = 0;
		}
	}
}

void readParameters(ros::NodeHandle &nh)
{
	nh.param<double>("detectObject/normal_variance_threshold_deg",normal_variance_threshold_deg,60);
	nh.param<double>("detectObject/coplanarity_deg",coplanarity_deg,75);
	nh.param<double>("detectObject/outlier_ratio",outlier_ratio,0.75);
	nh.param<double>("detectObject/min_plane_edge_length",min_plane_edge_length,5.0);
	nh.param<double>("detectObject/radius",radius,0.2);
	nh.param<double>("detectObject/distance_threshold",distance_threshold,2);
	nh.param<int>("detectObject/min_num_points",min_num_points,30);
	nh.param<int>("detectObject/max_nn",max_nn,100);
	nh.param<int>("detectObject/nrNeighbors",nrNeighbors,100);
	nh.param<int>("detectObject/ransac_n",ransac_n,3);
	nh.param<int>("detectObject/num_iteration",num_iteration,1000);
	nh.param<float>("detectObject/voxel_size",voxel_size,0.1);
	nh.param<double>("detectObject/guess_high",guess_high,3);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "floor");
    ros::NodeHandle n("~");
	ros::NodeHandle nh;
	readParameters(n);

    ros::Subscriber pose_sub = nh.subscribe("pose_for_detect", 1, poseCallback);

    pubCloud = nh.advertise<sensor_msgs::PointCloud2>("floor_cloud", 100000);

	floorData floor_data; 
	floor_data.cloud_idx = {-1};
	floor_data.floor_high = -10000;
	floor_vec.push_back(floor_data);
       
    while(ros::ok())
    {   
		ros::spinOnce();
    }

	double add_high = 0;
	for(int i = 0; i < high_buff.size(); i++)
		add_high += high_buff[i];
	
	double mean = add_high / high_buff.size();

	int floor_num = whichFloor(mean, cloud_buff.size() - 1);

	for(int i = 0; i < floor_vec.size(); i++)
	{
		cout << floor_vec[i].floor_high << endl;
	}

	for(int i = 0; i < floor_vec.size(); i++)
	{
		for(int j = 0; j < floor_vec[i].cloud_idx.size(); j++)
			cout << i << ":" << floor_vec[i].cloud_idx[j] << endl;
	}

	
	for(int j = 1; j < floor_vec.size(); j++)
    {
		auto wait_save = make_shared<geometry::PointCloud>();
		for(int m = 0; m < floor_vec[j].cloud_idx.size(); m+=2)
		{
			for(int i = floor_vec[j].cloud_idx[m]; i < floor_vec[j].cloud_idx[m+1]; i++)
			{
				std::shared_ptr<geometry::PointCloud> pcd = cloud_buff[i].first;
				*wait_save += *pcd;	
			}
		}

		auto wait_save_down = wait_save->VoxelDownSample(0.1);
        string all_points_dir(string(string(ROOT_DIR) + "PCD/floor_") + to_string(j) + string(".pcd"));
		cout << "current scan saved to /PCD/" << all_points_dir << endl;
        open3d::io::WritePointCloud(all_points_dir, *wait_save_down);
    }
	

    return 0;
}
