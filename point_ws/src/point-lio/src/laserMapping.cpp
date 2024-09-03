// #include <so3_math.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include <pcl/registration/icp.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "li_initialization.h"
#include <malloc.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "open3d/Open3D.h"

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

// #include <cv_bridge/cv_bridge.h>
// #include "matplotlibcpp.h"
// #include <ros/console.h>

using namespace std;     
using namespace open3d;
#define PUBFRAME_PERIOD     (20)

/*****************************************/
gtsam::NonlinearFactorGraph gtSAMgraph;
gtsam::Values initialEstimate;
gtsam::Values isamCurrentEstimate;
gtsam::ISAM2 *isam;

vector<pair<int, int>> loopIndexQueue;
vector<gtsam::Pose3> loopPoseQueue;
vector<gtsam::noiseModel::Diagonal::shared_ptr> loopNoiseQueue;
deque<std_msgs::Float64MultiArray> loopInfoVec;

bool aLoopIsClosed = false, findLoop = false, Reset = false, odomReset = false;
float transformTobeMapped[6];  // 滤波后的位姿
float updateTransform[6];      // 优化更新后的位姿

double last_roll = 0, last_pitch = 0, last_yaw = 0, last_x = 0, last_y = 0, last_z = 0;  // 上一时刻滤波的输出，用于计算增量
double timeLast = 0;

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;
typedef pcl::PointXYZI PointType3D;
vector<PointCloudXYZI::Ptr> loopCloudKeyFrames;
vector<PointCloudXYZI::Ptr> loopCloudKeyFramesBody;
vector<std::shared_ptr<open3d::geometry::PointCloud>> loopCloudKeyFramesBodyOpen3d;
PointCloudXYZI::Ptr loop_wait_save(new PointCloudXYZI());

ros::Publisher pubHistoryKeyFrames;
ros::Publisher pubLoopConstraintEdge;
ros::Publisher pubLoopOdom;
ros::Publisher pubLoopPath;
ros::Publisher pubLoopCloud;

map<int, int> loopIndexContainer; // from new to old

pcl::PointCloud<PointType3D>::Ptr   cloudKeyPoses3D;       // 存放关键帧的3D位置
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D;       // 存放关键帧的6D位姿
pcl::PointCloud<PointType3D>::Ptr   copy_cloudKeyPoses3D;
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D;
pcl::PointCloud<PointType3D>::Ptr   ICPPoses3D;       // 存放关键帧的3D位置

pcl::KdTreeFLANN<PointType3D>::Ptr kdtreeHistoryKeyPoses;

nav_msgs::Odometry loop_odom;
nav_msgs::Path globalPath;

/*****************************************/
const float MOV_THRESHOLD = 1.5f;

string root_dir = ROOT_DIR;

int time_log_counter = 0; //, publish_count = 0;

bool init_map = false, flg_first_scan = true;

// Time Log Variables
double match_time = 0, solve_time = 0, propag_time = 0, update_time = 0;

bool  flg_reset = false, flg_exit = false;

//surf feature in map
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body_space(new PointCloudXYZI());
PointCloudXYZI::Ptr init_feats_world(new PointCloudXYZI());
std::deque<PointCloudXYZI::Ptr> depth_feats_world;
pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;
pcl::VoxelGrid<PointType> downSizeFilterSave;
pcl::VoxelGrid<PointType> downSizeFilterShow;
V3D euler_cur;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::PoseStamped msg_body_pose;

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang;
    if (!use_imu_as_input)
    {
        rot_ang = SO3ToEuler(kf_output.x_.rot);
    }
    else
    {
        rot_ang = SO3ToEuler(kf_input.x_.rot);
    }
    
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    if (use_imu_as_input)
    {
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.pos(0), kf_input.x_.pos(1), kf_input.x_.pos(2)); // Pos  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.vel(0), kf_input.x_.vel(1), kf_input.x_.vel(2)); // Vel  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.bg(0), kf_input.x_.bg(1), kf_input.x_.bg(2));    // Bias_g  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.ba(0), kf_input.x_.ba(1), kf_input.x_.ba(2));    // Bias_a  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.gravity(0), kf_input.x_.gravity(1), kf_input.x_.gravity(2)); // Bias_a  
    }
    else
    {
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.pos(0), kf_output.x_.pos(1), kf_output.x_.pos(2)); // Pos  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.vel(0), kf_output.x_.vel(1), kf_output.x_.vel(2)); // Vel  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.bg(0), kf_output.x_.bg(1), kf_output.x_.bg(2));    // Bias_g  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.ba(0), kf_output.x_.ba(1), kf_output.x_.ba(2));    // Bias_a  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.gravity(0), kf_output.x_.gravity(1), kf_output.x_.gravity(2)); // Bias_a  
    }
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu;
    if (extrinsic_est_en)
    {
        if (!use_imu_as_input)
        {
            p_body_imu = kf_output.x_.offset_R_L_I * p_body_lidar + kf_output.x_.offset_T_L_I;
        }
        else
        {
            p_body_imu = kf_input.x_.offset_R_L_I * p_body_lidar + kf_input.x_.offset_T_L_I;
        }
    }
    else
    {
        p_body_imu = Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU;
    }
    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void MapIncremental() {
    PointVector points_to_add;
    int cur_pts = feats_down_world->size();
    points_to_add.reserve(cur_pts);

    for (size_t i = 0; i < cur_pts; ++i) {
        /* decide if need add to map */
        PointType &point_world = feats_down_world->points[i];
        if (!Nearest_Points[i].empty()) {
            const PointVector &points_near = Nearest_Points[i];

            Eigen::Vector3f center =
                ((point_world.getVector3fMap() / filter_size_map_min).array().floor() + 0.5) * filter_size_map_min;
            bool need_add = true;
            for (int readd_i = 0; readd_i < points_near.size(); readd_i++) {
                Eigen::Vector3f dis_2_center = points_near[readd_i].getVector3fMap() - center;
                if (fabs(dis_2_center.x()) < 0.5 * filter_size_map_min &&
                    fabs(dis_2_center.y()) < 0.5 * filter_size_map_min &&
                    fabs(dis_2_center.z()) < 0.5 * filter_size_map_min) {
                    need_add = false;
                    break;
                }
            }
            if (need_add) {
                points_to_add.emplace_back(point_world);
            }
        } else {
            points_to_add.emplace_back(point_world);
        }
    }
    ivox_->AddPoints(points_to_add);
}

void publish_init_map(const ros::Publisher & pubLaserCloudFullRes)
{
    int size_init_map = init_feats_world->size();

    sensor_msgs::PointCloud2 laserCloudmsg;
                
    pcl::toROSMsg(*init_feats_world, laserCloudmsg);
        
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "camera_init";
    pubLaserCloudFullRes.publish(laserCloudmsg);
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_show(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFullRes)
{
    // feats_down_show.reset(new PointCloudXYZI());
    downSizeFilterShow.setInputCloud(feats_down_world);
    downSizeFilterShow.filter(*feats_down_show);

    int size = feats_down_show->points.size();
    if (scan_pub_en)
    {
        PointCloudXYZI::Ptr   laserCloudWorld(new PointCloudXYZI(size, 1));
        for (int i = 0; i < size; i++)
        {
            laserCloudWorld->points[i].x = feats_down_show->points[i].x;
            laserCloudWorld->points[i].y = feats_down_show->points[i].y;
            laserCloudWorld->points[i].z = feats_down_show->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_show->points[i].intensity; // feats_down_world->points[i].y; //
        }
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
        // publish_count -= PUBFRAME_PERIOD;
    }
    
    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        PointCloudXYZI::Ptr   laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            laserCloudWorld->points[i].x = feats_down_show->points[i].x;
            laserCloudWorld->points[i].y = feats_down_show->points[i].y;
            laserCloudWorld->points[i].z = feats_down_show->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_show->points[i].intensity;
        }

        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        pointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    // publish_count -= PUBFRAME_PERIOD;
}

template<typename T>
void set_posestamp(T & out)
{
    if (!use_imu_as_input)
    {
        out.position.x = kf_output.x_.pos(0);
        out.position.y = kf_output.x_.pos(1);
        out.position.z = kf_output.x_.pos(2);
        Eigen::Quaterniond q(kf_output.x_.rot);
        out.orientation.x = q.coeffs()[0];
        out.orientation.y = q.coeffs()[1];
        out.orientation.z = q.coeffs()[2];
        out.orientation.w = q.coeffs()[3];
    }
    else
    {
        out.position.x = kf_input.x_.pos(0);
        out.position.y = kf_input.x_.pos(1);
        out.position.z = kf_input.x_.pos(2);
        Eigen::Quaterniond q(kf_input.x_.rot);
        out.orientation.x = q.coeffs()[0];
        out.orientation.y = q.coeffs()[1];
        out.orientation.z = q.coeffs()[2];
        out.orientation.w = q.coeffs()[3];
    }
}

PointCloudXYZI::Ptr detect_wait_save(new PointCloudXYZI());
void save_pcd_for_detect(const ros::Publisher & pubPoseDetect)
{
    int size_ = world_save_detect->points.size();
    PointCloudXYZI::Ptr   laserCloudWorld(new PointCloudXYZI(size_, 1));

    for (int i = 0; i < size_; i++)
    {
        laserCloudWorld->points[i].x = world_save_detect->points[i].x;
        laserCloudWorld->points[i].y = world_save_detect->points[i].y;
        laserCloudWorld->points[i].z = world_save_detect->points[i].z;
        laserCloudWorld->points[i].intensity = world_save_detect->points[i].intensity;
    }

    *detect_wait_save += *laserCloudWorld;

    static int scan_wait_num = 0;
    scan_wait_num ++;
    if (detect_wait_save->size() > 0 && scan_wait_num >= scan_save_interval)
    {  
        string all_points_dir(string(string(ROOT_DIR) + "PCD/for_detect") + string(".pcd"));
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << all_points_dir << endl;
        pcd_writer.writeBinary(all_points_dir, *detect_wait_save);
        detect_wait_save->clear();
        scan_wait_num = 0;

        set_posestamp(msg_body_pose.pose);

        // msg_body_pose.header.stamp = ros::Time::now();
        msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
        msg_body_pose.header.frame_id = "camera_init";
        pubPoseDetect.publish(msg_body_pose);
    
    }
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    if (publish_odometry_without_downsample)
    {
        odomAftMapped.header.stamp = ros::Time().fromSec(time_current);
    }
    else
    {
        odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    }
    set_posestamp(odomAftMapped.pose.pose);
    
    pubOdomAftMapped.publish(odomAftMapped);

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body") );
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose.pose);
    // msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";
    static int jjj = 0;
    jjj++;
    // if (jjj % 2 == 0) // if path is too large, the rvis will crash
    {
        path.poses.emplace_back(msg_body_pose);
        pubPath.publish(path);
    }
}        

Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
{ 
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

pcl::PointCloud<PointType>::Ptr transformPointCloud(pcl::PointCloud<PointType>::Ptr cloudIn, PointTypePose* transformIn)
{
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
    Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

    int cloudSize = cloudIn->points.size();

    cloudOut->resize(cloudSize);

    for (int i = 0; i < cloudSize; i++)
    {
        const auto &pointFrom = cloudIn->points[i];
        cloudOut->points[i].x = transCur(0,0) * pointFrom.x + transCur(0,1) * pointFrom.y + transCur(0,2) * pointFrom.z + transCur(0,3);
        cloudOut->points[i].y = transCur(1,0) * pointFrom.x + transCur(1,1) * pointFrom.y + transCur(1,2) * pointFrom.z + transCur(1,3);
        cloudOut->points[i].z = transCur(2,0) * pointFrom.x + transCur(2,1) * pointFrom.y + transCur(2,2) * pointFrom.z + transCur(2,3);
        cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
}

std::shared_ptr<open3d::geometry::PointCloud> transformPointCloud(std::shared_ptr<open3d::geometry::PointCloud> cloudIn, PointTypePose* transformIn)
{
    
    Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);

    int cloudSize = cloudIn->points_.size();
    std::shared_ptr<open3d::geometry::PointCloud> cloudOut(new open3d::geometry::PointCloud);
    cloudOut->points_.resize(cloudSize);

    for (int i = 0; i < cloudSize; i++)
    {
        const auto &pointFrom = cloudIn->points_[i];
        double x = transCur(0,0) * pointFrom.x() + transCur(0,1) * pointFrom.y() + transCur(0,2) * pointFrom.z() + transCur(0,3);
        double y = transCur(1,0) * pointFrom.x() + transCur(1,1) * pointFrom.y() + transCur(1,2) * pointFrom.z() + transCur(1,3);
        double z = transCur(2,0) * pointFrom.x() + transCur(2,1) * pointFrom.y() + transCur(2,2) * pointFrom.z() + transCur(2,3);
        cloudOut->points_[i] = Eigen::Vector3d(x, y, z);   
    }

    return cloudOut;
}

gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                                gtsam::Point3(double(thisPoint.x),    double(thisPoint.y),     double(thisPoint.z)));
}

gtsam::Pose3 trans2gtsamPose(float transformIn[])
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]), 
                                gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

// 对关键帧3D位姿构建kd树，并用当前帧位置从kd树寻找距离最近的几帧，挑选时间间隔最远的一帧最为匹配帧
// latestID 最后一帧关键帧索引； closestID 当前帧对应匹配帧的索引
bool detectLoopClosureDistance(int *latestID, int *closestID)
{
    int loopKeyCur = copy_cloudKeyPoses3D->size() - 1 - currentKeyframeSearchNum;
    int loopKeyPre = -1;

    // check loop constraint added before
    // 确认最后一帧关键帧没有被加入过回环关系中
    auto it = loopIndexContainer.find(loopKeyCur);
    if (it != loopIndexContainer.end())
        return false;

    // find the closest history key frame
    // 将关键帧的3D位置构建kdtree，并检索空间位置相近的关键帧
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);


    // 寻找空间距离相近的关键帧(10米)
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), historyKeyframeSearchRadius, pointSearchIndLoop, pointSearchSqDisLoop, 0);
    
    for (int i = 0; i < (int)pointSearchIndLoop.size(); ++i)
    {
        int id = pointSearchIndLoop[i];
        if (abs(copy_cloudKeyPoses6D->points[id].time - lidar_end_time) > historyKeyframeSearchTimeDiff)
        {
            loopKeyPre = id;
            break;
        }
    }

    if (loopKeyPre == -1 || loopKeyCur == loopKeyPre)
        return false;

    *latestID = loopKeyCur;
    *closestID = loopKeyPre;

    return true;
}

void loopFindNearKeyframes(std::shared_ptr<open3d::geometry::PointCloud>& nearKeyframes, const int& key, const int& searchNum)
{
    // extract near keyframes
    nearKeyframes->points_.clear();
    
    int cloudSize = copy_cloudKeyPoses6D->size();
    // int cloudSize = loopCloudKeyFrames.size();
    for (int i = -searchNum; i <= searchNum; ++i)
    {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize )
            continue;

        // PointCloudXYZI::Ptr Pose(new PointCloudXYZI());   
        
        // Pose = loopCloudKeyFrames[keyNear];
        // *nearKeyframes += *Pose;
        *nearKeyframes += *transformPointCloud(loopCloudKeyFramesBodyOpen3d[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
    }

    if (nearKeyframes->points_.empty())
        return;

    // downsample near keyframes
    // pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    // downSizeFilterSurf.setInputCloud(nearKeyframes);
    // downSizeFilterSurf.filter(*cloud_temp);
    // *nearKeyframes = *cloud_temp;
}

void loopFindNearKeyframes(PointCloudXYZI::Ptr& nearKeyframes, const int& key, const int& searchNum)
{
    // extract near keyframes
    nearKeyframes->clear();
    
    int cloudSize = copy_cloudKeyPoses6D->size();
    // int cloudSize = loopCloudKeyFrames.size();
    for (int i = -searchNum; i <= searchNum; ++i)
    {
        int keyNear = key + i;
        if (keyNear < 0 || keyNear >= cloudSize )
            continue;

        // PointCloudXYZI::Ptr Pose(new PointCloudXYZI());   
        
        // Pose = loopCloudKeyFrames[keyNear];
        // *nearKeyframes += *Pose;
        *nearKeyframes += *transformPointCloud(loopCloudKeyFramesBody[keyNear], &copy_cloudKeyPoses6D->points[keyNear]);
    }

    if (nearKeyframes->empty())
        return;

    // downsample near keyframes
    // pcl::PointCloud<PointType>::Ptr cloud_temp(new pcl::PointCloud<PointType>());
    // downSizeFilterSurf.setInputCloud(nearKeyframes);
    // downSizeFilterSurf.filter(*cloud_temp);
    // *nearKeyframes = *cloud_temp;
}

void performLoopClosureOpen3D()
{
    if (cloudKeyPoses3D->size() < currentKeyframeSearchNum + 1)
        return;

    mtx_buffer.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx_buffer.unlock();

    // find keys
    // 寻找匹配帧
    int loopKeyCur;
    int loopKeyPre;
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
        return;

    // extract cloud
    // 提取关键帧及其附近的点云
    std::shared_ptr<open3d::geometry::PointCloud> cureKeyframeCloud(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> prevKeyframeCloud(new open3d::geometry::PointCloud);
    {
        //提取当前帧的点云
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, currentKeyframeSearchNum);
        //提取匹配帧附近的局部点云
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);  // 前后各25帧

        if (cureKeyframeCloud->points_.size() < 300 || prevKeyframeCloud->points_.size() < 1000)
            return;
    }

    // ICP Settings
    prevKeyframeCloud->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(1, 30));
    cureKeyframeCloud->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(1, 30));
    int iteration = 100;
    Eigen::Matrix4d trans = Eigen::Matrix4d::Identity();
    auto result = open3d::pipelines::registration::RegistrationICP(*cureKeyframeCloud, *prevKeyframeCloud, historyKeyframeSearchRadius*2, trans,
                                                           open3d::pipelines::registration::TransformationEstimationPointToPlane(),
                                                           open3d::pipelines::registration::ICPConvergenceCriteria(1e-6, 1e-6, iteration));

    cout << result.inlier_rmse_ << endl;

    if (result.inlier_rmse_ > historyKeyframeFitnessScore || result.inlier_rmse_ == 0)
        return;

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Matrix4f matrix4f = result.transformation_.cast<float>();
    Eigen::Affine3f correctionLidarFrame(matrix4f);
    // transform from world origin to wrong pose
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    if((copy_cloudKeyPoses6D->points[loopKeyCur].z - copy_cloudKeyPoses6D->points[loopKeyPre].z) > loopHeightMax || (copy_cloudKeyPoses6D->points[loopKeyCur].z - copy_cloudKeyPoses6D->points[loopKeyPre].z) < loopHeightMin)
        return;
    // transform from world origin to corrected pose
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);   // ICP配准后的当前坐标
    gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = result.inlier_rmse_;
    if(noiseScore < 0.01)
        noiseScore = 0.01;
    cout << "Find Loop!!!" << endl;
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint
    mtx_buffer.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx_buffer.unlock();
    
    // add loop constriant
    loopIndexContainer[loopKeyCur] = loopKeyPre;  
    
    findLoop = true;
    
}

void performLoopClosureICP()
{
    if (cloudKeyPoses3D->size() < currentKeyframeSearchNum + 1)
        return;

    mtx_buffer.lock();
    *copy_cloudKeyPoses3D = *cloudKeyPoses3D;
    *copy_cloudKeyPoses6D = *cloudKeyPoses6D;
    mtx_buffer.unlock();

    // find keys
    // 寻找匹配帧
    int loopKeyCur;
    int loopKeyPre;
    if (detectLoopClosureDistance(&loopKeyCur, &loopKeyPre) == false)
        return;

    // extract cloud
    // 提取关键帧及其附近的点云
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr prevKeyframeCloud(new pcl::PointCloud<PointType>());
    {
        //提取当前帧的点云
        loopFindNearKeyframes(cureKeyframeCloud, loopKeyCur, currentKeyframeSearchNum);
        //提取匹配帧附近的局部点云
        loopFindNearKeyframes(prevKeyframeCloud, loopKeyPre, historyKeyframeSearchNum);  // 前后各25帧

        if (cureKeyframeCloud->size() < 300 || prevKeyframeCloud->size() < 1000)
            return;
    }

    // ICP Settings
    static pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(historyKeyframeSearchRadius*2);
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align clouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(prevKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);

    cout << icp.getFitnessScore() << endl;

    if (icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore)
        return;

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    // transform from world origin to wrong pose
    Eigen::Affine3f tWrong = pclPointToAffine3f(copy_cloudKeyPoses6D->points[loopKeyCur]);
    if((copy_cloudKeyPoses6D->points[loopKeyCur].z - copy_cloudKeyPoses6D->points[loopKeyPre].z) > loopHeightMax || (copy_cloudKeyPoses6D->points[loopKeyCur].z - copy_cloudKeyPoses6D->points[loopKeyPre].z) < loopHeightMin)
        return;
    // transform from world origin to corrected pose
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);   // ICP配准后的当前坐标
    gtsam::Pose3 poseFrom = gtsam::Pose3(gtsam::Rot3::RzRyRx(roll, pitch, yaw), gtsam::Point3(x, y, z));
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(copy_cloudKeyPoses6D->points[loopKeyPre]);
    gtsam::Vector Vector6(6);
    float noiseScore = icp.getFitnessScore();
    if(noiseScore < 0.01)
        noiseScore = 0.01;
    cout << "Find Loop!!!" << endl;
    Vector6 << noiseScore, noiseScore, noiseScore, noiseScore, noiseScore, noiseScore;
    gtsam::noiseModel::Diagonal::shared_ptr constraintNoise = gtsam::noiseModel::Diagonal::Variances(Vector6);

    // Add pose constraint
    mtx_buffer.lock();
    loopIndexQueue.push_back(make_pair(loopKeyCur, loopKeyPre));
    loopPoseQueue.push_back(poseFrom.between(poseTo));
    loopNoiseQueue.push_back(constraintNoise);
    mtx_buffer.unlock();
    
    // add loop constriant
    loopIndexContainer[loopKeyCur] = loopKeyPre;  
    
    findLoop = true;
    
}

void visualizeLoopClosure()
{
    if (loopIndexContainer.empty())
        return;
    
    visualization_msgs::MarkerArray markerArray;
    // loop nodes
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = "camera_init";
    markerNode.header.stamp = ros::Time().fromSec(lidar_end_time);
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3; markerNode.scale.y = 0.3; markerNode.scale.z = 0.3; 
    markerNode.color.r = 0; markerNode.color.g = 0.8; markerNode.color.b = 1;
    markerNode.color.a = 1;
    // loop edges
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = "camera_init";
    markerEdge.header.stamp = ros::Time().fromSec(lidar_end_time);
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9; markerEdge.color.g = 0.9; markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    for (auto it = loopIndexContainer.begin(); it != loopIndexContainer.end(); ++it)
    {
        int key_cur = it->first;
        int key_pre = it->second;
        geometry_msgs::Point p;
        p.x = copy_cloudKeyPoses6D->points[key_cur].x;
        p.y = copy_cloudKeyPoses6D->points[key_cur].y;
        p.z = copy_cloudKeyPoses6D->points[key_cur].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = copy_cloudKeyPoses6D->points[key_pre].x;
        p.y = copy_cloudKeyPoses6D->points[key_pre].y;
        p.z = copy_cloudKeyPoses6D->points[key_pre].z;
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    pubLoopConstraintEdge.publish(markerArray);
}

void loopClosureThread()
{
    ros::Rate rate(1.0);
    while (ros::ok())
    {
        rate.sleep();
        if(ifPCL)
            performLoopClosureICP();
        else
            performLoopClosureOpen3D();
        visualizeLoopClosure();
    }
}

void updatePath(const PointTypePose& pose_in)
{
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time().fromSec(lidar_end_time);
    pose_stamped.header.frame_id = "camera_init";
    pose_stamped.pose.position.x = pose_in.x;
    pose_stamped.pose.position.y = pose_in.y;
    pose_stamped.pose.position.z = pose_in.z;
    tf::Quaternion q = tf::createQuaternionFromRPY(pose_in.roll, pose_in.pitch, pose_in.yaw);
    pose_stamped.pose.orientation.x = q.x();
    pose_stamped.pose.orientation.y = q.y();
    pose_stamped.pose.orientation.z = q.z();
    pose_stamped.pose.orientation.w = q.w();

    globalPath.poses.push_back(pose_stamped);
}

void addOdomFactor()
{
    if (cloudKeyPoses3D->points.empty())
    {
        gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-2, 1e-2, M_PI*M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
        gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(0, trans2gtsamPose(transformTobeMapped), priorNoise));
        initialEstimate.insert(0, trans2gtsamPose(transformTobeMapped));
    }else{
        gtsam::noiseModel::Diagonal::shared_ptr odometryNoise = gtsam::noiseModel::Diagonal::Variances((gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        gtsam::Pose3 poseFrom = pclPointTogtsamPose3(cloudKeyPoses6D->points.back());
        gtsam::Pose3 poseTo   = trans2gtsamPose(transformTobeMapped);
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudKeyPoses3D->size()-1, cloudKeyPoses3D->size(), poseFrom.between(poseTo), odometryNoise));
        initialEstimate.insert(cloudKeyPoses3D->size(), poseTo);
    }
}

void addLoopFactor()
{
    if (loopIndexQueue.empty())
        return;

    for (int i = 0; i < (int)loopIndexQueue.size(); ++i)
    {
        int indexFrom = loopIndexQueue[i].first;
        int indexTo = loopIndexQueue[i].second;
        gtsam::Pose3 poseBetween = loopPoseQueue[i];
        gtsam::noiseModel::Diagonal::shared_ptr noiseBetween = loopNoiseQueue[i];
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(indexFrom, indexTo, poseBetween, noiseBetween));
    }

    loopIndexQueue.clear();
    loopPoseQueue.clear();
    loopNoiseQueue.clear();
    aLoopIsClosed = true;
}

void correctPoses()
{
    if (cloudKeyPoses3D->points.empty())
        return;

    if (aLoopIsClosed == true)
    {
        // clear path
        globalPath.poses.clear();
        // update key poses
        int numPoses = isamCurrentEstimate.size();
        for (int i = 0; i < numPoses; ++i)
        {
            cloudKeyPoses3D->points[i].x = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().x();
            cloudKeyPoses3D->points[i].y = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().y();
            cloudKeyPoses3D->points[i].z = isamCurrentEstimate.at<gtsam::Pose3>(i).translation().z();

            cloudKeyPoses6D->points[i].x = cloudKeyPoses3D->points[i].x;
            cloudKeyPoses6D->points[i].y = cloudKeyPoses3D->points[i].y;
            cloudKeyPoses6D->points[i].z = cloudKeyPoses3D->points[i].z;
            cloudKeyPoses6D->points[i].roll  = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().roll();
            cloudKeyPoses6D->points[i].pitch = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().pitch();
            cloudKeyPoses6D->points[i].yaw   = isamCurrentEstimate.at<gtsam::Pose3>(i).rotation().yaw();

            updatePath(cloudKeyPoses6D->points[i]);

        }

        aLoopIsClosed = false;
    }
}

bool saveFrame()
{
    if (cloudKeyPoses3D->points.empty())
        return true;

    Eigen::Affine3f transStart = pclPointToAffine3f(cloudKeyPoses6D->back());
    Eigen::Affine3f transFinal = pcl::getTransformation(transformTobeMapped[3], transformTobeMapped[4], transformTobeMapped[5], 
                                                        transformTobeMapped[0], transformTobeMapped[1], transformTobeMapped[2]);
    Eigen::Affine3f transBetween = transStart.inverse() * transFinal;
    float x, y, z, roll, pitch, yaw;
    pcl::getTranslationAndEulerAngles(transBetween, x, y, z, roll, pitch, yaw);

    if (abs(roll)  < surroundingkeyframeAddingAngleThreshold &&
        abs(pitch) < surroundingkeyframeAddingAngleThreshold && 
        abs(yaw)   < surroundingkeyframeAddingAngleThreshold &&
        sqrt(x*x + y*y + z*z) < surroundingkeyframeAddingDistThreshold)
        return false;

    return true;
}

void LoopandOptmization()
{
    double timeLaserInfoCur = lidar_end_time;
    static double timeLastProcessing = -1;
    
    if (timeLaserInfoCur - timeLastProcessing >= mappingProcessInterval)
        timeLastProcessing = timeLaserInfoCur;
    else
        return;

    /*****  计算优化前的输入位姿:transformTobeMapped  ******/
    set_posestamp(msg_body_pose.pose);   // 滤波输出的存在误差的位姿:msg_body_pose

    tf::Quaternion quat;
    tf::quaternionMsgToTF(msg_body_pose.pose.orientation, quat);
    double roll_, pitch_, yaw_;//定义存储r\p\y的容器
    tf::Matrix3x3(quat).getRPY(roll_, pitch_, yaw_);//进行转换

    // 计算滤波里程计增量
    double d_roll  = roll_  -  last_roll;
    double d_pitch = pitch_ -  last_pitch;
    double d_yaw   = yaw_   -  last_yaw;
    double d_x = msg_body_pose.pose.position.x - last_x;
    double d_y = msg_body_pose.pose.position.y - last_y;
    double d_z = msg_body_pose.pose.position.z - last_z;

    cout << d_z << endl;

    // 上一时刻的优化位姿加上滤波计算出的增量 = 优化前的输入位姿
    transformTobeMapped[0] = updateTransform[0] + d_roll;
    transformTobeMapped[1] = updateTransform[1] + d_pitch;
    transformTobeMapped[2] = updateTransform[2] + d_yaw;
    transformTobeMapped[3] = updateTransform[3] + d_x; 
    transformTobeMapped[4] = updateTransform[4] + d_y;
    transformTobeMapped[5] = updateTransform[5] + d_z;

    if(!saveFrame())  // 是否保存为关键帧
        return;


    
    /*****  开始优化  ******/
    // odom factor
    addOdomFactor();

    // loop factor
    addLoopFactor();

    // update iSAM
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();

    if (aLoopIsClosed == true)
    {
        isam->update();
        isam->update();
        isam->update();
        isam->update();
        isam->update();
    }

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    //save key poses
    PointType3D thisPose3D;
    PointTypePose thisPose6D;
    gtsam::Pose3 latestEstimate;

    isamCurrentEstimate = isam->calculateEstimate();
    latestEstimate = isamCurrentEstimate.at<gtsam::Pose3>(isamCurrentEstimate.size()-1);

    /*****  优化结束，保存数据  ******/
    thisPose3D.x = latestEstimate.translation().x();
    thisPose3D.y = latestEstimate.translation().y();
    thisPose3D.z = latestEstimate.translation().z();
    thisPose3D.intensity = cloudKeyPoses3D->size();
    cloudKeyPoses3D->push_back(thisPose3D);

    thisPose6D.x = thisPose3D.x;
    thisPose6D.y = thisPose3D.y;
    thisPose6D.z = thisPose3D.z;
    thisPose6D.intensity = thisPose3D.intensity ;
    thisPose6D.roll  = latestEstimate.rotation().roll();
    thisPose6D.pitch = latestEstimate.rotation().pitch();
    thisPose6D.yaw   = latestEstimate.rotation().yaw();
    thisPose6D.time = lidar_end_time;
    cloudKeyPoses6D->push_back(thisPose6D);

    // 保存为最优位姿
    updateTransform[0] = latestEstimate.rotation().roll();
    updateTransform[1] = latestEstimate.rotation().pitch();
    updateTransform[2] = latestEstimate.rotation().yaw();
    updateTransform[3] = latestEstimate.translation().x();
    updateTransform[4] = latestEstimate.translation().y();
    updateTransform[5] = latestEstimate.translation().z();

    // 发布轨迹
    updatePath(thisPose6D);

    // 更新位姿
    correctPoses();
    
    /*****  将关键帧的点云转换至最优位姿下  ******/
    tf::Quaternion q_ = tf::createQuaternionFromRPY(updateTransform[0],  updateTransform[1], updateTransform[2]);
    Eigen::Quaterniond q_world(q_.w(), q_.x(), q_.y(), q_.z()); 
    Eigen::Vector3d pos_world(updateTransform[3], updateTransform[4], updateTransform[5]);
    
    q_ = tf::createQuaternionFromRPY(0,  0, 0);
    Eigen::Quaterniond q_body(q_.w(), q_.x(), q_.y(), q_.z()); 
    Eigen::Vector3d pos_body(0, 0, 0); 

    int size = feats_down_body->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));
    PointCloudXYZI::Ptr laserCloudBody(new PointCloudXYZI(size, 1));

    std::shared_ptr<open3d::geometry::PointCloud> laserCloudBodyOpen3d(new open3d::geometry::PointCloud);
    laserCloudBodyOpen3d->points_.resize(size);

    for (int i = 0; i < size; i++)
    {
        V3D p_body(feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z);
        V3D p_global(q_world * (kf_output.x_.offset_R_L_I*p_body + kf_output.x_.offset_T_L_I) + pos_world);
        V3D p_global_body(q_body * (kf_output.x_.offset_R_L_I*p_body + kf_output.x_.offset_T_L_I) + pos_body);

        laserCloudWorld->points[i].x = p_global(0);
        laserCloudWorld->points[i].y = p_global(1);
        laserCloudWorld->points[i].z = p_global(2);
        laserCloudWorld->points[i].intensity = feats_down_body->points[i].intensity;

        laserCloudBody->points[i].x = p_global_body(0);
        laserCloudBody->points[i].y = p_global_body(1);
        laserCloudBody->points[i].z = p_global_body(2);
        laserCloudBody->points[i].intensity = feats_down_body->points[i].intensity;   
        
        laserCloudBodyOpen3d->points_[i] = Eigen::Vector3d(p_global_body(0), p_global_body(1), p_global_body(2));   
    }

    loopCloudKeyFrames.push_back(laserCloudWorld);
    loopCloudKeyFramesBody.push_back(laserCloudBody);
    loopCloudKeyFramesBodyOpen3d.push_back(laserCloudBodyOpen3d);
    


    if (pubLoopPath.getNumSubscribers() != 0)
    {
        globalPath.header.stamp =  ros::Time().fromSec(lidar_end_time);
        globalPath.header.frame_id = "camera_init";
        pubLoopPath.publish(globalPath);
    }

    // if (pubLoopCloud.getNumSubscribers() != 0)
    // {
    //     sensor_msgs::PointCloud2 laserCloudmsg;
    //     pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    //     laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    //     laserCloudmsg.header.frame_id = "camera_init";
    //     pubLoopCloud.publish(laserCloudmsg);
    // }


    // 保存位姿，用于计算下一时刻的增量
    last_roll = roll_;
    last_pitch = pitch_;
    last_yaw = yaw_;
    last_x = msg_body_pose.pose.position.x;
    last_y = msg_body_pose.pose.position.y;
    last_z = msg_body_pose.pose.position.z;
    

}

void floorDetectThread()
{
    ros::Rate rate(detect_rate);
    // vector<double> height_data;
    // window_size = 10;
    // height_threshold = 0.5;
    while (ros::ok())
    {
        rate.sleep();
        set_posestamp(msg_body_pose.pose);   // 滤波输出的存在误差的位姿:msg_body_pose
        cout << msg_body_pose.pose.position.z << endl;

        // 高度开始显著变化时，计算此前平面高度的平均值

        // 高度没有变化时，每隔n帧检测一次平面，计算高度值；把对应点云保存为天花板的语义信息
        
    }
    
}

PointCloudXYZI::Ptr body_save_detect(new PointCloudXYZI());
int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh("~");
    ros::AsyncSpinner spinner(0);
    spinner.start();
    readParameters(nh);
    cout<<"lidar_type: "<<lidar_type<<endl;
    ivox_ = std::make_shared<IVoxType>(ivox_options_);
    
    path.header.stamp    = ros::Time().fromSec(lidar_end_time);
    path.header.frame_id ="camera_init";

    /*** variables definition for counting ***/
    int frame_num = 0;
    double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_propag = 0;

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    // downSizeFilterSave.setLeafSize(filter_size_save_min, filter_size_save_min, filter_size_save_min);
    downSizeFilterShow.setLeafSize(filter_size_show_min, filter_size_show_min, filter_size_show_min);
    
        Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
        Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    
    if (extrinsic_est_en)
    {
        if (!use_imu_as_input)
        {
            kf_output.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_output.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        }
        else
        {
            kf_input.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_input.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        }
    }

    p_imu->lidar_type = p_pre->lidar_type = lidar_type;
    p_imu->imu_en = imu_en;

    kf_input.init_dyn_share_modified_2h(get_f_input, df_dx_input, h_model_input);
    kf_output.init_dyn_share_modified_3h(get_f_output, df_dx_output, h_model_output, h_model_IMU_output);
    Eigen::Matrix<double, 24, 24> P_init; // = MD(18, 18)::Identity() * 0.1;
    reset_cov(P_init);
    kf_input.change_P(P_init);
    Eigen::Matrix<double, 30, 30> P_init_output; // = MD(24, 24)::Identity() * 0.01;
    reset_cov_output(P_init_output);
    kf_output.change_P(P_init_output);
    Eigen::Matrix<double, 24, 24> Q_input = process_noise_cov_input();
    Eigen::Matrix<double, 30, 30> Q_output = process_noise_cov_output();
    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");
    open_file();

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);

    ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 1000);
    ros::Publisher pubLaserCloudFullRes_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 1000);
    // ros::Publisher pubLaserCloudEffect  = nh.advertise<sensor_msgs::PointCloud2>
            // ("/cloud_effected", 1000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 1000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/aft_mapped_to_init", 1000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 1000);
    // ros::Publisher plane_pub = nh.advertise<visualization_msgs::Marker>
            // ("/planner_normal", 1000);
    ros::Publisher pubPoseDetect    = nh.advertise<geometry_msgs::PoseStamped> 
            ("/pose_for_detect", 1000);
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate loop_rate(500);
    bool status = ros::ok();

    // gtsam初始化
    if(ifLoop)
    {
        gtsam::ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.1;
        parameters.relinearizeSkip = 1;
        isam = new gtsam::ISAM2(parameters);

        // 回环检测进程
        std::thread loopthread(loopClosureThread);

        cloudKeyPoses3D.reset(new pcl::PointCloud<PointType3D>());
        cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        copy_cloudKeyPoses3D.reset(new pcl::PointCloud<PointType3D>());
        copy_cloudKeyPoses6D.reset(new pcl::PointCloud<PointTypePose>());
        kdtreeHistoryKeyPoses.reset(new pcl::KdTreeFLANN<PointType3D>());
        ICPPoses3D.reset(new pcl::PointCloud<PointType3D>());

        pubHistoryKeyFrames   = nh.advertise<sensor_msgs::PointCloud2>("/icp_loop_closure_history_cloud", 1);
        pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints", 1);
        pubLoopOdom = nh.advertise<nav_msgs::Odometry>("/loop_odom", 1);
        pubLoopPath = nh.advertise<nav_msgs::Path>("/loop_path", 1);
        pubLoopCloud = nh.advertise<sensor_msgs::PointCloud2>("/loop_cloud", 1);
        loopthread.detach();
    }

    if(ifFloorDetect)
    {
        // 楼层检测进程
        std::thread floorthread(floorDetectThread);
        floorthread.detach();
    }

    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        // 有数据
        if(sync_packages(Measures)) 
        {
            if (flg_reset)
            {
                ROS_WARN("reset when rosbag play back");
                p_imu->Reset();
                feats_undistort.reset(new PointCloudXYZI());
                if (use_imu_as_input)
                {
                    // state_in = kf_input.get_x();
                    state_in = state_input();
                    kf_input.change_P(P_init);
                }
                else
                {
                    // state_out = kf_output.get_x();
                    state_out = state_output();
                    kf_output.change_P(P_init_output);
                }
                flg_first_scan = true;
                is_first_frame = true;
                flg_reset = false;
                init_map = false;
                
                {
                    ivox_.reset(new IVoxType(ivox_options_));
                }
            }

            // 第一次雷达数据
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                flg_first_scan = false;
                if (first_imu_time < 1)
                {
                    first_imu_time = imu_next.header.stamp.toSec();
                    printf("first imu time: %f\n", first_imu_time);
                }
                time_current = 0.0;
                if(imu_en)
                {
                    // imu_next = *(imu_deque.front());
                    kf_input.x_.gravity << VEC_FROM_ARRAY(gravity);
                    kf_output.x_.gravity << VEC_FROM_ARRAY(gravity);
                    // kf_output.x_.acc << VEC_FROM_ARRAY(gravity);
                    // kf_output.x_.acc *= -1; 

                    {
                        while (Measures.lidar_beg_time > imu_next.header.stamp.toSec()) // if it is needed for the new map?
                        {
                            imu_deque.pop_front();
                            if (imu_deque.empty())
                            {
                                break;
                            }
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            // imu_deque.pop();
                        }
                    }
                }
                else
                {
                    kf_input.x_.gravity << VEC_FROM_ARRAY(gravity); // _init);
                    kf_output.x_.gravity << VEC_FROM_ARRAY(gravity); //_init);
                    kf_output.x_.acc << VEC_FROM_ARRAY(gravity); //_init);
                    kf_output.x_.acc *= -1; 
                    p_imu->imu_need_init_ = false;
                    // p_imu->after_imu_init_ = true;
                }     
                G_m_s2 = std::sqrt(gravity[0] * gravity[0] + gravity[1] * gravity[1] + gravity[2] * gravity[2]);
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start;
            match_time = 0;
            solve_time = 0;
            propag_time = 0;
            update_time = 0;
            t0 = omp_get_wtime();
            
            /*** 降采样以及IMU初始化 ***/
            t1 = omp_get_wtime();
            p_imu->Process(Measures, feats_undistort);
            
            if(space_down_sample)
            {
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);
                sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list); 
            }
            else
            {
                feats_down_body = Measures.lidar;
                sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list); 
            }
            
            time_seq = time_compressing<int>(feats_down_body);
            feats_down_size = feats_down_body->points.size();

            body_save_detect.reset(new PointCloudXYZI());
            body_save_detect = Measures.lidar;
            int body_save_detect_size = body_save_detect->points.size();
            world_save_detect->resize(body_save_detect_size);
            
            if (!p_imu->after_imu_init_)
            {
                if (!p_imu->imu_need_init_)
                { 
                    V3D tmp_gravity;
                    if (imu_en)
                    {tmp_gravity = - p_imu->mean_acc / p_imu->mean_acc.norm() * G_m_s2;}
                    else
                    {tmp_gravity << VEC_FROM_ARRAY(gravity_init);
                    p_imu->after_imu_init_ = true;
                    }
                    // V3D tmp_gravity << VEC_FROM_ARRAY(gravity_init);
                    M3D rot_init;
                    p_imu->Set_init(tmp_gravity, rot_init);
                    kf_input.x_.rot = rot_init;
                    kf_output.x_.rot = rot_init;
                    // kf_input.x_.rot; //.normalize();
                    // kf_output.x_.rot; //.normalize();
                    kf_output.x_.acc = - rot_init.transpose() * kf_output.x_.gravity;
                }
                else{
                continue;}
            }
            
            /*** 初始化地图 ***/
            if(!init_map)
            {
                feats_down_world->resize(feats_undistort->size());
                for(int i = 0; i < feats_undistort->size(); i++)
                {
                    {
                        pointBodyToWorld(&(feats_undistort->points[i]), &(feats_down_world->points[i]));
                    }
                }
                for (size_t i = 0; i < feats_down_world->size(); i++) 
                {
                    init_feats_world->points.emplace_back(feats_down_world->points[i]);
                }
                if(init_feats_world->size() < init_map_size) 
                {init_map = false;}
                else
                {   
                    ivox_->AddPoints(init_feats_world->points);
                    publish_init_map(pubLaserCloudMap); //(pubLaserCloudFullRes);
                    
                    init_feats_world.reset(new PointCloudXYZI());
                    init_map = true;
                }
                continue;
            }

            /*** ICP and Kalman filter update ***/
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            crossmat_list.reserve(feats_down_size);
            pbody_list.reserve(feats_down_size);

            // 雷达数据转换至IMU坐标系下           
            for (size_t i = 0; i < feats_down_body->size(); i++)
            {
                V3D point_this(feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z);
                pbody_list[i]=point_this;           
                if (!extrinsic_est_en)
                {
                    point_this = Lidar_R_wrt_IMU * point_this + Lidar_T_wrt_IMU;
                    M3D point_crossmat;
                    point_crossmat << SKEW_SYM_MATRX(point_this);
                    crossmat_list[i]=point_crossmat;
                }
            }

            // 更新
            if (!use_imu_as_input)
            {     
                bool imu_upda_cov = false;
                effct_feat_num = 0;
                /**** point by point update ****/
                if (time_seq.size() > 0)
                {
                    double pcl_beg_time = Measures.lidar_beg_time;
                    idx = -1;
                    for (k = 0; k < time_seq.size(); k++)
                    {
                        PointType &point_body  = feats_down_body->points[idx+time_seq[k]];

                        time_current = point_body.curvature / 1000.0 + pcl_beg_time;

                        if (is_first_frame)
                        {
                            if(imu_en)
                            {
                                while (time_current > imu_next.header.stamp.toSec())
                                {
                                    imu_deque.pop_front();
                                    if (imu_deque.empty()) break;
                                    imu_last = imu_next;
                                    imu_next = *(imu_deque.front());
                                }
                                angvel_avr<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                                acc_avr   <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                            }
                            is_first_frame = false;
                            imu_upda_cov = true;
                            time_update_last = time_current;
                            time_predict_last_const = time_current;
                        }
                        
                        if(imu_en && !imu_deque.empty())
                        {
                            bool last_imu = imu_next.header.stamp.toSec() == imu_deque.front()->header.stamp.toSec();
                            while (imu_next.header.stamp.toSec() < time_predict_last_const && !imu_deque.empty())
                            {
                                if (!last_imu)
                                {
                                    imu_last = imu_next;
                                    imu_next = *(imu_deque.front());
                                    break;
                                }
                                else
                                {
                                    imu_deque.pop_front();
                                    if (imu_deque.empty()) break;
                                    imu_last = imu_next;
                                    imu_next = *(imu_deque.front());
                                }
                            }
                            bool imu_comes = time_current > imu_next.header.stamp.toSec();
                            while (imu_comes) 
                            {
                                imu_upda_cov = true;
                                angvel_avr<<imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                                acc_avr   <<imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;

                                /*** covariance update ***/
                                double dt = imu_next.header.stamp.toSec() - time_predict_last_const;
                                kf_output.predict(dt, Q_output, input_in, true, false);
                                time_predict_last_const = imu_next.header.stamp.toSec(); // big problem
                                
                                {
                                    double dt_cov = imu_next.header.stamp.toSec() - time_update_last; 

                                    if (dt_cov > 0.0)
                                    {
                                        time_update_last = imu_next.header.stamp.toSec();
                                        double propag_imu_start = omp_get_wtime();

                                        kf_output.predict(dt_cov, Q_output, input_in, false, true);

                                        propag_time += omp_get_wtime() - propag_imu_start;
                                        double solve_imu_start = omp_get_wtime();
                                        kf_output.update_iterated_dyn_share_IMU();
                                        solve_time += omp_get_wtime() - solve_imu_start;
                                    }
                                }
                                imu_deque.pop_front();
                                if (imu_deque.empty()) break;
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                                imu_comes = time_current > imu_next.header.stamp.toSec();
                            }
                        }
                        
                        if (flg_reset)
                        {
                            break;
                        }

                        double dt = time_current - time_predict_last_const;
                        double propag_state_start = omp_get_wtime();
                        if(!prop_at_freq_of_imu)
                        {
                            double dt_cov = time_current - time_update_last;
                            if (dt_cov > 0.0)
                            {
                                kf_output.predict(dt_cov, Q_output, input_in, false, true);
                                time_update_last = time_current;   
                            }
                        }
                        
                        kf_output.predict(dt, Q_output, input_in, true, false);
                        propag_time += omp_get_wtime() - propag_state_start;
                        time_predict_last_const = time_current;
                        double t_update_start = omp_get_wtime();

                        if (feats_down_size < 1)
                        {
                            ROS_WARN("No point, skip this scan!\n");
                            idx += time_seq[k];
                            continue;
                        }

                        if (!kf_output.update_iterated_dyn_share_modified()) 
                        {
                            idx = idx+time_seq[k];
                            continue;
                        }
                        solve_start = omp_get_wtime();
                            
                        if (publish_odometry_without_downsample)
                        {
                            /******* Publish odometry *******/

                            publish_odometry(pubOdomAftMapped);
                            if (runtime_pos_log)
                            {
                                euler_cur = SO3ToEuler(kf_output.x_.rot);
                                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_output.x_.pos.transpose() << " " << kf_output.x_.vel.transpose() \
                                <<" "<<kf_output.x_.omg.transpose()<<" "<<kf_output.x_.acc.transpose()<<" "<<kf_output.x_.gravity.transpose()<<" "<<kf_output.x_.bg.transpose()<<" "<<kf_output.x_.ba.transpose()<<" "<<feats_undistort->points.size()<<endl;
                            }
                        }

                        for (int j = 0; j < time_seq[k]; j++)
                        {
                            PointType &point_body_j  = feats_down_body->points[idx+j+1];
                            PointType &point_world_j = feats_down_world->points[idx+j+1];
                            pointBodyToWorld(&point_body_j, &point_world_j);

                            PointType &body_save = body_save_detect->points[idx+j+1];
                            PointType &world_save = world_save_detect->points[idx+j+1];
                            pointBodyToWorld(&body_save, &world_save);
                        }
                    
                        solve_time += omp_get_wtime() - solve_start;
        
                        update_time += omp_get_wtime() - t_update_start;
                        idx += time_seq[k];
                    }
                }
                else
                {
                    if (!imu_deque.empty())
                    { 
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        while (imu_next.header.stamp.toSec() > time_current && ((imu_next.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte )))
                        { // >= ?
                            if (is_first_frame)
                            {
                                {
                                    {
                                        while (imu_next.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte)
                                        {
                                            // meas.imu.emplace_back(imu_deque.front()); should add to initialization
                                            imu_deque.pop_front();
                                            if(imu_deque.empty()) break;
                                            imu_last = imu_next;
                                            imu_next = *(imu_deque.front());
                                        }
                                    }
                                    break;
                                }
                                angvel_avr<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                                                
                                acc_avr   <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;

                                imu_upda_cov = true;
                                time_update_last = time_current;
                                time_predict_last_const = time_current;

                                    is_first_frame = false;
                            }
                            time_current = imu_next.header.stamp.toSec();

                            if (!is_first_frame)
                            {
                            double dt = time_current - time_predict_last_const;
                            {
                                double dt_cov = time_current - time_update_last;
                                if (dt_cov > 0.0)
                                {
                                    kf_output.predict(dt_cov, Q_output, input_in, false, true);
                                    time_update_last = time_current;
                                }
                                kf_output.predict(dt, Q_output, input_in, true, false);
                            }

                            time_predict_last_const = time_current;

                            angvel_avr<<imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                            acc_avr   <<imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z; 
                            // acc_avr_norm = acc_avr * G_m_s2 / acc_norm;
                            kf_output.update_iterated_dyn_share_IMU();
                            imu_deque.pop_front();
                            if (imu_deque.empty()) break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        }
                        else
                        {
                            imu_deque.pop_front();
                            if (imu_deque.empty()) break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        }
                        }
                    }
                }
            }
            else
            {
                bool imu_prop_cov = false;
                effct_feat_num = 0;
                if (time_seq.size() > 0)
                {
                double pcl_beg_time = Measures.lidar_beg_time;
                idx = -1;
                for (k = 0; k < time_seq.size(); k++)
                {
                    PointType &point_body  = feats_down_body->points[idx+time_seq[k]];
                    time_current = point_body.curvature / 1000.0 + pcl_beg_time;
                    if (is_first_frame)
                    {
                        while (time_current > imu_next.header.stamp.toSec()) 
                        {
                            imu_deque.pop_front();
                            if (imu_deque.empty()) break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        }
                        imu_prop_cov = true;

                        is_first_frame = false;
                        t_last = time_current;
                        time_update_last = time_current; 
                        {
                            input_in.gyro<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;                 
                            input_in.acc<<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                            input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                        }
                    }
                    
                    while (time_current > imu_next.header.stamp.toSec()) // && !imu_deque.empty())
                    {
                        imu_deque.pop_front();
                        
                        input_in.gyro<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                        input_in.acc <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z; 
                        input_in.acc    = input_in.acc * G_m_s2 / acc_norm; 
                        double dt = imu_last.header.stamp.toSec() - t_last;

                        double dt_cov = imu_last.header.stamp.toSec() - time_update_last;
                        if (dt_cov > 0.0)
                        {
                            kf_input.predict(dt_cov, Q_input, input_in, false, true); 
                            time_update_last = imu_last.header.stamp.toSec(); //time_current;
                        }
                        kf_input.predict(dt, Q_input, input_in, true, false); 
                        t_last = imu_last.header.stamp.toSec();
                        imu_prop_cov = true;

                        if (imu_deque.empty()) break;
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        // imu_upda_cov = true;
                    }     
                    if (flg_reset)
                    {
                        break;
                    }     
                    double dt = time_current - t_last;
                    t_last = time_current;
                    double propag_start = omp_get_wtime();
                    
                    if(!prop_at_freq_of_imu)
                    {   
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0)
                        {    
                            kf_input.predict(dt_cov, Q_input, input_in, false, true); 
                            time_update_last = time_current; 
                        }
                    }
                    kf_input.predict(dt, Q_input, input_in, true, false); 

                    propag_time += omp_get_wtime() - propag_start;

                    double t_update_start = omp_get_wtime();
                    
                    if (feats_down_size < 1)
                    {
                        ROS_WARN("No point, skip this scan!\n");

                        idx += time_seq[k];
                        continue;
                    }
                    if (!kf_input.update_iterated_dyn_share_modified()) 
                    {
                        idx = idx+time_seq[k];
                        continue;
                    }

                    solve_start = omp_get_wtime();

                    if (publish_odometry_without_downsample)
                    {
                        /******* Publish odometry *******/

                        publish_odometry(pubOdomAftMapped);
                        if (runtime_pos_log)
                        {
                            euler_cur = SO3ToEuler(kf_input.x_.rot);
                            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_input.x_.pos.transpose() << " " << kf_input.x_.vel.transpose() \
                            <<" "<<kf_input.x_.bg.transpose()<<" "<<kf_input.x_.ba.transpose()<<" "<<kf_input.x_.gravity.transpose()<<" "<<feats_undistort->points.size()<<endl;
                        }
                    }

                    for (int j = 0; j < time_seq[k]; j++)
                    {
                        PointType &point_body_j  = feats_down_body->points[idx+j+1];
                        PointType &point_world_j = feats_down_world->points[idx+j+1];
                        pointBodyToWorld(&point_body_j, &point_world_j); 
                    }
                    solve_time += omp_get_wtime() - solve_start;
                
                    update_time += omp_get_wtime() - t_update_start;
                    idx = idx + time_seq[k];
                }  
                }
                else
                {
                    if (!imu_deque.empty())
                    { 
                    imu_last = imu_next;
                    imu_next = *(imu_deque.front());
                    while (imu_next.header.stamp.toSec() > time_current && ((imu_next.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte)))
                    { // >= ?
                        if (is_first_frame)
                        {
                            {
                                {
                                    while (imu_next.header.stamp.toSec() < Measures.lidar_beg_time + lidar_time_inte)
                                    {
                                        imu_deque.pop_front();
                                        if(imu_deque.empty()) break;
                                        imu_last = imu_next;
                                        imu_next = *(imu_deque.front());
                                    }
                                }
                                
                                break;
                            }
                            imu_prop_cov = true;
                            
                            t_last = time_current;
                            time_update_last = time_current; 
                            input_in.gyro<<imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            input_in.acc   <<imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                            input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                            
                                is_first_frame = false;
                            
                        }
                        time_current = imu_next.header.stamp.toSec();

                        if (!is_first_frame)
                        {
                        double dt = time_current - t_last;

                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0)
                        {        
                            // kf_input.predict(dt_cov, Q_input, input_in, false, true);
                            time_update_last = imu_next.header.stamp.toSec(); //time_current;
                        }
                        // kf_input.predict(dt, Q_input, input_in, true, false);

                        t_last = imu_next.header.stamp.toSec();
                    
                        input_in.gyro<<imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                        input_in.acc<<imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z; 
                        input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                        imu_deque.pop_front();
                        if (imu_deque.empty()) break;
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        }
                        else
                        {
                            imu_deque.pop_front();
                            if (imu_deque.empty()) break;
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                        }
                    }
                    }
                }
            }

            /******* loopClosure *******/
            

            if(ifLoop)
            {
                LoopandOptmization();
                double timeNow = ros::Time::now().toSec();
                // if(findLoop)
                // {
                //     Eigen::Vector3d mat1(state_point.pos(0), state_point.pos(1), state_point.pos(2));
                //     Eigen::Vector3d mat2(updateTransform[3], updateTransform[4], updateTransform[5]);
                //     double dis = (mat1 - mat2).norm();
                //     if(dis >= kdtreeResetDistance && timeNow - timeLast > kdtreeResetTime)
                //     {
    
                //         cout << "Remapping KDtree!" << endl;

                //         state_ikfom new_state;
                //         new_state = state_point;

                //         Eigen::Vector3d pos_(updateTransform[3], updateTransform[4], updateTransform[5]);
                //         tf::Quaternion q_ = tf::createQuaternionFromRPY(updateTransform[0],  updateTransform[1], updateTransform[2]);
                //         Eigen::Quaterniond q(q_.w(), q_.x(), q_.y(), q_.z()); 
                //         new_state.pos = pos_;
                //         new_state.rot = q;
                //         euler_cur = SO3ToEuler(new_state.rot);
                //         pos_lid = new_state.pos + new_state.rot * new_state.offset_T_L_I;
                //         geoQuat.x = new_state.rot.coeffs()[0];
                //         geoQuat.y = new_state.rot.coeffs()[1];
                //         geoQuat.z = new_state.rot.coeffs()[2];
                //         geoQuat.w = new_state.rot.coeffs()[3];
                //         kf.change_x(new_state);
                //         state_point = kf.get_x();
                //         feats_down_body->clear();
                //         feats_down_world->clear();
                //         kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);
                //         timeLast = timeNow;
  
                //         last_roll =  updateTransform[0];
                //         last_pitch = updateTransform[1];
                //         last_yaw =   updateTransform[2];
                //         last_x = updateTransform[3];
                //         last_y = updateTransform[4];
                //         last_z = updateTransform[5];

                //         Reset = true;
                //     } 
                // }
            }
            /******* Publish odometry downsample *******/
            if (!publish_odometry_without_downsample)
            {
                publish_odometry(pubOdomAftMapped);
            }

            /*** add the feature points to map ***/
            t3 = omp_get_wtime();
            
            if(feats_down_size > 4)
            {
                MapIncremental();
            }

            t5 = omp_get_wtime();
            /******* Publish points *******/
            if (path_en)                          publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)       publish_frame_world(pubLaserCloudFullRes);
            if (scan_pub_en && scan_body_pub_en)  publish_frame_body(pubLaserCloudFullRes_body);
            if (detect_object_en)                 save_pcd_for_detect(pubPoseDetect);                      
            
            /*** Debug variables Logging ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                {aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + update_time/frame_num;}
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + solve_time/frame_num;
                aver_time_propag = aver_time_propag * (frame_num - 1)/frame_num + propag_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = aver_time_consu;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f propogate: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu, aver_time_icp, aver_time_propag); 
                if (!publish_odometry_without_downsample)
                {
                    if (!use_imu_as_input)
                    {
                        euler_cur = SO3ToEuler(kf_output.x_.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_output.x_.pos.transpose() << " " << kf_output.x_.vel.transpose() \
                        <<" "<<kf_output.x_.omg.transpose()<<" "<<kf_output.x_.acc.transpose()<<" "<<kf_output.x_.gravity.transpose()<<" "<<kf_output.x_.bg.transpose()<<" "<<kf_output.x_.ba.transpose()<<" "<<feats_undistort->points.size()<<endl;
                    }
                    else
                    {
                        euler_cur = SO3ToEuler(kf_input.x_.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << kf_input.x_.pos.transpose() << " " << kf_input.x_.vel.transpose() \
                        <<" "<<kf_input.x_.bg.transpose()<<" "<<kf_input.x_.ba.transpose()<<" "<<kf_input.x_.gravity.transpose()<<" "<<feats_undistort->points.size()<<endl;
                    }
                }
                dump_lio_state_to_log(fp);
            }
        }
        status = ros::ok();
        loop_rate.sleep();
    }
    //--------------------------save map-----------------------------------
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }
    fout_out.close();
    fout_imu_pbp.close();
    return 0;
}
