#include <ceres/ceres.h>
#include <Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <memory>
#include <stdio.h>
#include "omp.h"
#include "open3d/Open3D.h"

using namespace open3d;

struct ObjectInfo {
    std::string id;
    Eigen::Vector3d center;
    Eigen::Vector3d size;
};

std::tuple<std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<pipelines::registration::Feature>>
PreprocessPointCloud(const char *file_name, const float voxel_size) {
    auto pcd = open3d::io::CreatePointCloudFromFile(file_name);
    auto pcd_down = pcd->VoxelDownSample(voxel_size);
    pcd_down->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(2 * voxel_size, 30));
    auto pcd_fpfh = pipelines::registration::ComputeFPFHFeature(
            *pcd_down,
            open3d::geometry::KDTreeSearchParamHybrid(5 * voxel_size, 100));
    return std::make_tuple(pcd, pcd_down, pcd_fpfh);
}

std::tuple<std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<pipelines::registration::Feature>>
PreprocessPointCloud(const open3d::geometry::PointCloud &pcd, const float voxel_size) {
    std::shared_ptr<geometry::PointCloud> pcd_ptr(new geometry::PointCloud);
    *pcd_ptr = pcd;
    auto pcd_down = pcd_ptr->VoxelDownSample(voxel_size);
    pcd_down->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(2 * voxel_size, 30));
    auto pcd_fpfh = pipelines::registration::ComputeFPFHFeature(
            *pcd_down,
            open3d::geometry::KDTreeSearchParamHybrid(5 * voxel_size, 100));
    return std::make_tuple( pcd_down, pcd_fpfh);
}

std::tuple<std::shared_ptr<geometry::PointCloud>>
PreprocessPointCloud_(const char *file_name, const float voxel_size) {
    auto pcd = open3d::io::CreatePointCloudFromFile(file_name);
    auto pcd_down = pcd->VoxelDownSample(voxel_size);
    pcd_down->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(2 * voxel_size, 30));
    return std::make_tuple( pcd_down);
}


struct PointCloudAlignmentError {
    PointCloudAlignmentError(const Eigen::Vector3d& observed_point) : observed_point(observed_point) {}

    template <typename T>
    bool operator()(const T* const transform, const T* const point, T* residuals) const {
        T transformed_point[3];
        ceres::AngleAxisRotatePoint(transform, point, transformed_point);

        transformed_point[0] += transform[3];
        transformed_point[1] += transform[4];
        transformed_point[2] += transform[5];

        residuals[0] = transformed_point[0] - T(observed_point[0]);
        residuals[1] = transformed_point[1] - T(observed_point[1]);
        residuals[2] = transformed_point[2] - T(observed_point[2]);

        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& observed_point) {
        return (new ceres::AutoDiffCostFunction<PointCloudAlignmentError, 3, 6, 3>(
            new PointCloudAlignmentError(observed_point)));
    }

    Eigen::Vector3d observed_point;
};

void GlobalOptimization(std::shared_ptr<geometry::PointCloud> source,
                        std::shared_ptr<geometry::PointCloud> target,
                        Eigen::Matrix4d& initial_transform) {
    ceres::Problem problem;

    for (size_t i = 0; i < source->points_.size(); ++i) {
        Eigen::Vector3d source_point = source->points_[i];
        Eigen::Vector3d target_point = target->points_[i];

        ceres::CostFunction* cost_function =
            PointCloudAlignmentError::Create(target_point);

        problem.AddResidualBlock(cost_function, nullptr, initial_transform.data(), source_point.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}

void VisualizeRegistration(const open3d::geometry::PointCloud &source,
                           const open3d::geometry::PointCloud &target,
                           const Eigen::Matrix4d &Transformation) {
    std::shared_ptr<geometry::PointCloud> source_transformed_ptr(
            new geometry::PointCloud);
    std::shared_ptr<geometry::PointCloud> target_ptr(new geometry::PointCloud);
    *source_transformed_ptr = source;
    source_transformed_ptr->PaintUniformColor(Eigen::Vector3d(1, 0.706, 0));
    *target_ptr = target;
    target_ptr->PaintUniformColor(Eigen::Vector3d(0, 0.651, 0.929));
    source_transformed_ptr->Transform(Transformation);
    visualization::DrawGeometries({source_transformed_ptr, target_ptr},
                                  "Registration result");
}


double ComputeRMSE(const open3d::geometry::PointCloud &source,const open3d::geometry::PointCloud& target, Eigen::Matrix4d_u &Transformation)
{
    geometry::PointCloud source_transformed = source;
    source_transformed.Transform(Transformation);
    
    auto result = open3d::pipelines::registration::EvaluateRegistration(source_transformed, target,0.5, Transformation);
    
    std::cout << "RMSE: " << result.inlier_rmse_ << std::endl;
    // 返回RMSE
    return result.inlier_rmse_;

}

std::vector<ObjectInfo> ReadObjectsFromFile(const std::string& file_path) {
    std::vector<ObjectInfo> objects;
    std::ifstream file(file_path);
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string id;
        std::string coordinates;
        if (std::getline(iss, id, ':') && std::getline(iss, coordinates)) {
            std::istringstream coord_stream(coordinates);
            std::vector<double> coords;
            std::string value;
            while (std::getline(coord_stream, value, ',')) {
                coords.push_back(std::stod(value));
            }

            if (coords.size() == 6) {
                ObjectInfo obj;
                obj.id = id;
                obj.center = Eigen::Vector3d(coords[0], coords[1], coords[2]);
                obj.size = Eigen::Vector3d(coords[3], coords[4], coords[5]);
                objects.push_back(obj);
            } else {
                std::cerr << "Error: Expected 6 values, but got " << coords.size() << std::endl;
            }
        } else {
            std::cerr << "Error: Failed to parse line: " << line << std::endl;
        }
    }

    return objects;
}

bool CompareObjectSizes(const Eigen::Vector3d& size1, const Eigen::Vector3d& size2, double tolerance = 0.05) {
    return (std::abs(size1.x() - size2.x()) < tolerance &&
            std::abs(size1.y() - size2.y()) < tolerance &&
            std::abs(size1.z() - size2.z()) < tolerance);
}

std::shared_ptr<geometry::PointCloud> ExtractNearbyPoints(const std::shared_ptr<geometry::PointCloud>& cloud, const Eigen::Vector3d& center) {
    auto bbox = geometry::AxisAlignedBoundingBox(center - Eigen::Vector3d(9,9,3), center + Eigen::Vector3d(9,9,3));
    return cloud->Crop(bbox);
}

std::shared_ptr<geometry::PointCloud> ProcessCurrentObject(const ObjectInfo& current_object, const std::vector<ObjectInfo>& map_objects, const std::shared_ptr<geometry::PointCloud>& global_map) {
    auto current_center = current_object.center;
    auto local_combined_cloud = std::make_shared<geometry::PointCloud>();
        std::cout<<"object choose "<<std::endl;
        int i = 0;
    for (const auto& map_object : map_objects) {
        auto map_center = map_object.center;
        std::cout<<"i "<<i++<<std::endl;
        if (CompareObjectSizes(current_object.size,map_object.size)) { 
            auto nearby_points = ExtractNearbyPoints(global_map, map_center);
        //     FixHeight(nearby_points);
            *local_combined_cloud += *nearby_points;
             local_combined_cloud->RemoveDuplicatedPoints();
        }
    }
    // 去除重复点
    return local_combined_cloud;
}

int main(int argc, char *argv[]) {
    
    char* map_pcd = "/home/bnxy/opene3d_ws/src/pcd/object_map.pcd";
    char* test_pcd ="/home/bnxy/opene3d_ws/src/pcd/object_test6.pcd";

    bool mutual_filter = true;

    float voxel_size = 0.5;
    float distance_multiplier = 1;
    float distance_threshold = 1.0;
    int max_iterations =  4000000;
    float confidence =  0.999;
    double t0,t1,t2,t3,t4,t5;
    // Prepare input
    std::shared_ptr<geometry::PointCloud> map, map_down, target,
            target_down;
    std::shared_ptr<pipelines::registration::Feature>  target_fpfh;
    t0 = omp_get_wtime();
    std::tie(map_down) =
            PreprocessPointCloud_(map_pcd, voxel_size);
    t1 = omp_get_wtime();
    std::cout<<"map voxel_down_time = "<<t1-t0<<std::endl;
    std::tie(target, target_down, target_fpfh) =
            PreprocessPointCloud(test_pcd, voxel_size);
    t2 = omp_get_wtime();
    std::cout<<" target voxel_down_time = "<<t2-t1<<std::endl;

    auto map_objects = ReadObjectsFromFile("/home/bnxy/opene3d_ws/src/pcd/detected_objects_info.txt");
    auto current_objects = ReadObjectsFromFile("/home/bnxy/opene3d_ws/src/pcd/current_objects_info.txt");
    
    auto combined_cloud = std::make_shared<geometry::PointCloud>();
        std::cout<<"current_objects.size()"<<current_objects.size()<<std::endl;
    for (size_t i = 0; i < current_objects.size(); ++i) {
        auto local_combined_cloud = ProcessCurrentObject(current_objects[i], map_objects, map_down);
        *combined_cloud += *local_combined_cloud;
    }
    t3 = omp_get_wtime();
    std::cout<<"object cloud took = "<<t3-t2<<std::endl;


//     std::cout<<source_down->points_.size()<<std::endl;
//     std::cout<<target_down->points_.size()<<std::endl;
        std::shared_ptr<geometry::PointCloud> source = combined_cloud;
        std::shared_ptr<geometry::PointCloud> source_down;
        std::shared_ptr<pipelines::registration::Feature> source_fpfh;
        std::tie(source_down,source_fpfh) = PreprocessPointCloud(*source,voxel_size);

       
    // Prepare checkers
    pipelines::registration::RegistrationResult registration_result;
    std::vector<std::reference_wrapper<
            const pipelines::registration::CorrespondenceChecker>>
            correspondence_checker;
    auto correspondence_checker_edge_length =
            pipelines::registration::CorrespondenceCheckerBasedOnEdgeLength(
                    0.9);
    auto correspondence_checker_distance =
            pipelines::registration::CorrespondenceCheckerBasedOnDistance(
                    distance_threshold);
    correspondence_checker.push_back(correspondence_checker_edge_length);
    correspondence_checker.push_back(correspondence_checker_distance);

        t4 =  omp_get_wtime();
        // Manually search correspondences
        int nPti = int(source_down->points_.size());
        int nPtj = int(target_down->points_.size());
        //目标点云和源点云的KDTree
        geometry::KDTreeFlann feature_tree_i(*source_fpfh);
        geometry::KDTreeFlann feature_tree_j(*target_fpfh);

        pipelines::registration::CorrespondenceSet corres_ji;
        std::vector<int> i_to_j(nPti, -1);

        // Buffer all correspondences
        for (int j = 0; j < nPtj; j++) {
            std::vector<int> corres_tmp(1);
            std::vector<double> dist_tmp(1);

            feature_tree_i.SearchKNN(Eigen::VectorXd(target_fpfh->data_.col(j)),
                                     1, corres_tmp, dist_tmp);
            int i = corres_tmp[0];
            corres_ji.push_back(Eigen::Vector2i(i, j));
        }
        pipelines::registration::CorrespondenceSet mutual_corres;
        for (auto &corres : corres_ji) {
        int j = corres(1);
        int j2i = corres(0);

        std::vector<int> corres_tmp(1);
        std::vector<double> dist_tmp(1);
        feature_tree_j.SearchKNN(
                Eigen::VectorXd(source_fpfh->data_.col(j2i)), 1,
                corres_tmp, dist_tmp);
        int i2j = corres_tmp[0];
        if (i2j == j) {
                mutual_corres.push_back(corres);
        }
        }

        // utility::LogDebug("{:d} points remain after mutual filter",
        //                   mutual_corres.size());
        registration_result = pipelines::registration::
                RegistrationRANSACBasedOnCorrespondence(
                        *source_down, *target_down, mutual_corres,
                        distance_threshold,
                        pipelines::registration::
                                TransformationEstimationPointToPoint(true),
                        3, correspondence_checker,
                        pipelines::registration::RANSACConvergenceCriteria(
                                max_iterations, confidence));
                                t5 = omp_get_wtime();
                                std::cout<<"RANSAC took time = "<<t5-t4<<std::endl;

        std::cout << "transformation " << registration_result.transformation_ << std::endl;
        std::cout << "rmse= " << registration_result.inlier_rmse_ << std::endl;
    double rmse = ComputeRMSE(*source, *target, registration_result.transformation_);
    

    VisualizeRegistration(*source_down, *target_down,
                          registration_result.transformation_);
    
    return 0;
}
