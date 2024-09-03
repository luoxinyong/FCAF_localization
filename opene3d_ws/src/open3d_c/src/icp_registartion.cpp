
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <memory>
#include <stdio.h>
#include "omp.h"
#include "open3d/Open3D.h"

using namespace open3d;

std::tuple<std::shared_ptr<geometry::PointCloud>,
           std::shared_ptr<geometry::PointCloud>>
PreprocessPointCloud(const char *file_name, const float voxel_size) {
    auto pcd = open3d::io::CreatePointCloudFromFile(file_name);
    auto pcd_down = pcd->VoxelDownSample(voxel_size);
    pcd_down->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(2 * voxel_size, 30));
//     auto pcd_fpfh = pipelines::registration::ComputeFPFHFeature(
//             *pcd_down,
//             open3d::geometry::KDTreeSearchParamHybrid(5 * voxel_size, 100));
    return std::make_tuple(pcd, pcd_down);
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

void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > RegistrationRANSAC source_pcd target_pcd"
                     "[--method=feature_matching] "
                     "[--voxel_size=0.05] [--distance_multiplier=1.5]"
                     "[--max_iterations 1000000] [--confidence 0.999]"
                     "[--mutual_filter]");
    // clang-format on
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
int main(int argc, char *argv[]) {
    using namespace open3d;

    // utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);

    char* map_pcd = "/home/bnxy/opene3d_ws/src/pcd/object_map.pcd";
    char* test_pcd ="/home/bnxy/opene3d_ws/src/pcd/object_test6.pcd";


    float voxel_size = 0.5;
    float distance_threshold = 0.5;
    double t0,t1,t2,t3;
    // Prepare input
    std::shared_ptr<geometry::PointCloud> source, source_down, target,
            target_down;
//     std::shared_ptr<pipelines::registration::Feature> source_fpfh, target_fpfh;
    t0 = omp_get_wtime();
    std::tie(source, source_down) =
            PreprocessPointCloud(map_pcd, voxel_size);
    std::tie(target, target_down) =
            PreprocessPointCloud(test_pcd, voxel_size);
    t1 = omp_get_wtime();
    std::cout<<"voxel_down_time = "<<t1-t0<<std::endl;

    std::cout<<source_down->points_.size()<<std::endl;
    std::cout<<target_down->points_.size()<<std::endl;
// ,15.773,-0.27137382328510284
    Eigen::Matrix4d TR = Eigen::Matrix4d::Identity();
    TR(0,3) = 11.64;
    TR(1,3) = 15.773;
    TR(2,3) = -0.27137382328510284;
    pipelines::registration::RegistrationResult registration_result;
    t2 = omp_get_wtime();
    registration_result  = open3d::pipelines::registration::RegistrationICP(*source_down,
                                                                *target_down,distance_threshold,
                                                                TR,
                                                                open3d::pipelines::registration::TransformationEstimationPointToPoint(),
                                                                open3d::pipelines::registration::ICPConvergenceCriteria(1e-6,1e-6,30));
    t3 = omp_get_wtime();
    std::cout<<"ICP took time = "<<t3-t2<<std::endl;
        
    std::cout << "transformation " << registration_result.transformation_ << std::endl;
    std::cout << "rmse= " << registration_result.inlier_rmse_ << std::endl;
    double rmse = ComputeRMSE(*source, *target, registration_result.transformation_);
    

    VisualizeRegistration(*source_down, *target_down,
                          registration_result.transformation_);
    
    return 0;
}
