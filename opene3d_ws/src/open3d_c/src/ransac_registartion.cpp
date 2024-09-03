
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <memory>
#include <stdio.h>
#include "omp.h"
#include "open3d/Open3D.h"

using namespace open3d;

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


    const std::string kMethodFeature = "feature_matching";
    std::string method = "correspondence";
    const std::string kMethodCorres = "correspondence";

    char* map_pcd = "../../pcd/combined_nearby_points.pcd";
    char* test_pcd ="../../pcd/object_test6.pcd";

    bool mutual_filter = true;

    float voxel_size = 0.5;
    float distance_multiplier = 1;
    float distance_threshold = 1.0;
    int max_iterations =  4000000;
    float confidence =  0.999;
    double t0,t1,t2,t3;
    // Prepare input
    std::shared_ptr<geometry::PointCloud> source, source_down, target,
            target_down;
    std::shared_ptr<pipelines::registration::Feature> source_fpfh, target_fpfh;
    t0 = omp_get_wtime();
    std::tie(source, source_down, source_fpfh) =
            PreprocessPointCloud(map_pcd, voxel_size);
    std::tie(target, target_down, target_fpfh) =
            PreprocessPointCloud(test_pcd, voxel_size);
    t1 = omp_get_wtime();
    std::cout<<"voxel_down_time = "<<t1-t0<<std::endl;

    std::cout<<source_down->points_.size()<<std::endl;
    std::cout<<target_down->points_.size()<<std::endl;

    pipelines::registration::RegistrationResult registration_result;

    // Prepare checkers
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

    if (method == kMethodFeature) {
        t2 =  omp_get_wtime();
        registration_result = pipelines::registration::
                RegistrationRANSACBasedOnFeatureMatching(
                        *source_down, *target_down, *source_fpfh, *target_fpfh,
                        mutual_filter, distance_threshold,
                        pipelines::registration::
                                TransformationEstimationPointToPoint(false),
                        3, correspondence_checker,
                        pipelines::registration::RANSACConvergenceCriteria(
                                max_iterations, confidence));
        t3 = omp_get_wtime();
        // std::cout<<"RANSAC took time = "<<t3-t2<<std::endl;
        // std::cout<<"rmse = "<<registration_result.inlier_rmse_<<std::endl;
    } else if (method == kMethodCorres) {
        t2 =  omp_get_wtime();
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

        if (mutual_filter) {
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
                                    t3 = omp_get_wtime();
                                    std::cout<<"RANSAC took time = "<<t3-t2<<std::endl;
        } else {
            // utility::LogDebug("{:d} points remain", corres_ji.size());
            registration_result = pipelines::registration::
                    RegistrationRANSACBasedOnCorrespondence(
                            *source_down, *target_down, corres_ji,
                            distance_threshold,
                            pipelines::registration::
                                    TransformationEstimationPointToPoint(false),
                            3, correspondence_checker,
                            pipelines::registration::RANSACConvergenceCriteria(
                                    max_iterations, confidence));
                                    t3 = omp_get_wtime();
                                    std::cout<<"RANSAC took time = "<<t3-t2<<std::endl;
        }
    }
        std::cout << "transformation " << registration_result.transformation_ << std::endl;
        std::cout << "rmse= " << registration_result.inlier_rmse_ << std::endl;
    double rmse = ComputeRMSE(*source, *target, registration_result.transformation_);
    

    VisualizeRegistration(*source_down, *target_down,
                          registration_result.transformation_);
    
    return 0;
}
