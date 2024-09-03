#include <open3d/Open3D.h>
#include <Eigen/Dense>
#include <iostream>

using namespace open3d;
struct ObjectInfo {
    std::string id;
    Eigen::Vector3d center;
    Eigen::Vector3d size;
};

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
// 假设 points_n 是 n 个物体的中心点云
std::shared_ptr<geometry::PointCloud> CreatePointCloud(const std::vector<Eigen::Vector3d>& points) {
    auto pointcloud = std::make_shared<geometry::PointCloud>();
    for (const auto& point : points) {
        pointcloud->points_.push_back(point);
    }
    return pointcloud;
}

int main() {
    // n 个物体的中心点
    std::vector<Eigen::Vector3d> points_n = {
        {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0},
        // 添加更多的点...
    };
    auto source_cloud = CreatePointCloud(points_n);

    // 在另一个坐标系下找到的三个物体的位置
    std::vector<Eigen::Vector3d> points_3 = {
        {1.1, 2.1, 3.1}, {4.1, 5.1, 6.1}, {7.1, 8.1, 9.1},
    };
    auto target_cloud = CreatePointCloud(points_3);

    // FGR 配准
    pipelines::registration::FastGlobalRegistrationOption option;
    option.maximum_correspondence_distance_ = 0.5;  // 设置合理的距离阈值

    auto result = pipelines::registration::FastGlobalRegistration(*source_cloud, *target_cloud, option);

    std::cout << "Transformation: \n" << result.transformation_ << std::endl;

    // 根据配准结果，查找匹配的点
    geometry::PointCloud transformed_target_cloud = *target_cloud;
    transformed_target_cloud.Transform(result.transformation_);

    for (const auto& target_point : transformed_target_cloud.points_) {
        double min_distance = std::numeric_limits<double>::max();
        Eigen::Vector3d matched_point;
        for (const auto& source_point : source_cloud->points_) {
            double distance = (target_point - source_point).norm();
            if (distance < min_distance) {
                min_distance = distance;
                matched_point = source_point;
            }
        }
        std::cout << "Matched point: " << matched_point.transpose() << std::endl;
    }

    return 0;
}
