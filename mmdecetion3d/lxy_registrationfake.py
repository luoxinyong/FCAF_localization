#!/usr/bin/env python
import open3d as o3d
import numpy as np
import time

cls_names = ['table', 'chair', 'sofa', 'bookshelf', 'bed']
class PointCloudLocalization:
    def __init__(self):
        t0 = time.time()
        print("read map and objects'")
        self.global_map = self.load_point_cloud('object_map.pcd')
        self.map_objects = self.read_objects_from_file('detected_objects_info.txt')
        self.map_centers = self.calculate_object_center(self.map_objects)
        self.map_sizes = self.calculate_all_object_sizes(self.map_objects)
        print(f"map size = {self.global_map}")
        t1 = time.time() - t0
        print(f"read map objects : {t1:.6f} seconds")
        # self.current_cloud = self.load_point_cloud('object_test2.pcd')
        self.current_objects = self.read_objects_from_file('current_objects_info.txt')
        self.current_centers = self.calculate_object_center(self.current_objects)
        
        # self.current_cloud = self.Pointcloud_process(self.current_cloud)
        self.robot_position = np.array([0, 0])  # 机器人当前位置
        self.save_local_pcd()

    def Pointcloud_process(self,pointcloud):
        cl,ind = pointcloud.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.0)
        return cl
    
    def load_point_cloud(self, file_path):
        return o3d.io.read_point_cloud(file_path)

    def calculate_object_center(self, objects):
        centers = {}
        for obj in objects:
            cls_id, coords = obj
            # x, y, z are the center coordinates of the object
            center = np.array([coords[0], coords[1], coords[2]])
            centers[cls_id] = center
        return centers
    
    def get_point_cloud_dimensions(point_cloud):
        aabb = point_cloud.get_axis_aligned_bounding_box()
        dimensions = aabb.get_extent()
        return dimensions
    
    def calculate_distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

    def read_objects_from_file(self, file_path):
        objects = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(':')
                cls_id = parts[0].strip()
                coordinates = list(map(float, parts[1].strip().split(',')))
                objects.append((cls_id, coordinates))
        return objects
        # 现在objects的格式应该是[
#     ('chair_0', [2.717749993006388, 0.9273694256941477, 0.12218259274959564, 0.6922661662101746, 0.7518542011578878, 1.0895437995592754]),
#     ('table_1', [1.23456789, 2.34567891, 3.45678912, 0.56789012, 0.67890123, 0.78901234])
# ]
    def calculate_all_object_sizes(self, objects):
        t0 = time.time()
        print("all size'")
        sizes = {}
        for obj in objects:
            cls_id, coords = obj
            size = self.calculate_object_size(coords)
            sizes[cls_id] = size
        
        t1 = time.time() - t0
        print(f"all size took : {t1:.6f} seconds")
        return sizes
    
    def calculate_object_size(self, coordinates):
        dx, dy, dz = coordinates[3], coordinates[4], coordinates[5]
        size = np.array([dx, dy, dz])
        return size
    
    def extract_nearby_points(self, point_cloud, center, size, distance_to_robot):
        print("save points'")
        aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=center - np.array([10,10,3]),
                                                   max_bound=center + np.array([10,10,3]))
        # search_radius = np.linalg.norm(size)/2  + distance_to_robot
        # print(f"search_radius{search_radius}'")
        # aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=center - search_radius,
        #                                            max_bound=center + search_radius)
        nearby_points = point_cloud.crop(aabb)
        return nearby_points
    
    def fix_height(self, point_cloud, height=3.0):
        points = np.asarray(point_cloud.points)
        points[:, 2] = height
        point_cloud.points = o3d.utility.Vector3dVector(points)
        return point_cloud
    
    def save_local_pcd(self):
        combined_pcd = o3d.geometry.PointCloud()
        t0 = time.time()
        i = 0
        print("save start'")
        local_point_clouds = []
        for current_id, current_coords in self.current_objects:
            current_center = self.current_centers[current_id]
            current_size = self.calculate_object_size(current_coords)
            distance_to_robot = self.calculate_distance(current_center[:2], self.robot_position)
            for map_obj in self.map_objects:
                print(f"map object i{i}'")
                map_id, map_coords = map_obj
                map_center = self.map_centers[map_id]
                map_size = self.map_sizes[map_id]
                i+=1
                if np.allclose(current_size, map_size, atol=0.05):  # Tolerance can be adjusted
                    nearby_points = self.extract_nearby_points(self.global_map, map_center, current_size, distance_to_robot)
                    # nearby_points = self.fix_height(nearby_points)
                    combined_pcd += nearby_points
                    combined_pcd = combined_pcd.remove_duplicated_points()
        
        if len(combined_pcd.points) > 0:
            o3d.io.write_point_cloud("combined_nearby_points.pcd", combined_pcd)
            t1 = time.time() - t0
            print(f"all done : {t1:.6f} seconds")
            print("Saved combined nearby points to 'combined_nearby_points.pcd'")
        else:
            print("No matching objects found to extract nearby points.")

if __name__ == '__main__':
        pcl_localization = PointCloudLocalization()

