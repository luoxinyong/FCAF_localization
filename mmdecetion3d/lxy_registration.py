#!/usr/bin/env python
import rospy
from lxy_fcaf3d import FCAFDemo
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray
import open3d as o3d
import numpy as np
import random
import math
import time
import ros_numpy
cls_names = ['table', 'chair', 'sofa', 'bookshelf', 'bed']
class PointCloudLocalization:
    def __init__(self):
        self.z_threshold = 0.8
        self.current_cloud = None
        self.current_objects = []
        self.global_map = self.load_point_cloud('object_scan.pcd')
        # map_objects保存的是map里面保存的物体，从txt里面拿
        self.map_objects = self.read_objects_from_file('detected_objects_info.txt')
        # map_centers保存的是map里面所有物体的中心点位置
        self.map_centers = self.calculate_object_center(self.map_objects)
        self.points_array= self.read_pcd_and_extract_xyz('object_test1.pcd')
        self.current_cloud = o3d.io.read_point_cloud('object_test1.pcd')
        
        rospy.init_node('point_cloud_localization', anonymous=True)
        config = 'configs/fcaf3d/fcaf3d_2xb8_s3dis-3d-5class.py'
        checkpoint = 'work_dirs/fcaf3d_2xb8_s3dis-3d-5class/epoch_12.pth'
        self.pcd = o3d.geometry.PointCloud()
        self.fcaf_demo = FCAFDemo(config, checkpoint)
        # rospy.Subscriber('/current_objects', Float32MultiArray, self.objects_callback)
        self.result_pub = rospy.Publisher('/localization_result', Point, queue_size=10)
        self.point_cloud_callback(self.points_array)

    def load_point_cloud(self, file_path):
        return o3d.io.read_point_cloud(file_path)
    
    def read_pcd_and_extract_xyz(self,pcd_file):
        # 读取PCD文件
        start_time = time.time()
        pcd = o3d.io.read_point_cloud(pcd_file)
        
        # 获取点云的大小
        points = np.asarray(pcd.points)
        filtered_points = points[points[:, 2] <= self.z_threshold]
        print(f"Cloud size = : {len(filtered_points)}")
        # 初始化一个数组来存储XYZ坐标
        create_time = time.time() - start_time
        print(f"Time taken to read PCD file: {create_time:.6f} seconds")
        return filtered_points

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



    def calculate_object_center(self, objects):
        centers = []
        for obj in objects:
            center = obj[1][:3]  # 只取前三个值，即中心点坐标
            centers.append(center)
        return np.array(centers)

    def generate_particles(self, center, radius, num_particles=100):
        particles = []
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, radius)
            x = center[0] + r * math.cos(angle)
            y = center[1] + r * math.sin(angle)
            z = 0  # 保持z坐标不变，假设是2D平面
            particles.append([x, y, z])
        return particles

    def point_cloud_callback(self, data):
        rgb = np.ones((data.shape[0], 3), dtype=np.uint8) * 255
        self.pcd.points = o3d.utility.Vector3dVector(data)
        self.pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
        data_np = np.concatenate((np.array(self.pcd.points),rgb),axis=1)
        results,output_bboxes = self.fcaf_demo.infer(self.pcd,data_np)
        print("infer done")
        for i,(x, y, z, dx, dy, dz, angle, cls, score) in enumerate(output_bboxes):
            self.current_objects.append((cls_names[int(cls)],[x,y,z,dx,dy,dz,angle,score]))

        self.localize()

    def convert_ros_to_open3d(self, ros_point_cloud):
        points = list(pc2.read_points(ros_point_cloud, field_names=("x", "y", "z"), skip_nans=True))
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.array(points))
        return cloud

    def convert_ros_to_objects(self, data):
        objects = []
        for i in range(0, len(data), 6):
            cls_id = int(data[i])
            coordinates = data[i+1:i+6]
            objects.append((cls_id, coordinates))
        return objects

    def compute_fitness(self, current_cloud, particle):
        # 将粒子位置转换为变换矩阵
        transformation = np.eye(4)
        transformation[0:3, 3] = particle

        # 将变换应用到当前点云
        transformed_cloud = current_cloud.transform(transformation)

        # 计算匹配得分
        distances = np.asarray(transformed_cloud.compute_point_cloud_distance(self.global_map))
        fitness = np.sum(distances < 0.02) / len(distances)  # 假设阈值为0.02

        return fitness

    def localize(self):
        if self.current_cloud is None or not self.current_objects :
            print("current cloud or current object is empty")
            return
        
        distances = []

        for obj in self.current_objects:
            detected_center = np.array(obj[1][:3])
            origin = np.array([0, 0, 0])
            distance = np.linalg.norm(detected_center - origin)
            distances.append(distance)
        
        # 找到距离中的最小值
        if distances:
            min_distance = min(distances)

        all_particles = []
        for center in self.map_centers:
            particles = self.generate_particles(center, min_distance)
            all_particles.extend(particles)

        best_fitness = -1
        best_particle = None
        start_time = time.time()
        for i in range(100):  # 进行100次迭代
                print("i =%d",i)
                sample_particles = random.sample(all_particles, 3)
                
                avg_particle = np.mean(sample_particles, axis=0)
                fitness = self.compute_fitness(self.current_cloud, avg_particle)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_particle = avg_particle

        if best_particle is not None:
            result = Point()
            result.x = best_particle[0]
            result.y = best_particle[1] 
            result.z = best_particle[2]
            self.result_pub.publish(result)
            print(f"Localization result: {result.x}, {result.y}, {result.z}")
            print("object localization took %.3f sec.\n" % (time.time() - start_time))
            
            transformation = np.eye(4)
            transformation[0:3, 3] = best_particle
            self.current_cloud.transform(transformation)
            o3d.io.write_point_cloud('transformed_current_cloud.pcd', self.current_cloud)
            o3d.visualization.draw_geometries([self.global_map, self.current_cloud])
if __name__ == '__main__':
    try:
        pcl_localization = PointCloudLocalization()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
