#读取PCD文件，并按照一定高度进行过滤，返回过滤后的点
#可以根据需要计算读取文件的耗时
import open3d as o3d
import numpy as np
import time

z_threshold = 1.2

def read_pcd_and_extract_xyz(pcd_file):
    # 读取PCD文件
    start_time = time.time()
    pcd = o3d.io.read_point_cloud(pcd_file)

    # 获取点云的大小
    points = np.asarray(pcd.points)
    filtered_points = points[points[:, 2] <= z_threshold]
    # 初始化一个数组来存储XYZ坐标
    create_time = time.time() - start_time
    print(f"Time taken to read PCD file: {create_time:.6f} seconds")
    return filtered_points

if __name__ == '__main__':
    pcd_file = 'scans.pcd'
    points = read_pcd_and_extract_xyz(pcd_file)
    print("First five points (x, y, z):")
    for i in range(min(50, len(points))):
        print(f"Point {i+1}: ({points[i, 0]}, {points[i, 1]}, {points[i, 2]})")