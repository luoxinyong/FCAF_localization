# import open3d as o3d
# import numpy as np

# def load_and_transform_point_cloud(file_path, transformation):
#     pcd = o3d.io.read_point_cloud(file_path)
#     pcd.transform(transformation)
#     return pcd

# def main():
#     # 加载地图点云
#     map_pcd = o3d.io.read_point_cloud("object_map.pcd")
#     object_pcd1 = o3d.io.read_point_cloud("object_0.pcd")
#     object_pcd2 = o3d.io.read_point_cloud("object_1.pcd")
#     object_pcd3 = o3d.io.read_point_cloud("object_2.pcd")
#     object_pcd4 = o3d.io.read_point_cloud("object_3.pcd")
#     object_pcd5 = o3d.io.read_point_cloud("object_4.pcd")

#     # 定义凳子点云的文件路径
    
#     # 定义每个凳子点云的初始变换矩阵 (例如平移和旋转)
#     transformations = [
#         np.eye(4),  # 第一个凳子保持原位
#         np.array([[1, 0, 0, 1],
#                   [0, 1, 0, 0],
#                   [0, 0, 1, 0],
#                   [0, 0, 0, 1]]),  # 第二个凳子平移1单位
#         np.array([[1, 0, 0, 2],
#                   [0, 1, 0, 0],
#                   [0, 0, 1, 0],
#                   [0, 0, 0, 1]]),  # 第三个凳子平移2单位
#         np.array([[1, 0, 0, 3],
#                   [0, 1, 0, 0],
#                   [0, 0, 1, 0],
#                   [0, 0, 0, 1]]),  # 第四个凳子平移3单位
#         np.array([[1, 0, 0, 4],
#                   [0, 1, 0, 0],
#                   [0, 0, 1, 0],
#                   [0, 0, 0, 1]])   # 第五个凳子平移4单位
#     ]

#     # 加载并变换每个凳子点云
#     map_pcd += object_pcd1.transform(transformations[0])
#     map_pcd += object_pcd2.transform(transformations[1])
#     map_pcd += object_pcd3.transform(transformations[2])
#     map_pcd += object_pcd4.transform(transformations[3])
#     map_pcd += object_pcd5.transform(transformations[4])

#     # 保存合并后的点云
#     o3d.io.write_point_cloud("combined_map_with_stools.pcd", map_pcd)

#     # 显示合并后的点云
#     o3d.visualization.draw_geometries([map_pcd])

# if __name__ == "__main__":
#     main()
import open3d as o3d
import numpy as np

def translate_point_cloud(pcd, translation):
    translated_pcd = pcd.translate(translation, relative=False)
    return translated_pcd

positions = [
    (0.167680,7.647321),
    (11.64, 15.773),
    (-9.95, 5.95),
    (-2.647982, 11.29),
    (0.093, 18.19),
    (2.568, 28.379),
    (-19.81, 40.785),
    (-27.278, 35.078),
    (-49.098, 46.636),
    (-39.191544, 61.072)
]

# Load existing map point cloud
map_pcd = o3d.io.read_point_cloud("object_map.pcd")

# Load 10 chair point clouds
chair_pcds = [o3d.io.read_point_cloud(f"object_{i}.pcd") for i in range(10)]

# Create point clouds for chairs and add to the map point cloud
all_chairs_pcd = o3d.geometry.PointCloud()
for i, pos in enumerate(positions):
    chair_pcd = chair_pcds[i]  # Use the i-th chair PCD
    translated_chair_pcd = translate_point_cloud(chair_pcd, np.array([pos[0], pos[1], -0.3]))
    all_chairs_pcd += translated_chair_pcd
    print(f"Position {pos} -> Chair object_{i}")

# Combine map and chairs
combined_pcd = map_pcd + all_chairs_pcd

# Save combined point cloud
o3d.io.write_point_cloud("combined_map.pcd", combined_pcd)

print("Combined point cloud saved to 'combined_map.pcd'.")
