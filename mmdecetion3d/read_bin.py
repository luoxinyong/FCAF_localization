# import numpy as np

# def load_point_bin_file(file_path):
#     # Load data from the bin file
#     points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 6)  # Assuming 6 features (XYZ + RGB)
#     return points

# # Example usage
# file_path = 'cloud_pos.bin'
# points = load_point_bin_file(file_path)


# # Define a dictionary to describe the data
# point_cloud_description = {
#     'x': points[:, 0],
#     'y': points[:, 1],
#     'z': points[:, 2],
#     'r': points[:, 3],
#     'g': points[:, 4],
#     'b': points[:, 5]
# }

# print("Points shape:", points.shape)  # Should be (N, 6)
# print("First few points (x, y, z):")
# for i in range(5):
#     print(f"Point {i}: ({point_cloud_description['x'][i]}, "
#           f"{point_cloud_description['y'][i]}, "
#           f"{point_cloud_description['z'][i]})")
    
# #进行测试的部分代码，实际要用的应该还是前面的

# print("First few RGB values:")
# for i in range(5):
#     print(f"Point {i}: (R: {point_cloud_description['r'][i]}, "
#           f"G: {point_cloud_description['g'][i]}, "
#           f"B: {point_cloud_description['b'][i]})")
import numpy as np

def load_point_bin_file(file_path):
    data = np.fromfile(file_path, dtype=np.float32)
    num_points = (data.size - 3) // 3 # 每个点有 x, y, z 四个值，最后三个是位置
    points = data[:num_points * 3].reshape(-1, 3)
    position = data[num_points * 3:]
    return points, position

# Example usage
file_path = 'scans51.bin'
points, position = load_point_bin_file(file_path)

print("Points shape:", points.shape)  # Should be (N, 4)
print("First few points (x, y, z):")
for i in range(5):
    print(f"Point {i}: ({points[i, 0]}, {points[i, 1]}, {points[i, 2]})")

print("Position data:", position)  # Should be (3,)
