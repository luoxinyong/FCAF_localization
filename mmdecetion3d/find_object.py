import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

def find_positions_of_new_points(coords_known, distances_new_points, reference_point):
    """
    在已知坐标中找到与新给定点的距离矩阵最匹配的点集合。
    
    参数：
    coords_known (numpy.ndarray): 已知物体的坐标，形状为(n, 3)。
    distances_new_points (numpy.ndarray): 新给定点的距离矩阵，形状为(m, m)。
    reference_point (numpy.ndarray): 新给定点的参考点，形状为(1, 3)。
    
    返回：
    tuple: 最佳匹配点的索引和新给定点在已知坐标中的位置。
    """
    def distance_matrix(coords):
        n = len(coords)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
                dist_matrix[j, i] = dist_matrix[i, j]
        return dist_matrix

    def error_function(subset_indices):
        subset_coords = coords_known[list(subset_indices)]
        D_subset = distance_matrix(subset_coords)
        return np.sum((D_subset - distances_new_points)**2)

    n = len(coords_known)
    m = len(distances_new_points)
    min_error = float('inf')
    best_subset = None

    for subset in combinations(range(n), m):
        error = error_function(subset)
        if error < min_error:
            min_error = error
            best_subset = subset

    best_coords = coords_known[list(best_subset)]
    
    # 将新给定点从参考点转换到已知物体的坐标系中
    translation_vector = best_coords[0] - reference_point
    new_points_positions = reference_point + translation_vector

    return list(best_subset), new_points_positions

def visualize_matching_process(coords_known, new_points_positions, best_subset, reference_point):
    """
    可视化已知物体的坐标、新给定点的坐标及匹配过程。
    
    参数：
    coords_known (numpy.ndarray): 已知物体的坐标，形状为(n, 3)。
    new_points_positions (numpy.ndarray): 新给定点的坐标，形状为(m, 3)。
    best_subset (list): 最佳匹配点的索引。
    reference_point (numpy.ndarray): 新给定点的参考点，形状为(1, 3)。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制已知物体的坐标
    ax.scatter(coords_known[:, 0], coords_known[:, 1], coords_known[:, 2], c='b', marker='o', label='已知物体')
    
    # 绘制参考点的坐标
    ax.scatter(reference_point[0], reference_point[1], reference_point[2], c='g', marker='x', label='参考点')
    ax.text(reference_point[0], reference_point[1], reference_point[2], 'Ref', color='green')
    
    # 绘制新给定点的坐标
    ax.scatter(new_points_positions[:, 0], new_points_positions[:, 1], new_points_positions[:, 2], c='r', marker='^', label='新给定点')
    
    # 标注已知物体的坐标
    for i in range(len(coords_known)):
        ax.text(coords_known[i, 0], coords_known[i, 1], coords_known[i, 2], f'A{i+1}', color='blue')
    
    # 标注新给定点的坐标
    for i in range(len(new_points_positions)):
        ax.text(new_points_positions[i, 0], new_points_positions[i, 1], new_points_positions[i, 2], f'P{i+1}', color='red')
    
    # 绘制匹配过程
    for i, idx in enumerate(best_subset):
        ax.plot([coords_known[idx, 0], new_points_positions[i, 0]],
                [coords_known[idx, 1], new_points_positions[i, 1]],
                [coords_known[idx, 2], new_points_positions[i, 2]], 'g--')
    
    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    ax.set_zlabel('Z 坐标')
    ax.legend()
    plt.show()

# 示例使用
coords_known = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

distances_new_points = np.array([
    [0, 5, 8],
    [5, 0, 3],
    [8, 3, 0]
])

reference_point = np.array([1, 2, 3])

best_subset, new_points_positions = find_positions_of_new_points(coords_known, distances_new_points, reference_point)
print("新给定点的坐标是：", new_points_positions)

# 可视化
visualize_matching_process(coords_known, new_points_positions, best_subset, reference_point)
