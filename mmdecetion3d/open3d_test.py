import open3d as o3d
import copy  # 点云深拷贝
# import open3d as o3d
import numpy as np
import time

""" # -------------------------- 加载点云 ------------------------
print("->正在加载点云... ")
pcd = o3d.io.read_point_cloud("scans81.pcd")
print(pcd)
pcd.paint_uniform_color([1,0,0])
print("->pcd质心:",pcd.get_center())
# ===========================================================

# -------------------------- transform ------------------------
print("\n->点云的一般变换")
pcd_T = copy.deepcopy(pcd)
T = np.eye(4)
T[ :3, :3] = pcd.get_rotation_matrix_from_xyz((np.pi/6,np.pi/4,0))	# 旋转矩阵
T[0,3] = 5.0	# 平移向量的dx
T[1,3] = 3.0	# 平移向量的dy
print("\n->变换矩阵：\n",T)
pcd_T.transform(T)
pcd_T.paint_uniform_color([0,0,1])
print("\n->pcd_scale1质心:",pcd_T.get_center())
# ===========================================================

# -------------------------- 可视化 --------------------------
o3d.visualization.draw_geometries([pcd, pcd_T])
# ===========================================================
o3d.io.write_point_cloud('scans_rt.pcd',pcd_T) """

def preprocess_point_cloud(pcd, voxel_size):
    print(":: 使用大小为为{}的体素下采样点云.".format(voxel_size))
    pcd_down = pcd.voxel_down_sample(voxel_size)
 
    radius_normal = voxel_size * 2
    print(":: 使用搜索半径为{}估计法线".format(radius_normal))
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
 
    radius_feature = voxel_size * 5
    print(":: 使用搜索半径为{}计算FPFH特征".format(radius_feature))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

# def preprocess_point_cloud(pcd):
#     # 计算法线
#     radius_normal = 0.1  # 可根据点云密度调整
#     pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
#         radius=radius_normal, max_nn=30))

#     # 计算FPFH特征
#     radius_feature = 0.25  # 可根据点云密度调整
#     pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd,
#         o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
#     return pcd, pcd_fpfh


def execute_global_registration(source, target, source_fpfh, target_fpfh):
    distance_threshold = 0.1  # 可根据点云密度调整
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh,True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100, 0.999))
    return result

def refine_registration(source, target, result_ransac):
    distance_threshold = 0.02  # 可根据点云密度调整
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def execute_global_registration_ransac(source_down, target_down, source_fpfh,
                               target_fpfh, voxel_size):
    distance_threshold = 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d. pipelines.registration.RANSACConvergenceCriteria(100000,0.999))
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result

def main():
    voxel_size = 0.1
    # 加载点云
    source = o3d.io.read_point_cloud("scans.pcd")
    target = o3d.io.read_point_cloud("scans_test.pcd")

    # 点云预处理
    print("start preprocessing" )
    start_time = time.time()
    source_down, source_fpfh = preprocess_point_cloud(source,voxel_size)
    print("source featrue took %.3f sec.\n" % (time.time() - start_time))
    target_down, target_fpfh = preprocess_point_cloud(target,voxel_size)
    print("target featrue took %.3f sec.\n" % (time.time() - start_time))
    start_time = time.time()
    print(source_down)
    print(target_down)
    # 执行全局配准
    print("start RANSAC" )
    result_ransac = execute_global_registration_ransac(source_down, target_down, source_fpfh, target_fpfh,voxel_size)
    print("RANSAC result:")
    print(result_ransac)
    print("RANSAC took %.3f sec.\n" % (time.time() - start_time))

    # 显示配准结果
    draw_registration_result(source_down,target_down,result_ransac.transformation)

if __name__ == "__main__":
    main()
