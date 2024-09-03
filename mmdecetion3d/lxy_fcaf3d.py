#用3d点云测试出来的框本身就不带angle值，全是0
import logging
import os
from argparse import ArgumentParser
from mmengine.logging import print_log
from mmdet3d.apis import LidarDet3DInferencer
# import pandas as pd
import open3d as o3d
# import torch
import time
import numpy as np
# import read_bin as rs
z_threshold = 1.2
global pred_score_thr  #定义得分阈值
pred_score_thr = 0.4

cls_names = ['桌子', '椅子', '沙发', '书柜', '床']

class FCAFDemo:
    def __init__(self, config, checkpoint, device='cuda:0'):
        self.inferencer = LidarDet3DInferencer(model=config, weights=checkpoint, device=device)

    def infer(self, pcd,points_array):

        start_time3 = time.time()
        inputs =  {'points': points_array}
        results = self.inferencer(inputs)
        create_time3 = time.time() - start_time3
        print(f"Time taken to infer: {create_time3:.6f} seconds")
        output_bboxes = results['predictions'][0]['bboxes_3d']
        labels_3d = results['predictions'][0]['labels_3d']
        scores_3d = results['predictions'][0]['scores_3d']

        # first_bbox = output_bboxes[0]

        filtered_indices = [i for i, score in enumerate(scores_3d) if score >= pred_score_thr]
        output_bboxes = [output_bboxes[i] for i in filtered_indices]
        labels_3d = [labels_3d[i] for i in filtered_indices]
        scores_3d = [scores_3d[i] for i in filtered_indices]

        pred_pnums = []
        for x, y, z, dx, dy, dz, angle in output_bboxes:
            # print("\n角度:%.2f",angle)
            z += dz / 2  
            R_box = pcd.get_rotation_matrix_from_axis_angle(np.array([0, 0, angle]).T)
            boundingbox = o3d.geometry.OrientedBoundingBox(np.array([x, y, z]),
                                                       R_box,
                                                       np.array([dx, dy, dz])
                                                       )
            indices = boundingbox.get_point_indices_within_bounding_box(pcd.points)
            pred_pnums.append(len(indices))
    
        pred_bboxes_transform = output_bboxes
        out_res = []
        print("\n检测结果数目:",len(pred_bboxes_transform))
        for i,(x,y,z,dx,dy,dz,angle) in enumerate(pred_bboxes_transform): #这里输出的center_z表示的是底部中心的高度
            z += dz/2 #移动到真的中心

            score = scores_3d[i]
            label_id = labels_3d[i]
            # pnums = pred_pnums[i]
            cls_name = cls_names[label_id]

            # out_res.append( [x,y,z,dx,dy,dz,angle,label_id,score,pnums] )
            out_res.append( [x,y,z,dx,dy,dz,angle,label_id,score] )
            print("检测结果:%d  类别:%s 位置xyz:(%.2f %.2f %.2f) 尺寸dxyz:(%.2f %.2f %.2f)  得分：%.4f" % (i+1,cls_name,x,y,z,dx,dy,dz,score))
        return results,np.array(out_res)

def load_point_bin_file(file_path):
    data = np.fromfile(file_path, dtype=np.float32)
    num_points = (data.size - 3) // 3 # 每个点有 x, y, z 四个值，最后三个是位置
    points = data[:num_points * 3].reshape(-1, 3)
    position = data[num_points * 3:]
    return points, position

def read_pcd_and_extract_xyz(pcd_file):
    # 读取PCD文件
    start_time = time.time()
    pcd = o3d.io.read_point_cloud(pcd_file)
    
    # 获取点云的大小
    points = np.asarray(pcd.points)
    filtered_points = points[points[:, 2] <= z_threshold]
    print(f"Cloud size = : {len(filtered_points)}")
    # 初始化一个数组来存储XYZ坐标
    create_time = time.time() - start_time
    print(f"Time taken to read PCD file: {create_time:.6f} seconds")
    return filtered_points

def main():
    # 配置文件路径和模型检查点路径
    config = 'configs/fcaf3d/fcaf3d_2xb8_s3dis-3d-5class.py'
    checkpoint = 'work_dirs/fcaf3d_2xb8_s3dis-3d-5class/epoch_12.pth'

    # 初始化推理类
    fcaf_demo = FCAFDemo(config, checkpoint)
    pcd = o3d.geometry.PointCloud()
    # 假设你有一个点云数据 points_array
    # points = np.fromfile('scans.bin',dtype=np.float32).reshape()
    points_array= read_pcd_and_extract_xyz('0820_3_chairs_pointcloud_translat.pcd')
    start_time2 = time.time()
    rgb = np.ones((points_array.shape[0], 3), dtype=np.uint8) * 255
    create_time2 = time.time() - start_time2
    print(f"Time taken to rgb: {create_time2:.6f} seconds")

    pcd.points = o3d.utility.Vector3dVector(points_array)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    data_np = np.concatenate((np.array(pcd.points),rgb),axis=1)
    results,output_res = fcaf_demo.infer(pcd,data_np)


if __name__ == '__main__':
    main()
