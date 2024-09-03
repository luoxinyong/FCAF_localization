import logging
import os
from argparse import ArgumentParser
from plyfile import PlyData
import pandas as pd
import open3d as o3d
from mmengine.logging import print_log
import torch
from mmdet3d.apis import LidarDet3DInferencer
import numpy as np
# 关闭numpy中的科学计数输出
np.set_printoptions(precision=4, suppress=True)
# 关闭pytorch中的科学计数输出
torch.set_printoptions(precision=4, sci_mode=False)

def read_ply(input_path):
    plydata = PlyData.read(input_path)  # read file
    data = plydata.elements[0].data  # read data
    data_pd = pd.DataFrame(data)  # convert to DataFrame
    data_np = np.zeros(data_pd.shape, dtype=np.float32)  # initialize array to store data
    property_names = data[0].dtype.names  # read names of properties
    for i, name in enumerate(
            property_names):  # read data by property
        data_np[:, i] = data_pd[name]

    return data_np
def parse_args():

    # 直接在代码中指定参数值
    call_args = {
        'pcd': 'data/s3dis/points/Area_1_office_23.bin',
        'model': 'configs/fcaf3d/fcaf3d_2xb8_s3dis-3d-5class.py',
        'weights': 'work_dirs/fcaf3d_2xb8_s3dis-3d-5class/epoch_12.pth',
        'device': 'cuda:0',
        'pred_score_thr': 0.3,
        'out_dir': 'outputs',
        'show': False,
        'wait_time': -1,
        'no_save_vis': False,
        'no_save_pred': False,
        'print_result': False
    }

    call_args['inputs'] = dict(points=call_args.pop('pcd'))

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    init_kws = ['model', 'weights', 'device']
    init_args = {init_kw: call_args.pop(init_kw) for init_kw in init_kws}

    # 检查 DISPLAY 环境变量以防止显示错误
    if os.environ.get('DISPLAY') is None and call_args['show']:
        print_log(
            'Display device not found. `--show` is forced to False',
            logger='current',
            level=logging.WARNING)
        call_args['show'] = False

    return init_args, call_args

def main():
    # TODO: Support inference of point cloud numpy file.
    init_args, call_args = parse_args() #把推理所需要的变量全部搞完

    inferencer = LidarDet3DInferencer(**init_args) 
    input_file_name ='/home/bnxy/mmdetection3d/data/s3dis/s3dis_infos_Area_1.pkl'
    if not os.path.exists(input_file_name):
        return

    data_np = read_ply(input_file_name)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data_np[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(data_np[:,3:]/255.0)

    results = inferencer(pcd)

    for prediction in results['predictions']:
        labels_3d = prediction['labels_3d']
        # 处理 labels_3d，提取你需要的信息
        # 例如，打印每个物体的类别信息
        for label in labels_3d:
            print(f'Object detected: Class {label}')

    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(
            f'results have been saved at {call_args["out_dir"]}',
            logger='current')

if __name__ == '__main__':
    main()
