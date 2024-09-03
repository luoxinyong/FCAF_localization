# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from argparse import ArgumentParser

from mmengine.logging import print_log

from mmdet3d.apis import LidarDet3DInferencer


def parse_args():
    parser = ArgumentParser()   #定义一个参数变量
    parser.add_argument('pcd', help='Point cloud file')  #数据位置
    parser.add_argument('model', help='Config file')    #模型的配置位置
    parser.add_argument('weights', help='Checkpoint file')  #预训练模型位置
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference') #显卡数量
    parser.add_argument(
        '--pred-score-thr',
        type=float,
        default=0.3,
        help='bbox score threshold')    #检测框的阈值？
    parser.add_argument(
        '--out-dir',
        type=str,
        default='outputs',
        help='Output directory of prediction and visualization results.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show online visualization results')   #实时展示检测结果
    parser.add_argument(
        '--wait-time',
        type=float,
        default=-1,
        help='The interval of show (s). Demo will be blocked in showing'
        'results, if wait_time is -1. Defaults to -1.')
    parser.add_argument(
        '--no-save-vis',
        action='store_true',
        help='Do not save detection visualization results')
    parser.add_argument(
        '--no-save-pred',
        action='store_true',
        help='Do not save detection prediction results')
    parser.add_argument(
        '--print-result',
        action='store_true',
        help='Whether to print the results.')
    call_args = vars(parser.parse_args())

    call_args['inputs'] = dict(points=call_args.pop('pcd'))

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    init_kws = ['model', 'weights', 'device']   #拿出推理的预训练模型、配置文件和显卡配置,通过字典的方式保存
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw) #拿出推理的预训练模型、配置文件和显卡配置

    # NOTE: If your operating environment does not have a display device,
    # (e.g. a remote server), you can save the predictions and visualize
    # them in local devices. 离线保存推理的结果
    if os.environ.get('DISPLAY') is None and call_args['show']:
        print_log(
            'Display device not found. `--show` is forced to False',
            logger='current',
            level=logging.WARNING)
        call_args['show'] = False

    return init_args, call_args

# def print_all_keys(d, parent_key=''):
#     """递归打印字典的所有键"""
#     for key, value in d.items():
#         full_key = f"{parent_key}.{key}" if parent_key else key
#         print(full_key)
#         if isinstance(value, dict):
#             print_all_keys(value, full_key)
#         elif isinstance(value, list) and value and isinstance(value[0], dict):
#             for i, item in enumerate(value):
#                 print_all_keys(item, f"{full_key}[{i}]")

def main():
    # TODO: Support inference of point cloud numpy file.
    init_args, call_args = parse_args() #把推理所需要的变量全部搞完

    inferencer = LidarDet3DInferencer(**init_args)  #根据配置信息实例化一个推理对象
    inferencer(**call_args)
    # inputs = dict(points='/home/bnxy/mmdetection3d/data/s3dis/points/Area_1_conferenceRoom_1.bin')
    # results = inferencer(inputs)
    
    # print_all_keys(results)
    
    if call_args['out_dir'] != '' and not (call_args['no_save_vis']
                                           and call_args['no_save_pred']):
        print_log(
            f'results have been saved at {call_args["out_dir"]}',
            logger='current')


if __name__ == '__main__':
    main()
