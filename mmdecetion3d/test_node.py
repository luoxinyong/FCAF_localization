#本python脚本适用于宇树雷达的数据，已经做了10帧点云的累计，从fast-lio拿到了cloud_registed话题的点云
#还需要添加一个当前位置的订阅者
#还需要验证10帧累计的点云数量够不够。。
# TODOx2
#!/usr/bin/env python

import rospy
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from lxy_fcaf3d import FCAFDemo
from visualization_msgs.msg import Marker,MarkerArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation as Rota
import ros_numpy

# dtype_list = np.dtype([
#     ('x', np.float32),
#     ('y', np.float32),
#     ('z', np.float32),
#     ('intensity', np.float32),
#     ('normal_x', np.float32),
#     ('normal_y', np.float32),
#     ('normal_z', np.float32)
# ])

cls_names = ['table', 'chair', 'sofa', 'bookshelf', 'bed']
point_clouds=[]
colors = {
    0:(255, 255, 255),
    1: (0, 255, 0),  # 类别1的颜色（绿色）
    2: (0, 0, 255),  # 类别2的颜色（红色）
    3: (255, 0, 0),
    4: (255, 255, 0),
    5: (255, 0, 255),
    # 添加更多类别的颜色
}

line_indexs = [[0, 1], [2, 3], [4, 5], [6, 7],
         [0, 2], [2, 4], [4, 6], [6, 0],
         [1, 3], [3, 5], [5, 7], [7, 1]]

def box_to_corners(box):
    # generate clockwise corners and rotate it clockwise
    # 顺时针方向返回角点位置
    cx, cy, cz, dx, dy, dz, angle,cls, conf = box

    a_cos = np.cos(-angle) #注意这里角度yao jia fu hao
    a_sin = np.sin(-angle)
    corners_x = [-dx / 2, -dx / 2, dx / 2, dx / 2]
    corners_y = [-dy / 2, dy / 2, dy / 2, -dy / 2]

    WZ_bottom = cz - dz/2
    WZ_top =    cz + dz/2

    corners = []
    for i in range(4):
        X = a_cos * corners_x[i] + \
                     a_sin * corners_y[i] + cx
        Y = -a_sin * corners_x[i] + \
                     a_cos * corners_y[i] + cy
        corners.append([X, Y, WZ_bottom])
        corners.append([X, Y, WZ_top])

    return corners #返回8个顶点坐标

def pointcloud_callback(data):
    global point_clouds
    point_clouds.append(data)
    if len(point_clouds) == 10:
         process_point_clouds(point_clouds)
         point_clouds = []

def path_callback(msg):
    ros_numpy

def process_point_clouds(clouds):
    combined_points = []
    dtype_list = np.dtype([
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32)
            # ('intensity', np.float32),
            # ('normal_x', np.float32),
            # ('normal_y', np.float32), 
            # ('normal_z', np.float32),
            # ('curvature', np.float32)
            ])
    for pc_msg in clouds:
        # print("done")
        # pc = ros_numpy.numpify(pc_msg)
        pc = np.frombuffer(pc_msg.data, dtype=dtype_list)
        #速腾格式的点云不行，是个1800*16的数组，我真的服了
        # print("PC Fields:", pc.dtype.names)
        # print("PC Shape[0]:", pc.shape[0])
        points = np.zeros((pc.shape[0], 3))
        points[:, 0] = pc['x']
        points[:, 1] = pc['y']
        points[:, 2] = pc['z']
        combined_points.append(points)

    combined_points = np.concatenate(combined_points, axis=0)
    rgb = np.zeros((combined_points.shape[0], 3))
    rgb[:, 0] = 255
    rgb[:, 1] = 255
    rgb[:, 2] = 255

    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    data_np = np.concatenate((np.array(pcd.points),rgb),axis=1)
    results,output_bboxes = fcaf_demo.infer(pcd,data_np) #output_bboxes里面是每一个框的xyz（中心点）+dxdydz（长宽高）+angle+classname（类别）+score（得分）
    
    print("done")
    marker_array = MarkerArray()
    for i, (x, y, z, dx, dy, dz, angle, cls, score ) in enumerate(output_bboxes):
        # 创建Marker
        #1、制作物体检测的长方体
        marker = Marker()
        marker.header.frame_id = "camera_init"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "objects"
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = dx
        marker.scale.y = dy
        marker.scale.z = dz

            # 设置Marker颜色
        color = colors.get(cls)  # 默认白色
        marker.color.r = color[0] / 255.0
        marker.color.g = color[1] / 255.0
        marker.color.b = color[2] / 255.0
        marker.color.a = 0.3  # 透明度
            
            # marker.lifetime = rospy.Duration(1.0)
        # 2、制作一个3D的包围框
            # 添加到MarkerArray
        marker_array.markers.append(marker)
            
        marker_boxes = Marker()
        marker_boxes.header.frame_id = "camera_init"
        marker_boxes.header.stamp = rospy.Time.now()
        marker_boxes.ns = "objects_boxes"
        marker_boxes.id = i
        marker_boxes.type = Marker.LINE_LIST
        marker_boxes.action = Marker.ADD

        marker_boxes.pose.orientation.x = 0.0
        marker_boxes.pose.orientation.y = 0.0
        marker_boxes.pose.orientation.z = 0.0
        marker_boxes.pose.orientation.w = 1.0
        
        marker_boxes.color.r = 138 / 255
        marker_boxes.color.g = 226 / 255
        marker_boxes.color.b = 52 / 255
        marker_boxes.color.a = 1  # 透明度

        marker_boxes.scale.x = 0.03
        marker_boxes.points = []


        corners = box_to_corners((x,y,z,dx,dy,dz,angle, cls, score))

        for line_index in line_indexs:
            sta = corners[line_index[0]]
            end = corners[line_index[1]]
            marker_boxes.points.append(Point(sta[0],sta[1],sta[2]))
            marker_boxes.points.append(Point(end[0],end[1],end[2]))

            # 设置Marker颜色
        marker_array.markers.append(marker_boxes)
            # 3、做这个检测的文本标签
        marker_word = Marker()
        marker_word.header.frame_id = "camera_init"
        marker_word.header.stamp = rospy.Time.now()
        marker_word.ns = "objects_word"
        marker_word.id = i
        marker_word.type = Marker.TEXT_VIEW_FACING
        marker_word.action = Marker.ADD

        marker_word.pose.position.x = x
        marker_word.pose.position.y = y
        marker_word.pose.position.z = z + dz / 2 + 0.1
        r = Rota.from_euler('xyz', (0, 0, angle), degrees=False)

        quaternion = r.as_quat()
        marker_word.pose.orientation.x = quaternion[0]
        marker_word.pose.orientation.y = quaternion[1]
        marker_word.pose.orientation.z = quaternion[2]
        marker_word.pose.orientation.w = quaternion[3]

        marker_word.scale.z = 0.2

            # 设置Marker颜色
        marker_word.color.r = 138 / 255
        marker_word.color.g = 226 / 255
        marker_word.color.b = 52 / 255
        marker_word.color.a = 1  # 透明度

        marker_word.text = "%s%d_%.2f"%(cls_names[int(cls)],i+1,float(score))
        marker_array.markers.append(marker_word)
        # 发布MarkerArray
    marker_pub.publish(marker_array)
    

if __name__ == '__main__':
    rospy.init_node('pointcloud_inference_node', anonymous=True)

    config = 'configs/fcaf3d/fcaf3d_2xb8_s3dis-3d-5class.py'
    checkpoint = 'work_dirs/fcaf3d_2xb8_s3dis-3d-5class/epoch_12.pth'
    
    fcaf_demo = FCAFDemo(config, checkpoint)
    marker_pub = rospy.Publisher("box_marker", MarkerArray, queue_size=10)
    pcd = o3d.geometry.PointCloud()
    # rospy.Subscriber('/rslidar_points', PointCloud2, pointcloud_callback)
    # rospy.Subscriber('/unilidar/cloud', PointCloud2, pointcloud_callback)
    rospy.Subscriber('/livox/lidar', PointCloud2, pointcloud_callback)
    # rospy.Subscriber('/path', Path, path_callback)
    # rospy.Subscriber('/velodyne_points', PointCloud2, pointcloud_callback)
    rospy.spin()
