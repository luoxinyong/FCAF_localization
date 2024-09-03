import rospy
from std_srvs.srv import Trigger

from lxy_fcaf3d import FCAFDemo
import numpy as np
from visualization_msgs.msg import Marker,MarkerArray
import open3d as o3d
from geometry_msgs.msg import Point
from scipy.spatial.transform import Rotation as Rota
import json
import time
import atexit

# save_object_point_cloud = True

colors = {
    0:(255, 255, 255),
    1: (0, 255, 0),  # 类别1的颜色（绿色）
    2: (0, 0, 255),  # 类别2的颜色（红色）
    3: (255, 0, 0),
    4: (255, 255, 0),
    5: (255, 0, 255),
    # 添加更多类别的颜色
}

cls_names = ['table', 'chair', 'sofa', 'bookshelf', 'bed']

class PosePointCloudData:
    def __init__(self, index, pose, points):
        self.index = index
        self.pose = pose
        self.points = points

detected_objects = {}
marker_id = 0
marker_array = MarkerArray()
distance_threshold = 0.5
current_id = 0 
full_cloud = o3d.geometry.PointCloud()

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

def infer_point_cloud(data):
    # combined_points = []
    rgb = np.ones((data.shape[0], 3), dtype=np.uint8) * 255
    pcd.points = o3d.utility.Vector3dVector(data)
    pcd.colors = o3d.utility.Vector3dVector(rgb / 255.0)
    data_np = np.concatenate((np.array(pcd.points),rgb),axis=1)
    results,output_bboxes = fcaf_demo.infer(pcd,data_np) #output_bboxes里面是每一个框的xyz（中心点）+dxdydz（长宽高）+angle+classname（类别）+score（得分）
    global marker_id,detected_objects,current_id,full_cloud
    new_detected_objects = {}

    # print("done")
    #这里发布扫描框
    for i, (x, y, z, dx, dy, dz, angle, cls, score ) in enumerate(output_bboxes):
        new_center = np.array([x, y, z])
        new_size = np.array([dx, dy, dz])
        is_new_object = True
        assigned_id = None
        
        for obj_key, obj_info in detected_objects.items():
            old_center = np.array(obj_key[:3])
            if np.linalg.norm(new_center - old_center) < distance_threshold:
                is_new_object = False
                assigned_id = obj_info["id"]
                # 更新中心点为旧中心和新中心的累积均值
                obj_info["center"] = (obj_info["center"] * obj_info["count"] + new_center) / (obj_info["count"] + 1)
                obj_info["size"] = (obj_info["size"] * obj_info["count"] + new_size) / (obj_info["count"] + 1)
                obj_info["count"] += 1
                break

        if is_new_object:
            assigned_id = current_id
            current_id += 1
            obj_key = (x, y, z)
            new_detected_objects[obj_key] = {
            "id": assigned_id,
            "center": new_center,
            "size": new_size,
            "count": 1,
            "cube": None,
            "boxes": None,
            "word": None,
            "class":cls
            }

            obj_info = new_detected_objects[obj_key]

        
        if not is_new_object:
            continue

        if obj_info["cube"] is None:
            marker = Marker()
            marker.header.frame_id = "camera_init"
            marker.ns = "objects"
            marker.id = obj_info["id"] 
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.color.a = 0.3
            obj_info["cube"] = marker
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
            marker.color.a = 0.3
            marker_array.markers.append(marker)
        else:
            marker = obj_info["cube"]
        #1、制作物体检测的长方体
          # 透明度
            
            # marker.lifetime = rospy.Duration(1.0)
        # 2、制作一个3D的包围框
            # 添加到MarkerArray
        if obj_info["boxes"] is None:
            marker_boxes = Marker()
            marker_boxes.header.frame_id = "camera_init"
            marker_boxes.ns = "objects_boxes"
            marker_boxes.id = obj_info["id"]
            marker_boxes.type = Marker.LINE_LIST
            marker_boxes.action = Marker.ADD
            marker_boxes.color.r = 138 / 255
            marker_boxes.color.g = 226 / 255
            marker_boxes.color.b = 52 / 255
            marker_boxes.color.a = 1
            marker_boxes.scale.x = 0.03
            obj_info["boxes"] = marker_boxes
            marker_boxes.points = []
            corners = box_to_corners((x,y,z,dx,dy,dz,angle, cls, score))
            for line_index in line_indexs:
                sta = corners[line_index[0]]
                end = corners[line_index[1]]
                marker_boxes.points.append(Point(sta[0],sta[1],sta[2]))
                marker_boxes.points.append(Point(end[0],end[1],end[2]))
            marker_array.markers.append(marker_boxes)
        else:
            marker_boxes = obj_info["boxes"]
            # 设置Marker颜色
        # new_marker_array.markers.append(marker_boxes)
            # 3、做这个检测的文本标签
        if obj_info["word"] is None:
            marker_word = Marker()
            marker_word.header.frame_id = "camera_init"
            marker_word.ns = "objects_word"
            marker_word.id = obj_info["id"] 
            marker_word.type = Marker.TEXT_VIEW_FACING
            marker_word.action = Marker.ADD
            marker_word.color.r = 138 / 255
            marker_word.color.g = 226 / 255
            marker_word.color.b = 52 / 255
            marker_word.color.a = 1
            marker_word.scale.z = 0.2
            obj_info["word"] = marker_word
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

            marker_word.text = "%s%d_%.2f"%(cls_names[int(cls)],obj_info["id"]+1,float(score))
            # new_marker_array.markers.append(marker_word)
            marker_array.markers.append(marker_word)
        else:
            marker_word = obj_info["word"]
        
        # 发布MarkerArray
        full_cloud += pcd
    detected_objects.update(new_detected_objects) 
    # marker_array.markers.extend(new_marker_array.markers) 
    marker_pub.publish(marker_array)
    

def save_detected_objects_point_clouds():
    with open("detected_objects_info.txt", "w") as f:
        for obj_key, obj_info in detected_objects.items():
            center = obj_info['center']
            dx, dy, dz = obj_info['size']  
            cls = obj_info['class']
            # 创建一个包围盒来提取物体的点云
            bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound=center - [dx / 2, dy / 2, dz / 2],
                                                            max_bound=center + [dx / 2, dy / 2, dz / 2])
            object_cloud = full_cloud.crop(bounding_box)#这里保存的是物体包围盒内部的点云
            #2027/7/30👍：尝试把周围一定距离的点云保存进来，提取特征。
            #不需要实际实现。。
            # 保存物体的点云到PCD文件
            file_name = f"object_{obj_info['id']}.pcd"
            o3d.io.write_point_cloud(file_name, object_cloud)
            print(f"Saved {file_name}")
            
            object_info = f"{cls_names[int(cls)]}_{obj_info['id']}: {center[0]},{center[1]},{center[2]},{dx},{dy},{dz}\n"
            f.write(object_info)
            print(f"Saved {cls}_{obj_info['id']} info to txt file")

def client():
    rospy.init_node('pose_pcd_client')
    rospy.wait_for_service('get_data')
    try:
        get_data = rospy.ServiceProxy('get_data', Trigger)
        while not rospy.is_shutdown():
            response = get_data()
            if response.success:   
                start_time = time.time()
                data_dict = json.loads(response.message)
                create_time = time.time() - start_time
                print(f"Time taken to load data:{create_time:.6f} seconds")
                points = np.array(data_dict['points'])
                start_time2 = time.time()
                infer_point_cloud(points)
                create_time2 = time.time() - start_time2
                print(f"Time taken to infer object: {create_time2:.6f} seconds")
            # else:
            #     rospy.logwarn("No data available.")

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if __name__ == "__main__":
    config = 'configs/fcaf3d/fcaf3d_2xb8_s3dis-3d-5class.py'
    checkpoint = 'work_dirs/fcaf3d_2xb8_s3dis-3d-5class/epoch_12.pth'
    fcaf_demo = FCAFDemo(config, checkpoint)
    marker_pub = rospy.Publisher("box_marker", MarkerArray, queue_size=100)
    pcd = o3d.geometry.PointCloud()
    
    client()
    atexit.register(save_detected_objects_point_clouds) 

