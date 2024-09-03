#本脚本改编于rs_lidar，不需要做累计
#实现所需：1、从pcd读取文件；2、订阅pose话题并读取文件保存
#TODO：1、对velodyne格式的cloud_registed订阅，还要记录Path或者位置
#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
import ros_numpy
from read_pcd import read_pcd_and_extract_xyz
from std_srvs.srv import Trigger, TriggerResponse
import json
class PosePointCloudData:
    def __init__(self, index, pose, points):
        self.index = index
        self.pose = pose
        self.points = points

pcd_path = "/home/bnxy/point_ws/src/point-lio/PCD/for_detect.pcd"
current_index = 0

def pose_callback(pose_msg):
    global current_index, shared_list
    print(f"got pose_msg, go read pcd")
    points_array = read_pcd_and_extract_xyz(pcd_path)#这里的数据已经去掉了z轴一部分的值
    current_index +=1
    data = PosePointCloudData(index=current_index, pose=pose_msg, points=points_array)
    shared_list.append(data)
    rospy.loginfo(f"Saved data with index {current_index}")

def server_callback(req):
    global current_index,shared_list
    
    if not shared_list:
        return TriggerResponse(success=False, message="No data available.")
        
    data = shared_list.pop(0)
    data_dict = {
        'index': data.index,
        'pose': {
            'position': {'x': data.pose.pose.position.x, 'y': data.pose.pose.position.y, 'z': data.pose.pose.position.z},
            'orientation': {'x': data.pose.pose.orientation.x, 'y': data.pose.pose.orientation.y, 'z': data.pose.pose.orientation.z, 'w': data.pose.pose.orientation.w}
        },
        'points': data.points.tolist()
    }
    response_message = json.dumps(data_dict)
    return TriggerResponse(success=True, message=response_message)


if __name__ == '__main__':
    global shared_list
    rospy.init_node('pose_pcd_saver')
    shared_list = []
    # 启动ROS节点并订阅/path话题
    print(f"listening path message")
    rospy.Subscriber('/pose_for_detect', PoseStamped, pose_callback)

    # 启动服务
    service = rospy.Service('get_data', Trigger,  server_callback)
    rospy.spin()
