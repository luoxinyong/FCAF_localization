## 使用mmdetection3d——FCAF3D实现从ROS的点云物体检测

FCAF3D是2022年论文《[FCAF3D: Fully Convolutional Anchor-Free 3D Object Detection](https://arxiv.org/abs/2112.00322)》所提出的对于点云的3D物体检测算法，做这个项目的初衷是因为要实现我的项目中对点云语义信息的运用的要求，所以决定找一个能够实现物体检测的代码完成项目要求（物体检测也算是一个语义信息的使用了）。在B站上也找到了一个基于ORB_slam和FCAF3D的物体级语义SLAM：[物体级语义SLAM（ORBSLAM2+FCAF3D）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1W84115782/?spm_id_from=333.999.top_right_bar_window_custom_collection.content.click&vd_source=2a4a421fff86197316764ff403f3fe6e)，也算找到了一个类似的，而且他的代码是开源的，所以看着效果不错就打算试试看。

FCAF3D是基于MMdetection3D这个目标检测开源工具箱，Github链接是：https://github.com/open-mmlab/mmdetection3d.git。这个工具箱隐藏了所有深度学习的有的没的的操作，对我这个5月份才开始学深度学习的比较友好，但美中不足的是，MMdetection这整个框架的说明文档就是一坨，对于纯新手而言，网络上所有能找到的教程都已经跟不上当前的版本，**过时**了，尤其是官方文档，很多关键的API函数什么都不写，最重要的推理的结果返回值也是一个字典。这些结果都是我通过print+ChatGpt得来的，就现在而言，有效性只保证到我所安装的版本，下面就我的demo进行一个比较详细（我认为的）的介绍。

### 1、mmdetection3d的安装

这里所有的安装都是跟着官方文档的安装流程来的，在安装之前先简单介绍一下我的环境配置：

Windows-WSL2：Linux环境是Ubuntu20.04，ROS版本为noetic，虚拟环境用的Miniconda，这里为了保证ROS2和Miniconda环境不冲突参考了另外的几个博客：[Ubuntu16.04安装anaconda并且设置不与ros产生冲突 - 简书 (jianshu.com)](https://www.jianshu.com/p/a5418864a416)、[【ROS】ros-noetic和anaconda联合使用【教程】_ros中安装anaconda-CSDN博客](https://blog.csdn.net/qq_44940689/article/details/133813086)。这里我的建议是Ubuntu版本一定要20以上，因为ROS只有在noetic以后所用的python是python3的版本，不会和虚拟环境产生过多的干扰

虚拟环境配置：

Python version：3.8.19

GPU：天选3自带的RTX3060，CUDA版本最高支持12.4，下图是在Ubuntu的终端，WSL2自动解决了Windows和Ubuntu的GPU问题，有关这个不用问我了![image-20240617172332017](C:\Users\15041\AppData\Roaming\Typora\typora-user-images\image-20240617172332017.png)

CUDA版本：虚拟环境安装的是CUDA12.1，这里的安装可以上网看其它的CUDA安装教程，大体就是去官方网站[CUDA Toolkit 12.5 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)找到自己对应的版本，通过执行对应的代码进行安装![image-20240617172726826](C:\Users\15041\AppData\Roaming\Typora\typora-user-images\image-20240617172726826.png)

对应的历史版本去这里找：![image-20240617172939782](C:\Users\15041\AppData\Roaming\Typora\typora-user-images\image-20240617172939782.png)

但唯一要补充的是，我在安装WSL版本的CUDA时，不管是deb（local）还是deb（network）都会引导我的cuda去安装最新的12.5的版本，为此搞得我还重新删掉cuda-12.5重新装，所以稳妥起见，我的建议是对于老的版本最好是选择本地安装的方式**runfile（local）**，唯一的不足是下载会有点慢。

### 2、解决ROS（noetic）python环境和miniconda的python冲突问题

这里给出几个博客上给出的解决方案：1）[Ubuntu16.04安装anaconda并且设置不与ros产生冲突 - 简书 (jianshu.com)](https://www.jianshu.com/p/a5418864a416)  2）[【ROS】ros-noetic和anaconda联合使用【教程】_ros中安装anaconda-CSDN博客](https://blog.csdn.net/qq_44940689/article/details/133813086#comments_33344958)

总的来说，在安装anaconda或者miniconda的时候，**要选择不把conda的bin添加到.bashrc里面**，虽然后面要联合使用的时候需要在**对应的终端手动source miniconda/bin/activate**来启动虚拟环境，但这样才能保证两个python环境不冲突。

后面的步骤主要可以参考链接2，<1>在虚拟环境中安装ros依赖，**pip install rospkg rospy catkin_tools** <2>检查自己环境中使用ros是否成功。按照步骤创建功能包，启动ros脚本。

**我所遇到的错误**

在第五步创建工作空间和ROS功能包的时候，ros_init_node（）报错：

Traceback (most recent call last):
File "/home/bnxy/test_ws/src/test_ros_python/scripts/test_node.py", line 16, in <module>
rospy.init_node('test_lxy22', anonymous=True)
File "/opt/ros/noetic/lib/python3/dist-packages/rospy/client.py", line 323, in init_node
raise rospy.exceptions.ROSInitException("init_node interrupted before it could complete")
rospy.exceptions.ROSInitException: init_node interrupted before it could complete

之后去查看ros输出的.log文件发现错误是ros python环境中缺少netifaces这个module。

### 3、训练自己的facf3d模型

fcaf3d的官方Github上给出的他们的模型和配置文件在我的电脑上也搞不了，这也是为什么我要自己训练的原因了。

这里因为我的机器配置用的是Windows11下WLS2的Linux，Ubuntu20.04系统，所以我没找到可以在我的Ubuntu系统中安装Matlab的方法，这里就没有使用SUN-RGBD数据集了。整体的训练过程可以参考[【三维目标检测】FCAF3D（一） - 代码天地 (codetd.com)](https://www.codetd.com/article/14618124)这里的训练

**下载S3DIS数据集**

```Sh
cd data/s3dis
python collect_indoor3d_data.py
cd ../..
python tools/create_data.py s3dis --root-path ./data/s3dis --out-dir ./data/s3dis --extra-tag s3dis
```

这里的命令都要在mmdecetion3d的文件夹下，下载的S3DIS数据集默认放到data/s3dis中解压。

**训练**

mmdecetion3d的训练分为单GPU和多GPU两种训练方式，两种方式启动的训练脚本也不一样，具体可见mmdecetion3d的官方文档：[教程4：使用现有模型进行训练和测试 — MMSegmentation 1.2.2 文档](https://mmsegmentation.readthedocs.io/zh-cn/main/user_guides/4_train_test.html)

我的电脑只有一块3060的显卡，所以就使用单GPU训练

```shell
python tools/train.py configs/fcaf3d/fcaf3d_2xb8_s3dis-3d-5class.py 
```

但在实验室的服务器上有两块90显卡，命令也就不一样了

3060的显卡训练12轮大概花了3天左右的时间（垃圾笔记本还是不要拿来训练了。。。），最后得到了epoch_12.pth的模型和一个fcaf3d_2xb8_s3dis-3d-5class.py的配置文件，在输出的work_dir文件夹里面。
	由于我刚入深度学习不到一个月，所以这些个模型微调啊，训练参数设置什么的完全不懂，全都是按照默认的来了，有关这个的没什么能说的，就跳过了。

### 4、ROS联合fcaf3d一起实现点云的目标检测

这里主要参考的还是那个B站up主的开源代码:[物体级语义SLAM（ORBSLAM2+FCAF3D）_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1W84115782/?spm_id_from=333.999.top_right_bar_window_custom_collection.content.click&vd_source=2a4a421fff86197316764ff403f3fe6e)感兴趣的可以去看看他的代码，我的代码很大程度上和他的类似，但我的是基于纯激光雷达点云的。

我这里使用的雷达是速腾的Helios-16p多线雷达，但录制的数据集为了跑Fast_lio2所以用了rs_to_velodyne的功能包转换了。具体的检测流程就是通过rospy的脚本文件订阅“/velodyne”的激光点云话题，从里面得到激光点云每个点的坐标值xyz，打包成一个数组后作为参数传递给我的fcaf3d_demo的infer进行检测，然后从检测结果当中拿到检测框的坐标值，再通过ROS的Marker发布出来，在rviz里面就能看到检测出来的实时框。
