#!/usr/bin/env python

import rospy
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
import struct
import tf2_ros
import tf_conversions
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import TransformStamped, PoseStamped
import tf.transformations as transformations
import tf2_ros
from tf import transformations as tf_trans
from scipy.spatial.transform import Rotation as R
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose
from nav_msgs.msg import Path
from sensor_msgs.msg import Image
import numpy as np

class PointCloudVisualizer:
    def __init__(self):
        rospy.init_node('point_cloud_visualizer', anonymous=True)
        self.pub_gt = rospy.Publisher('pc_gt', PointCloud2, queue_size=10)
        self.pub_infer = rospy.Publisher('pc_infer', PointCloud2, queue_size=10)
        self.pub_sparse = rospy.Publisher('sparse_pts', PointCloud2, queue_size=10)
        self.pub_refine = rospy.Publisher('pc_refine', PointCloud2, queue_size=10)
        self.marker_array_pub = rospy.Publisher('path_markers', MarkerArray, queue_size=10)
        self.path_pub = rospy.Publisher('path_viz', Path, queue_size=10)
        self.pub_normals = rospy.Publisher('pc_normals', PointCloud2, queue_size=10)
        self.pub_tgt_img = rospy.Publisher('tgt_img', Image, queue_size=10)
        self.pub_active = rospy.Publisher('active_pts', PointCloud2, queue_size=10)
        self.pub_slam = rospy.Publisher('slam_pts', PointCloud2, queue_size=10)
        # self.tf_buffer = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()


    def publish_path(self, poses):
        # Create a Path message
        path_msg = Path()
        
        # Set the header
        path_msg.header.frame_id = "global"
        path_msg.header.stamp = rospy.Time.now()

        # Create a PoseStamped message for each pose in the array
        for pose in poses:
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "global"
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.pose = pose

            # Append to the path message
            path_msg.poses.append(pose_stamped)
        
        # Publish the path message
        self.path_pub.publish(path_msg)
        #print("publish path")
        # # Create a marker array message
        # marker_array_msg = MarkerArray()
        
        # # Create a marker message for each pose in the array
        # for i, pose in enumerate(poses):
        #     marker = Marker()
        #     marker.header.frame_id = "global"
        #     marker.header.stamp = rospy.Time.now()
        #     marker.ns = "path_markers"
        #     marker.id = i
        #     marker.type = Marker.ARROW
        #     marker.action = Marker.ADD
        #     marker.pose = pose
        #     marker.scale.x = 0.2
        #     marker.scale.y = 0.05
        #     marker.scale.z = 0.05
        #     marker.color.a = 1.0
        #     marker.color.r = 1.0
        #     marker.color.g = 0.0
        #     marker.color.b = 0.0
            
        #     marker_array_msg.markers.append(marker)
        
        # # Publish the marker array
        # self.marker_array_pub.publish(marker_array_msg)
    
    def pose_callback(self, p_CinG, R_GtoC):
        # Create a transform message
        transform_msg = TransformStamped()
        transform_msg.header.stamp = rospy.Time.now()
        transform_msg.header.frame_id = "global"
        transform_msg.child_frame_id = "cam"

        rotation = R.from_matrix(R_GtoC)
        quaternion = rotation.as_quat()

        # Assuming msg is a PoseStamped message containing the camera's pose in the global frame
        transform_msg.transform.translation.x = p_CinG[0]
        transform_msg.transform.translation.y = p_CinG[1]
        transform_msg.transform.translation.z = p_CinG[2]
        transform_msg.transform.rotation.x = quaternion[0]
        transform_msg.transform.rotation.y = quaternion[1]
        transform_msg.transform.rotation.z = quaternion[2]
        transform_msg.transform.rotation.w = quaternion[3]

        # Publish the transform
        self.tf_broadcaster.sendTransform(transform_msg)
    
    def publish_sparse_points(self, sparse_pts):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "global"  # Change the frame ID if necessary

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('rgb', 12, PointField.UINT32, 1)]  # RGBA color information

        points_with_colors = []
        for i in range(len(sparse_pts)):
            rgb = (255 << 16)  # Red channel only
            point = list(sparse_pts[i]) + [rgb]
            points_with_colors.append(point)

        point_cloud_msg = pc2.create_cloud(header, fields, points_with_colors)
        self.pub_sparse.publish(point_cloud_msg)

    def publish_slam_points(self, sparse_pts):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "global"  # Change the frame ID if necessary

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('rgb', 12, PointField.UINT32, 1)]  # RGBA color information

        points_with_colors = []
        for i in range(len(sparse_pts)):
            rgb = (255 << 16)  # Red channel only
            point = list(sparse_pts[i]) + [rgb]
            points_with_colors.append(point)

        point_cloud_msg = pc2.create_cloud(header, fields, points_with_colors)
        self.pub_slam.publish(point_cloud_msg)

    def publish_active_points(self, sparse_pts):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "global"  # Change the frame ID if necessary

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('rgb', 12, PointField.UINT32, 1)]  # RGBA color information

        points_with_colors = []
        for i in range(len(sparse_pts)):
            rgb = (255)  # Red channel only
            point = list(sparse_pts[i]) + [rgb]
            points_with_colors.append(point)

        point_cloud_msg = pc2.create_cloud(header, fields, points_with_colors)
        self.pub_active.publish(point_cloud_msg)
    
    def publish_point_cloud_refine(self, points_refine, colors_refine):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "global"  # Change the frame ID if necessary 
        
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('rgb', 12, PointField.UINT32, 1)]
        points_with_colors = []
        for i in range(len(points_refine)):
            rgb = ((int(colors_refine[i][0]) & 0xFF) << 16) | ((int(colors_refine[i][1]) & 0xFF) << 8) | (int(colors_refine[i][2]) & 0xFF)
            point = list(points_refine[i]) + [rgb]
            points_with_colors.append(point)
        
        point_cloud_msg = pc2.create_cloud(header, fields, points_with_colors)
        self.pub_refine.publish(point_cloud_msg)
    
    def publish_normals(self, sparse_pts, normals):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "global"

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('normal_x', 12, PointField.FLOAT32, 1),
                    PointField('normal_y', 16, PointField.FLOAT32, 1),  
                    PointField('normal_z', 20, PointField.FLOAT32, 1),
                    PointField('curvature', 24, PointField.FLOAT32, 1) ]
        
        points_with_normals = []
        for i in range(len(sparse_pts)):
            point = list(sparse_pts[i]) + list(normals[i]) + [0.0]
            points_with_normals.append(point)
        
        print("Publishing normals: ", len(points_with_normals))
        point_cloud_msg = pc2.create_cloud(header, fields, points_with_normals)
        self.pub_normals.publish(point_cloud_msg)


    def publish_point_cloud_infer(self, points, colors):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "global"  # Change the frame ID if necessary

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb', 12, PointField.UINT32, 1)]  # RGBA color information

        points_with_colors = []
        for i in range(len(points)):
            rgb = ((int(colors[i][0]) & 0xFF) << 16) | ((int(colors[i][1]) & 0xFF) << 8) | (int(colors[i][2]) & 0xFF)
            point = list(points[i]) + [rgb]
            points_with_colors.append(point)

        point_cloud_msg = pc2.create_cloud(header, fields, points_with_colors)
        self.pub_infer.publish(point_cloud_msg)
        
    def publish_point_cloud_gt(self, points, colors):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "global"  # Change the frame ID if necessary

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1)]  # RGBA color information

        points_with_colors = []
        for i in range(min(len(points), len(colors))):
            rgb = ((int(colors[i][0]) & 0xFF) << 16) | ((int(colors[i][1]) & 0xFF) << 8) | (int(colors[i][2]) & 0xFF)
            point = list(points[i]) + [rgb]
            points_with_colors.append(point)

        print("Publishing GT: ", len(points_with_colors))
        point_cloud_msg = pc2.create_cloud(header, fields, points_with_colors)
        self.pub_gt.publish(point_cloud_msg)
    
    def publish_tgt_img(self, tgt_img):
        assert tgt_img.ndim == 3, "tgt_img should have 3 dimensions (H, W, C)"
        tgt_img_msg = Image()
        tgt_img = tgt_img.astype(np.uint8)
        tgt_img_msg.header.stamp = rospy.Time.now()
        tgt_img_msg.header.frame_id = "camera"
        tgt_img_msg.height = tgt_img.shape[0]
        tgt_img_msg.width = tgt_img.shape[1]
        tgt_img_msg.encoding = 'rgb8'
        tgt_img_msg.is_bigendian = 0
        tgt_img_msg.step = 3 * tgt_img.shape[1]
        tgt_img_msg.data = tgt_img.tobytes()
        self.pub_tgt_img.publish(tgt_img_msg)
