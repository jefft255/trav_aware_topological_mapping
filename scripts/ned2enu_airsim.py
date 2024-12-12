#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import *
from airsim_ros_pkgs.msg import CarControls


def odom_callback(msg: Odometry, pub):
    rot_quat = quaternion_about_axis(np.pi / 2.0, (0, 0, 1))
    ori = quaternion_about_axis(0, (0, 0, 1))
    ori[0] = msg.pose.pose.orientation.x
    ori[1] = msg.pose.pose.orientation.y
    ori[2] = msg.pose.pose.orientation.z
    ori[3] = msg.pose.pose.orientation.w
    rot_pose = quaternion_multiply(ori, rot_quat)
    msg.pose.pose.orientation.x = rot_pose[0]
    msg.pose.pose.orientation.y = rot_pose[1]
    msg.pose.pose.orientation.z = rot_pose[2]
    msg.pose.pose.orientation.w = rot_pose[3]
    # msg.header.frame_id = "odom"

    # x = msg.twist.twist.angular.x
    # y = msg.twist.twist.angular.y
    # msg.twist.twist.angular.x = x
    # msg.twist.twist.angular.y = -y
    # # msg.twist.twist.angular.z = -msg.twist.twist.angular.z

    # x = msg.twist.twist.linear.x
    # y = msg.twist.twist.linear.y
    # msg.twist.twist.linear.x = x
    # msg.twist.twist.linear.y = -y
    # # msg.twist.twist.linear.z = -msg.twist.twsit.linear.z

    pub.publish(msg)


last_msg_out = None
last_msg = None


def joystick_callback(msg: Twist):
    global last_msg_out, last_msg

    msg_out = CarControls()
    if msg.linear.x < 0:
        msg_out.manual = True
        msg_out.manual_gear = -1
    else:
        msg_out.manual = True
        msg_out.manual_gear = 1

    msg_out.gear_immediate = True
    msg_out.header.stamp = rospy.Time.now()
    msg_out.throttle = msg.linear.x
    msg_out.steering = -msg.angular.z
    last_msg_out = msg_out
    last_msg = msg


# rospy.loginfo("in main")
rospy.init_node("ned2enu", anonymous=True)
pub = rospy.Publisher("/odometry/filtered", Odometry, queue_size=10)
pub_cmd = rospy.Publisher("/airsim_node/Husky/car_cmd", CarControls, queue_size=10)
pub_vel = rospy.Publisher("/husky_velocity_controller/cmd_vel", Twist, queue_size=10)
rospy.Subscriber(
    "/airsim_node/Husky/odom_local_enu", Odometry, odom_callback, callback_args=pub
)
rospy.Subscriber("/joy_teleop/cmd_vel", Twist, joystick_callback)
rate = rospy.Rate(100)
while not rospy.is_shutdown():
    if last_msg_out is not None and last_msg is not None:
        pub_cmd.publish(last_msg_out)
        pub_vel.publish(last_msg)
    rate.sleep()
