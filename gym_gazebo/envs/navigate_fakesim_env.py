import gym
import numpy as np
import os
import rospy
import roslaunch
import subprocess
import time
import math

import cv2
from cv_bridge import CvBridge, CvBridgeError

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from gym.utils import seeding

from mavros_msgs.msg import OverrideRCIn, ParamValue
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode, ParamSet, ParamGet
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan, NavSatFix, Image
from std_msgs.msg import Float64
from gazebo_msgs.msg import ModelStates, ContactState
from geometry_msgs.msg import Twist, Point
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import tf
import smtplib
from email.mime.text import MIMEText

class GazeboErleCopterNavigateEnvFakeSim(): 
	def __init__(self):
		self.reset_x = 0.0
		self.reset_y = 0.0
		self.reset_z = 2.0
		self.reset_position = Point(self.reset_x, self.reset_y, self.reset_z)
		self.SPEEDUPFACTOR = 10.0
		# dem MDP rewards tho
		self.MIN_LASER_DEFINING_CRASH = 2.0
		self.MIN_LASER_DEFINING_NEGATIVE_REWARD = 4.0
		self.REWARD_AT_LASER_DEFINING_NEGATIVE_REWARD = 0.0
		self.REWARD_AT_LASER_JUST_BEFORE_CRASH = -5.0
		self.REWARD_AT_CRASH = -10
		self.REWARD_FOR_FLYING_SAFE = 1.0 # at each time step
		self.REWARD_FOR_FLYING_FRONT_WHEN_SAFE = 1.0

		subprocess.Popen("roscore")
		print ("Roscore launched!")

		rospy.init_node('gym', anonymous=True)
		subprocess.Popen(["roslaunch","dji_gazebo", "dji_rl.launch"])

		print "Initializing environment. Wait 5 seconds"
		rospy.sleep(5)
		print "############### DONE ###############"
		self.num_actions = 9
		self.action_space = spaces.Discrete(self.num_actions) # F, L, R, B
		self.reward_range = (-np.inf, np.inf)
		self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
		self.vel_pub = rospy.Publisher('/dji_sim/target_velocity', Twist, queue_size=10, latch=False)
		self.pose_subscriber = rospy.Subscriber('/dji_sim/odometry', Odometry, self.pose_callback)

	def pose_callback(self, msg):
		self.position =  msg.pose.pose.position
		self.quat = msg.pose.pose.orientation
		# rospy.loginfo("Point Position: [ %f, %f, %f ]"%(position.x, position.y, position.z))
		# rospy.loginfo("Quat Orientation: [ %f, %f, %f, %f]"%(quat.x, quat.y, quat.z, quat.w))
		self.euler = tf.transformations.euler_from_quaternion([self.quat.x, self.quat.y, self.quat.z, self.quat.w])
		# rospy.loginfo("Euler Angles: %s"%str(euler))
		self.pose = msg.pose

	def step(self, action):
		# print "step was called"
		vel_cmd = Twist()
		speed = 10

		delta_theta_deg = 10
		# 4 is forward, 0-3 are to left, 5-8 are right. all separated by 10 deg each.
		action_norm = action - ((self.num_actions-1)/2)
		# 0 is forward in action_norm. negatives are left
		vel_x_body = speed*math.cos(action_norm*(math.radians(delta_theta_deg)))
		vel_y_body = speed*math.sin(action_norm*(math.radians(delta_theta_deg)))

		vel_cmd.linear.x = vel_x_body
		vel_cmd.linear.y = vel_y_body
		vel_cmd.linear.z = 0
		# quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
		# print "current yaw", current_yaw
		# print "taking action_norm", action_norm, ":: velocity (x,y,z)", vel_cmd.twist.linear.x, vel_cmd.twist.linear.y, vel_cmd.twist.linear.z
		self.vel_pub.publish(vel_cmd)
	
		observation = self._get_frame()
		
		data = None
		while data is None:
			try:
				data = rospy.wait_for_message('/scan', LaserScan, timeout = 5)
			except:
				pass

		# is_terminal = self.check_terminal(data)
		min_laser_scan = np.min(data.ranges)
		# print "max laser", np.max(data.ranges)
		state, is_terminal = self.discretize_observation(data,len(data.ranges))

		# dist_to_goal = math.sqrt((self.position_y - 220.0)**2 + (self.position_x - 0.0)**2)
		# reward_dist_to_goal = 1 / dist_to_goal

		# if still alive
		if not is_terminal:
			# if obstacles are faraway
			if min_laser_scan > self.MIN_LASER_DEFINING_NEGATIVE_REWARD:
				# if flying forward
				if action_norm == 0:
					reward = self.REWARD_FOR_FLYING_FRONT_WHEN_SAFE
				else:
					reward = self.REWARD_FOR_FLYING_SAFE
			# if obstacles are near, -20 for MIN_LASER_DEFINING_CRASH, 0 for MIN_LASER_DEFINING_NEGATIVE_REWARD 
			else:
				# y = y1 + (y2-y1)/(x2-x1) * (x-x1)
				reward = self.REWARD_AT_LASER_DEFINING_NEGATIVE_REWARD + \
						((self.REWARD_AT_LASER_JUST_BEFORE_CRASH - self.REWARD_AT_LASER_DEFINING_NEGATIVE_REWARD)/ \
						(self.MIN_LASER_DEFINING_CRASH - self.MIN_LASER_DEFINING_NEGATIVE_REWARD)* \
						(min_laser_scan - self.MIN_LASER_DEFINING_NEGATIVE_REWARD))
		else:
			reward = self.REWARD_AT_CRASH
		# if action_norm < 0:
		# 	# print "min_laser : {} dist_to_goal : {} reward_dist_to_goal : {} action : {} reward : {}".format(round(min_laser_scan,2), round(dist_to_goal,2), \
		# 				# round(reward_dist_to_goal,2), action_norm, reward)
		# 	print "min_laser : {} action : {} reward : {}".format(round(min_laser_scan,2), action_norm, reward)

		# else:
		# 	print "min_laser : {} action : +{} reward : {}".format(round(min_laser_scan,2), action_norm, reward)

		return observation, reward, is_terminal, {}	

	def _get_frame(self):
		frame = None;
		while frame is None:
			try:
				frame = rospy.wait_for_message('/camera/rgb/image_raw',Image, timeout = 5)
				cv_image = CvBridge().imgmsg_to_cv2(frame, desired_encoding="passthrough")
				frame = np.asarray(cv_image)
				# print frame.shape # (480, 640, 3)
				# cv2.imshow('frame', frame)
				# cv2.waitKey(1)
				return frame
			except:
				raise ValueError('could not get frame')

	def reset(self):
		vel_cmd = Twist() # zero msg
		self.vel_pub.publish(vel_cmd)
		# time.sleep(1)
		rospy.loginfo('Gazebo RESET')
		subprocess.Popen(["python","/home/vaibhav/madratman/drlc_gym_gazebo/forest_generator/make_forest.py"])
		while (not self.reset_position.x == self.position.x) and (not self.reset_position.y == self.position.y) and (not self.reset_position.z == self.position.z):
			self.reset_proxy()
			# rospy.sleep(1)
		return self._get_frame()

	def discretize_observation(self,data,new_ranges):
		# print data
		discretized_ranges = []
		done = False
		mod = len(data.ranges)/new_ranges
		for i, item in enumerate(data.ranges):
			if (i%mod==0):
				if data.ranges[i] == float ('Inf'):
					discretized_ranges.append(6)
				elif np.isnan(data.ranges[i]):
					discretized_ranges.append(0)
				else:
					discretized_ranges.append(int(data.ranges[i]))
			if (self.MIN_LASER_DEFINING_CRASH > data.ranges[i] > 0):
				done = True
		return discretized_ranges,done
