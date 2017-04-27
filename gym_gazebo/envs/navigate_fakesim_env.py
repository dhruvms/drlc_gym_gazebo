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
from gazebo_msgs.msg import ModelState, ContactState, GetModelState, SetModelState
from geometry_msgs.msg import Twist, Point
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Odometry
import tf
import smtplib
from email.mime.text import MIMEText

import message_filters

class GazeboErleCopterNavigateEnvFakeSim(gym.Env): 
	def __init__(self):
		self.reset_x = 0.0
		self.reset_y = 0.0
		self.reset_z = 2.0
		self.reset_position = Point(self.reset_x, self.reset_y, self.reset_z)
		self.position = Point(self.reset_x, self.reset_y, self.reset_z) # initialize to same coz reset is called before pose callback

		# dem MDP rewards tho
		self.MIN_LASER_DEFINING_CRASH = 1.0
		self.MIN_LASER_DEFINING_NEGATIVE_REWARD = 2.0
		self.REWARD_AT_LASER_DEFINING_NEGATIVE_REWARD = 0.0
		self.REWARD_AT_LASER_JUST_BEFORE_CRASH = -5.0
		self.REWARD_AT_CRASH = -10
		self.REWARD_FOR_FLYING_SAFE = 0.25 # at each time step
		self.REWARD_FOR_FLYING_FRONT_WHEN_SAFE = 0.25

		subprocess.Popen("roscore")
		print ("Roscore launched!")

		rospy.init_node('gym', anonymous=True)
		subprocess.Popen(["roslaunch","dji_gazebo", "dji_rl.launch"])

		print "Initializing environment. Wait 5 seconds"
		rospy.sleep(5)
		print "############### DONE ###############"
		self.num_actions = 9
		self.action_space = spaces.Discrete(self.num_actions)
		self.reward_range = (-np.inf, np.inf)
		self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
		self.vel_pub = rospy.Publisher('/dji_sim/target_velocity', Twist, queue_size=10, latch=False)
		self.pose_subscriber = rospy.Subscriber('/dji_sim/odometry', Odometry, self.pose_callback)
		self.previous_min_laser_scan = 0.0
		self.done = False
		# the following are absolutes
		self.MAX_POSITION_X = 90.0
		self.MAX_POSITION_Y = 30.0

		self.laser_subscriber = message_filters.Subscriber('/scan', LaserScan)
		self.image_subscriber = message_filters.Subscriber('/camera/rgb/image_raw', Image)
		self.synchro = message_filters.ApproximateTimeSynchronizer(self.laser_subscriber, self.image_subscriber)
		self.synchro.registerCallback(self.synchro_callback)

		self.observation = None
		self.laser = None
		self.HAVE_DATA = False

		self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
		self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

	def pose_callback(self, msg):
		self.position =  msg.pose.pose.position
		# self.quat = msg.pose.pose.orientation
		# rospy.loginfo("Point Position: [ %f, %f, %f ]"%(self.position.x, self.position.y, self.position.z))
		# rospy.loginfo("Quat Orientation: [ %f, %f, %f, %f]"%(quat.x, quat.y, quat.z, quat.w))
		# self.euler = tf.transformations.euler_from_quaternion([self.quat.x, self.quat.y, self.quat.z, self.quat.w])
		# rospy.loginfo("Euler Angles: %s"%str(euler))
		# self.pose = msg.pose

		# end episode if out of forest's box
		if (abs(self.position.x) > self.MAX_POSITION_X) or (abs(self.position.y) > self.MAX_POSITION_Y):
			self.done = True
			print "went out of range. ending episode"
			rospy.loginfo("Point Position: [ %f, %f, %f ]"%(self.position.x, self.position.y, self.position.z))

	def laser_callback(self, msg):
		data = msg
		self.min_laser_scan = np.min(data.ranges)
		new_ranges = len(data.ranges)
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
		
		self.done = done

		self.previous_min_laser_scan = self.min_laser_scan
		# print "min_laser : {} previous {} ".format(round(self.min_laser_scan,2), round(self.previous_min_laser_scan,2))

	def _get_frame(self):
		frame = None;
		data = None;
		start = time.time()

		while (frame is None) or (data is None):
			diff = time.time() - start
			# print "diff", diff
			if diff > 0.5: # ghost mode send zero vel to stop
				print "ghost mode : diff", diff
				vel_cmd = Twist() # zero msg
				self.vel_pub.publish(vel_cmd)

			frame = rospy.wait_for_message('/camera/rgb/image_raw',Image, timeout = 5)
			data = rospy.wait_for_message('/scan', LaserScan, timeout = 5)
			cv_image = CvBridge().imgmsg_to_cv2(frame, desired_encoding="passthrough")
			frame = np.asarray(cv_image)
			# print frame.shape # (480, 640, 3)
			# cv2.imshow('frame', frame)
			# cv2.waitKey(1)

			self.min_laser_scan = np.min(data.ranges)
			if self.min_laser_scan < self.MIN_LASER_DEFINING_CRASH:
				self.done = True
			self.previous_min_laser_scan = self.min_laser_scan
		
	 	return frame
			# except:
				# raise ValueError('could not get frame')

	def synchro_callback(self, laser, image):
    	assert laser.header.stamp == image.header.stamp

    	cv_image = CvBridge().imgmsg_to_cv2(image, desired_encoding="passthrough")
		self.observation = np.asarray(cv_image)
		self.laser = laser

		self.HAVE_DATA = True

		self.min_laser_scan = np.min(self.laser.ranges)
		if self.min_laser_scan < self.MIN_LASER_DEFINING_CRASH:
			self.done = True

	def _step(self, action):
		# # hack 
		# data = None
		# while data is None:
		# 	data = rospy.wait_for_message('/scan', LaserScan, timeout = 5)
		# 	self.min_laser_scan = np.min(data.ranges)
		# 	if self.min_laser_scan < self.MIN_LASER_DEFINING_CRASH:
		# 		self.done = True
		if self.done:
			return self.observation, reward, self.done, {}	

		while not self.HAVE_DATA:
			continue

		# print "step was called"
		vel_cmd = Twist()
		speed = 2.5

		delta_theta_deg = 10
		# 4 is forward, 0-3 are to left, 5-8 are right. all separated by 10 deg each.
		action_norm = action - ((self.num_actions-1)/2)
		# 0 is forward in action_norm. negatives are left
		vel_x_body = speed*math.cos(action_norm*(math.radians(delta_theta_deg)))
		vel_y_body = speed*math.sin(action_norm*(math.radians(delta_theta_deg)))

		# vel_cmd.linear.x = vel_x_body
		# vel_cmd.linear.y = vel_y_body
		# vel_cmd.linear.z = 0

		vel_cmd.linear.x = speed
		vel_cmd.angular.z = action_norm*(math.radians(delta_theta_deg))
		# quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
		# print "current yaw", current_yaw
		# print "taking action_norm", action_norm, ":: velocity (x,y,z)", vel_cmd.twist.linear.x, vel_cmd.twist.linear.y, vel_cmd.twist.linear.z
		self.vel_pub.publish(vel_cmd)

		self.HAVE_DATA = False
		while not self.HAVE_DATA:
			continue
		
		# start = time.time()
		# observation = self._get_frame()
		# time_taken = time.time() - start
		# if time_taken > 0.5:
		# 	print "ghost in step()"
		# data = None
		# while data is None:
		# 	data = rospy.wait_for_message('/scan', LaserScan, timeout = 5)

		# min_laser_scan = np.min(data.ranges)
		# if min_laser_scan == self.previous_min_laser_scan:
			# print "fuck"

		# print "max laser", np.max(data.ranges)
		# state, is_terminal = self.discretize_observation(data,len(data.ranges))

		dist_to_goal = math.sqrt((self.position.y - 0.0)**2 + (self.position.x - self.MAX_POSITION_X/)**2)
		# reward_dist_to_goal = 1 / dist_to_goal
		reward_dist_to_goal = (self.MAX_POSITION_X-dist_to_goal) / float(self.MAX_POSITION_X)

		# if still alive
		if not self.done:
			# if obstacles are faraway
			if self.min_laser_scan > self.MIN_LASER_DEFINING_NEGATIVE_REWARD:
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
						(self.min_laser_scan - self.MIN_LASER_DEFINING_NEGATIVE_REWARD))
		else:
			reward = self.REWARD_AT_CRASH
 		# print self.min_laser_scan, action

		if action_norm < 0:
			print "min_laser : {} dist_to_goal : {} reward_dist_to_goal : {} action : {} reward : {}".format(round(self.min_laser_scan,2), round(dist_to_goal,2), \
						round(reward_dist_to_goal,2), action_norm, round(reward,2))
			# print "min_laser : {} action : {} reward : {}".format(round(self.min_laser_scan,2), action_norm, reward)
		else:
			print "min_laser : {} dist_to_goal : {} reward_dist_to_goal : {} action : +{} reward : {}".format(round(self.min_laser_scan,2), round(dist_to_goal,2), \
						round(reward_dist_to_goal,2), action_norm, round(reward,2))
			# print "min_laser : {} action : +{} reward : {}".format(round(self.min_laser_scan,2), action_norm, reward)

		return self.observation, reward, self.done, {}	

	def _reset(self):
		self.done = False
		self.HAVE_DATA = False
		while not self.HAVE_DATA:
			continue
		vel_cmd = Twist() # zero msg
		self.vel_pub.publish(vel_cmd)
		# time.sleep(1)
		rospy.loginfo('Gazebo RESET')
		fuck_ctr = 0
		# subprocess.Popen(["python","/home/vaibhav/madratman/drlc_gym_gazebo/forest_generator/make_forest.py"])
		
		# EPSILON = 1e-100
		# while (not abs(self.reset_position.x-self.position.x) < EPSILON) and \
			  # (not abs(self.reset_position.y-self.position.y) < EPSILON) and \
			  # (not abs(self.reset_position.z-self.position.z) < EPSILON):
		while (not self.reset_position.x == self.position.x) and \
			  (not self.reset_position.y == self.position.y) and \
			  (not self.reset_position.z == self.position.z):
		
			# generate random samples
			nx = 15
			spacing_x = 6
			random_interval_x = spacing_x/3
			offset_x = 5

			ny = 10
			spacing_y = 6
			random_interval_y = spacing_y
			offset_y = -int(ny*spacing_y/2)+3

			x = np.linspace(offset_x, offset_x+(nx-1)*spacing_x, nx)
			y = np.linspace(offset_y, offset_y+(ny-1)*spacing_y, ny)

			# positions_x=np.zeros([nx,ny])
			# positions_y=np.zeros([nx,ny])

			counter=0
			np.random.seed() #use seed from sys time to build new env on reset
			for i in range(nx):
				for j in range(ny):
					name='unit_cylinder_'+str(counter)


					counter+=1
					noise_x=np.random.random()-0.5
					noise_x*=random_interval_x
					noise_y=np.random.random()-0.5
					noise_y*=random_interval_y
					x_tree=x[i]+noise_x
					y_tree=y[j]+noise_y

					rospy.wait_for_service('/gazebo/get_model_state')
					try:
						model_data = self.get_model_state_proxy(name, '')

						model_data.pose.position.x = x_tree
						model_data.pose.position.y = y_tree
						model_data.pose.position.z = 3.0
						model_data.pose.orientation.x = 0.0
						model_data.pose.orientation.y = 0.0
						model_data.pose.orientation.z = 0.0
						model_data.pose.orientation.w = 0.0

						model_data.twist.linear.x = 0.0
						model_data.twist.linear.y = 0.0
						model_data.twist.linear.z = 0.0
						model_data.twist.angular.x = 0.0
						model_data.twist.angular.y = 0.0
						model_data.twist.angular.z = 0.0

						model_state = ModelState()
						model_state.model_name = name
						model_state.pose = model_data.pose
						model_state.twist = model_data.twist
						model_state.reference_frame = '' # change to 'world'?
						rospy.wait_for_service('/gazebo/set_model_state')
						try:
							self.set_model_state_proxy(model_state)
						except rospy.ServiceException, e:
							print "Service call failed: %s"%e

					except rospy.ServiceException, e:
						print "Service call failed: %s"%e

			try:
				model_data = self.get_model_state_proxy('dji', '')

				model_data.pose.position.x = self.reset_position.x
				model_data.pose.position.y = self.reset_position.y
				model_data.pose.position.z = self.reset_position.z
				model_data.pose.orientation.x = 0.0
				model_data.pose.orientation.y = 0.0
				model_data.pose.orientation.z = 0.0
				model_data.pose.orientation.w = 0.0

				model_data.twist.linear.x = 0.0
				model_data.twist.linear.y = 0.0
				model_data.twist.linear.z = 0.0
				model_data.twist.angular.x = 0.0
				model_data.twist.angular.y = 0.0
				model_data.twist.angular.z = 0.0

				model_state = ModelState()
				model_state.model_name = 'dji'
				model_state.pose = model_data.pose
				model_state.twist = model_data.twist
				model_state.reference_frame = '' # change to 'world'?
				rospy.wait_for_service('/gazebo/set_model_state')
				try:
					self.set_model_state_proxy(model_state)
				except rospy.ServiceException, e:
					print "Service call failed: %s"%e

			except rospy.ServiceException, e:
				print "Service call failed: %s"%e

		return self.observation

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
		return discretized_ranges, done
