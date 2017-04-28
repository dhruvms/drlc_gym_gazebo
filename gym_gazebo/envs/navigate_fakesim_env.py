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

from std_srvs.srv import Empty
from gazebo_msgs.srv import GetModelState, SetModelState
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist, Point, Pose
from nav_msgs.msg import Odometry
import message_filters
import threading

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
		self.vel_pub = rospy.Publisher('/dji_sim/target_velocity', Twist, queue_size=1)
		self.pose_subscriber = rospy.Subscriber('/dji_sim/odometry', Odometry, self.pose_callback)
		self.previous_min_laser_scan = 0.0
		self.done = False
		# the following are absolutes
		self.MAX_POSITION_X = 90.0
		self.MIN_POSITION_X = 0.0
		self.MAX_POSITION_Y = 30.0

		self.laser_subscriber = message_filters.Subscriber('/scan', LaserScan)
		self.image_subscriber = message_filters.Subscriber('/camera/rgb/image_raw', Image)
		self.synchro = message_filters.ApproximateTimeSynchronizer([self.laser_subscriber, self.image_subscriber], 10, 0.05)
		self.synchro.registerCallback(self.synchro_callback)

		self.observation = None
		self.laser = None
		self.HAVE_DATA = False
		self.first = True
		self.last_time_step_was_called = 0.0
		self.duration_since_step_was_called = 0.0
		# self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
		self.set_model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

		self.RED = '\033[91m'
		self.BLUE = '\033[94m'
		self.BOLD = '\033[1m'
		self.ENDC = '\033[0m'        
		self.LINE = "%s%s##############################################################################%s" % (self.BLUE, self.BOLD, self.ENDC)
		self.PURPLE = '\033[95m'
		self.YELLOW = '\033[93m'
		self.Red = '\033[91m'

		UNDERLINE = '\033[4m'
		thread = threading.Thread(target=self.get_time_since_step_was_called, args=())
		thread.daemon = True                            # Daemonize thread
		thread.start()                                  # Start the execution
		
		self.MAX_DURATION_BETWEEN_STEP_CALLS = 0.3
		self.MAX_NO_LASER_TIME = 0.1
	
	# check when was step() called last. this is a background thread. 
	# todo if dqn.update_policy() is in "control", it should (hopefully) still send a zero vel cmd
	# ref : http://sebastiandahlgren.se/2014/06/27/running-a-method-as-a-background-thread-in-python/
	def get_time_since_step_was_called(self):
		while True:
			if not self.done: # avoid extraneous checks when it's resetting dji and cylinder pose
				self.duration_since_step_was_called = time.time() - self.last_time_step_was_called
				# print self.YELLOW + "self.duration_since_step_was_called {:.2f} s".format(self.duration_since_step_was_called) + self.ENDC
				if self.duration_since_step_was_called > self.MAX_DURATION_BETWEEN_STEP_CALLS:
					print self.BLUE + "Ghost Mode. Step not called for {:.2f} s: sending zero vel. time : {:.2f} ".format(self.duration_since_step_was_called, time.time()) + self.ENDC
					vel_cmd_zero = Twist()
					self.vel_pub.publish(vel_cmd_zero)

			time.sleep(0.5) # this is hardcoded. if it's self.MAX_DURATION_BETWEEN_STEP_CALLS, it didn't work. (?!)

	# checks for bot's pose and sets self.is_terminal to True if it goes outside the forest'
	def pose_callback(self, msg):
		self.position =  msg.pose.pose.position

		# end episode if out of forest's box
		if (self.position.x < self.MIN_POSITION_X) or (self.position.x > self.MAX_POSITION_X) or (abs(self.position.y) > self.MAX_POSITION_Y):
			self.done = True
			# print "went out of range. ending episode"
			# rospy.loginfo("Point Position: [ %f, %f, %f ]"%(self.position.x, self.position.y, self.position.z))

			# experimental : set dji pose here itself. this can cause a bad callback error sometimes. tocheck
			# self.reset_dji()

	# where there is an image, let there be a laser message.
	# into that heaven of learning, let my bot awake  
	def synchro_callback(self, laser, image):
		cv_image = CvBridge().imgmsg_to_cv2(image, desired_encoding="passthrough")
		self.observation = np.asarray(cv_image)
		self.laser = laser
		# print self.laser.header #good for debugging ghost mode. check seq dropped by uncommenting
		self.HAVE_DATA = True

		self.min_laser_scan = np.min(self.laser.ranges)
		if self.min_laser_scan < self.MIN_LASER_DEFINING_CRASH:
			self.done = True

	def _step(self, action):
		self.last_time_step_was_called = time.time()
		vel_cmd = Twist()
		speed = 5.0
		delta_theta_deg = 10 # diff of heading (or pseudo-heading) between each action

		#### action set with varying heading ####
		# 4 is forward, 0-3 are to left, 5-8 are right. all separated by 10 deg each.
		action_norm = action - ((self.num_actions-1)/2)
		# 0 is forward in action_norm. negatives are left

		#### action set with same heading ####
		# vel_x_body = speed*math.cos(action_norm*(math.radians(delta_theta_deg)))
		# vel_y_body = speed*math.sin(action_norm*(math.radians(delta_theta_deg)))
		# vel_cmd.linear.x = vel_x_body
		# vel_cmd.linear.y = vel_y_body
		# vel_cmd.linear.z = 0

		vel_cmd.linear.x = speed
		vel_cmd.angular.z = action_norm*(math.radians(delta_theta_deg))
		self.vel_pub.publish(vel_cmd)

		# this time will roughly define no of steps per second
		# can't make it too high (or else it follows one action for too long
		# nor too low, or else, it does't have enough time to pick up spped and actually move
		# time.sleep(1e-1)
		
		# pubish zero after sleeping for a small time to avoid ghost mode bug
		# vel_cmd_zero = Twist()
		# self.vel_pub.publish(vel_cmd_zero)
		
		# keep on waiting for getting laser data
		self.HAVE_DATA = False
		start_time = time.time()
		while not self.HAVE_DATA:
			no_laser_time = time.time() - start_time
			# print no_laser_time #this is ~ 0.01 seconds
			if no_laser_time > self.MAX_NO_LASER_TIME:
				print self.PURPLE + "Ghost mode :: step () :: no laser data for {:.2f} s: sending zero vel. time : {:.2f}".format(no_laser_time, time.time()) + self.ENDC
				vel_cmd_zero = Twist()
				self.vel_pub.publish(vel_cmd_zero)
			# print "step() : self.HAVE_DATA is False!"
			continue

		# find distance to goal and give some reward based on it 
		dist_to_goal = math.sqrt((self.position.y - 0.0)**2 + (self.position.x - self.MAX_POSITION_X)**2)
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

		print "min_laser : {:.2f} dist_to_goal : {:.2f} reward_dist_to_goal : {:.2f} action : {:+d} reward : {:+.2f}"\
			.format(self.min_laser_scan, dist_to_goal, reward_dist_to_goal, action_norm, reward)

		# print "exiting step()"
		return self.observation, reward, self.done, {}	

	# utility function to set the bot's pose
	def reset_dji(self):
		# until the dji pose is actually set. This condition is added as there's some weird thing in the sim which causes the bot's' urdf
		# to fluctuate between it's last position and the reset position.  
		while not (self.reset_position.x == self.position.x) and \
			  not (self.reset_position.y == self.position.y) and \
			  not (self.reset_position.z == abs(self.position.z)):
		
			model_pose = Pose()
			model_pose.position.x = self.reset_position.x
			model_pose.position.y = self.reset_position.y
			model_pose.position.z = self.reset_position.z
			model_pose.orientation.x = 0.0
			model_pose.orientation.y = 0.0
			model_pose.orientation.z = 0.0
			model_pose.orientation.w = 1.0

			model_twist = Twist()

			model_state = ModelState()
			model_state.model_name = 'dji'
			model_state.pose = model_pose
			model_state.twist = model_twist
			model_state.reference_frame = 'world' # change to 'world'?
			rospy.wait_for_service('/gazebo/set_model_state')

			self.set_model_state_proxy(model_state)
			# rospy.loginfo("DJI position updated. Point Position: [ %f, %f, %f ]"%(self.position.x, self.position.y, self.position.z))
			# print(self.reset_position.x, self.reset_position.y, self.reset_position.z)
			# print(self.reset_position.x == self.position.x, self.reset_position.y == self.position.y, self.reset_position.z == self.position.z)

		rospy.loginfo("DJI position updated")
		# time.sleep(0.1)

		# recursive to ensure
		while not (self.reset_position.x == self.position.x) and \
			not (self.reset_position.y == self.position.y) and \
			not (self.reset_position.z == abs(self.position.z)):
			print "reset_dji() : recursion "
			self.reset_dji()
		
	# generate random poses for trees and call set model pose for each tree 
	def make_a_brave_new_forest(self):
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

				model_pose = Pose()
				model_pose.position.x = x_tree
				model_pose.position.y = y_tree
				model_pose.position.z = 5.0
				model_pose.orientation.x = 0.0
				model_pose.orientation.y = 0.0
				model_pose.orientation.z = 0.0
				model_pose.orientation.w = 1.0

				model_twist = Twist()

				model_state = ModelState()
				model_state.model_name = name
				model_state.pose = model_pose
				model_state.twist = model_twist
				model_state.reference_frame = 'world' # change to 'world'?
				rospy.wait_for_service('/gazebo/set_model_state')
				try:
					self.set_model_state_proxy(model_state)
				except rospy.ServiceException, e:
					print "Service call failed: %s"%e

		rospy.loginfo("Cylinder positions updated.")
		# time.sleep(0.1)
		# assert
		while not (self.reset_position.x == self.position.x) and \
			not (self.reset_position.y == self.position.y) and \
			not (self.reset_position.z == abs(self.position.z)):
			print "reset_forest => reset_dji()"
			self.reset_dji()

	def _reset(self):
		if not self.first:
			vel_cmd = Twist() # zero msg
			self.vel_pub.publish(vel_cmd)
			rospy.loginfo('reset called()')

			# reset drone
			self.reset_dji()

			# make a new forest
			self.make_a_brave_new_forest()

			self.HAVE_DATA = False

			while not self.HAVE_DATA:
				# print "_reset() :: self.HAVE_DATA is False!"
				continue
			# assert
			while not (self.reset_position.x == self.position.x) and \
				not (self.reset_position.y == self.position.y) and \
				not (self.reset_position.z == abs(self.position.z)):
				print "reset itself() => reset_dji()"
				self.reset_dji()
			
			self.done = False

		self.first = False

		return self.observation