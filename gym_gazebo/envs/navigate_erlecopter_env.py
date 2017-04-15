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
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import PoseStamped
import tf

class GazeboErleCopterNavigateEnv(gazebo_env.GazeboEnv):
	def _takeoff(self, altitude):
		print "Waiting for mavros..."
		data = None
		while data is None:
			try:
				data = rospy.wait_for_message('/mavros/global_position/rel_alt', Float64, timeout=5)
			except:
				pass
		
		takeoff_successful = False
		start = time.time() 

		while not takeoff_successful:
			diff = time.time() - start
			if diff > 15.0:
				rospy.loginfo('Changing mode to STABILIZE')
				# Set STABILIZE mode
				rospy.wait_for_service('/mavros/set_mode')
				try:
					self.mode_proxy(0,'STABILIZE')
					start = time.time()
				except rospy.ServiceException, e:
					print ("/mavros/set_mode service call failed: %s"%e)
				
				rospy.loginfo('DISARMing throttle')
				rospy.wait_for_service('/mavros/cmd/arming')
				try:
					self.arm_proxy(False)
				except rospy.ServiceException, e:
					print ("/mavros/set_mode service call failed: %s"%e)


			print "Taking off..."
			alt = altitude
			err = alt * 0.1 # 10% error

			rospy.loginfo('Changing mode to GUIDED')
			# Set GUIDED mode
			rospy.wait_for_service('/mavros/set_mode')
			try:
				self.mode_proxy(0,'GUIDED')
			except rospy.ServiceException, e:
				print ("/mavros/set_mode service call failed: %s"%e)

			time.sleep(0.1)

			rospy.loginfo('ARMing throttle')
			# Arm throttle
			rospy.wait_for_service('/mavros/cmd/arming')
			try:
				self.arm_proxy(True)
			except rospy.ServiceException, e:
				print ("/mavros/set_mode service call failed: %s"%e)

			time.sleep(0.1)
			
			rospy.loginfo('TAKEOFF to %d meters', alt)
			# Takeoff
			rospy.wait_for_service('/mavros/cmd/takeoff')
			try:
				self.takeoff_proxy(0, 0, 0, 0, alt) # 1m altitude
			except rospy.ServiceException, e:
				print ("/mavros/cmd/takeoff service call failed: %s"%e)

			# time.sleep(alt)

			alt_msg = None
			while alt_msg is None:
				try:
					alt_msg = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=10)
				except:
					pass

			erlecopter_index = 0
			print "Finding erle-copter index"
			for name in alt_msg.name:
				if name == "erlecopter":
					break
				else:
					erlecopter_index +=1
			try:
				erlecopter_alt = alt_msg.pose[erlecopter_index].position.z * 2
			except:
				erlecopter_alt = -1

			if erlecopter_alt > (alt - err):
				takeoff_successful = True
				print "Takeoff successful"
			else:
				print "Takeoff failed, retrying..."

		rospy.wait_for_service('/mavros/param/get')
		gcs = self.param_get_proxy('SYSID_MYGCS').value.integer
		if gcs != 1:
			# Set Mavros as GCS
			rospy.wait_for_service('/mavros/param/set')
			try:
				info = ParamSet()
				info.param_id = 'SYSID_MYGCS'

				val = ParamValue()
				val.integer = 1
				val.real = 0.0
				info.value = val

				self.param_set_proxy(info.param_id, info.value)

				rospy.loginfo('Changed SYSID_MYGCS from %d to %d', gcs, val.integer)
			except rospy.ServiceException, e:
				print ("/mavros/set_mode service call failed: %s"%e)

		time.sleep(0.1)

		self.msg = OverrideRCIn()
		self.msg.channels[0] = 0 # Roll
		self.msg.channels[1] = 0 # Pitch
		self.msg.channels[2] = 1500 # Throttle
		self.msg.channels[3] = 0    # Yaw
		self.msg.channels[4] = 0
		self.msg.channels[5] = 0
		self.msg.channels[6] = 0
		self.msg.channels[7] = 0
		rospy.loginfo('Sending RC THROTTLE %d', self.msg.channels[2])
		self.pub.publish(self.msg)

		time.sleep(0.1)

		# rospy.loginfo('Changing mode to ALT_HOLD')
		# # Set ALT_HOLD mode
		# rospy.wait_for_service('/mavros/set_mode')
		# try:
		# 	self.mode_proxy(0,'ALT_HOLD')
		# except rospy.ServiceException, e:
		# 	print ("/mavros/set_mode service call failed: %s"%e)

	def _launch_apm(self):
		sim_vehicle_sh = str(os.environ["ARDUPILOT_PATH"]) + "/Tools/autotest/sim_vehicle.sh"
		subprocess.Popen(["xterm","-e",sim_vehicle_sh,"-j4","-f","Gazebo", "-S10", "-v","ArduCopter"])

	def _pause(self, msg):
		programPause = raw_input(str(msg))

	def __init__(self):

		self._launch_apm()

		RED = '\033[91m'
		BOLD = '\033[1m'
		ENDC = '\033[0m'        
		LINE = "%s%s##############################################################################%s" % (RED, BOLD, ENDC)
		msg = "\n%s\n" % (LINE)
		msg += "%sLoad Erle-Copter parameters in MavProxy console (sim_vehicle.sh):%s\n\n" % (BOLD, ENDC)
		msg += "MAV> param load %s\n\n" % (str(os.environ["ERLE_COPTER_PARAM_PATH"]))
		msg += "%sThen, press <Enter> here to launch Gazebo...%s\n\n%s" % (BOLD, ENDC,  LINE)
		# self._pause(msg)
		print(str(msg))
		time.sleep(3)

		# Launch the simulation with the given launchfile name
		gazebo_env.GazeboEnv.__init__(self, "GazeboErleCopterHover-v0.launch")    

		self.num_actions = 9
		self.action_space = spaces.Discrete(self.num_actions) # F, L, R, B
		self.reward_range = (-np.inf, np.inf)

		# self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		# self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
		self.mode_proxy = rospy.ServiceProxy('/mavros/set_mode', SetMode)
		self.param_set_proxy = rospy.ServiceProxy('/mavros/param/set', ParamSet)
		self.param_get_proxy = rospy.ServiceProxy('/mavros/param/get', ParamGet)
		self.arm_proxy = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
		self.takeoff_proxy = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)

		self.pub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=1)
		self.vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10, latch=False)
		self.setpoint_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10, latch=True)
		self.pose_subscriber = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.pose_callback)

		self.rtl_time = 5
		self.reset_time = 3
		self.disarm = False

		# CANNOT SET. ERROR.
		rospy.wait_for_service('/mavros/param/set')
		try:
			info = ParamSet()
			info.param_id = 'RTL_ALT'

			val = ParamValue()
			val.integer = 610
			val.real = 0.0
			info.value = val

			self.param_set_proxy(info.param_id, info.value)
			rospy.loginfo('Changed RTL_ALT to %d', val.integer)

		except rospy.ServiceException, e:
			print ("/mavros/set_mode service call failed: %s"%e)

		countdown = 10
		while countdown > 0:
			print ("Taking off in in %ds"%countdown)
			countdown-=1
			time.sleep(1)

		self._takeoff(2)

		self._seed()

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	# def _state(self, action):
	# 	return discretized_ranges, done

	def pose_callback(self, msg):
		position = msg.pose.position
		quat = msg.pose.orientation
		# rospy.loginfo("Point Position: [ %f, %f, %f ]"%(position.x, position.y, position.z))
		# rospy.loginfo("Quat Orientation: [ %f, %f, %f, %f]"%(quat.x, quat.y, quat.z, quat.w))
		euler = tf.transformations.euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
		# rospy.loginfo("Euler Angles: %s"%str(euler))
		self.pose = msg.pose
		self.euler = euler

	def _step(self, action):
		print "Taking action", action

		# ######### Postion ############## 
		# target_pose_msg = PoseStamped()
		# now = rospy.get_rostime()
		# target_pose_msg.header.frame_id = 'world'
		# target_pose_msg.header.stamp.secs = now.secs
		# target_pose_msg.header.stamp.nsecs = now.nsecs

		# curr_x = self.pose.position.x
		# curr_y = self.pose.position.y
		# curr_z = self.pose.position.z
		# current_yaw = self.euler[2]

		# speed = 5

		# action = action - 3 # 3 is forward, 0,1,2 are to left, separated by 10 deg each
		# vel_x_body = speed*math.sin(action*(math.pi/10))
		# vel_y_body = speed*math.cos(action*(math.pi/10))

		# v_x = vel_y_body*math.sin(current_yaw) + vel_x_body*math.cos(current_yaw)
		# v_y = vel_y_body*math.cos(current_yaw) - vel_x_body*math.sin(current_yaw)

		# delta = 0.2
		# target_pose_msg.pose.position.x = curr_x + v_x*delta
		# target_pose_msg.pose.position.y = curr_y + v_y*delta

		# # quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
		# print "current yaw", current_yaw
		# print "curr_x", curr_x, "curr_y", curr_y
		# print "target_x", curr_x + v_x*delta, "target_y", curr_y + v_y*delta
		# print "delta_x", v_x*delta, "delta_y", v_y*delta
		# print "\n"

		# target_pose_msg.pose.position.z = curr_z
		# target_pose_msg.pose.orientation = self.pose.orientation
		# self.setpoint_pub.publish(target_pose_msg)
		# time.sleep(0.2)
	
		######### RC ############## 

		# action_msg = OverrideRCIn()
		# mean_yaw_pwm = 1500
		# delta = 150

		# if action == 0: #FORWARD
		# 	action_msg.channels[3] = mean_yaw_pwm  # Yaw
		# elif action == 1: 
		# 	action_msg.channels[3] = mean_yaw_pwm + delta
		# elif action == 2: 
		# 	action_msg.channels[3] = mean_yaw_pwm + (delta*1)
		# elif action == 3: 
		# 	action_msg.channels[3] = mean_yaw_pwm + (delta*2)
		# elif action == 4:
		# 	action_msg.channels[3] = mean_yaw_pwm - delta
		# elif action == 5:
		# 	action_msg.channels[3] = mean_yaw_pwm - (delta*2)
		# elif action == 6:
		# 	action_msg.channels[3] = mean_yaw_pwm - (delta*3)

		# action_msg.channels[0] = 1500 # Roll
		# action_msg.channels[1] = 1425 # Pitch
		# action_msg.channels[2] = 1500 # Throttle
		# action_msg.channels[4] = 0
		# action_msg.channels[5] = 0
		# action_msg.channels[6] = 0
		# action_msg.channels[7] = 0

		# self.pub.publish(action_msg)
		# time.sleep(0.5)

		# action_msg.channels[3] = 0
		# action_msg.channels[1] = 1500
		# self.pub.publish(action_msg)
		# time.sleep(0.1)


		######### VELOCITY ############## 
		curr_x = self.pose.position.x
		current_yaw = self.euler[2]
		vel_cmd = TwistStamped()
		now = rospy.get_rostime()
		vel_cmd.header.stamp.secs = now.secs
		vel_cmd.header.stamp.nsecs = now.nsecs

		speed = 1

		delta_theta_deg = 10
		action = action - ((self.num_actions-1)/2) # 4 is forward, 0-3 are to left(now -3 to -1), separated by 10 deg each. 
		vel_x_body = speed*math.sin(action*(math.radians(delta_theta_deg)))
		vel_y_body = speed*math.cos(action*(math.radians(delta_theta_deg)))
		speed = 1

		vel_cmd.twist.linear.x = vel_x_body
		vel_cmd.twist.linear.y = vel_y_body
		# vel_cmd.twist.linear.x = vel_y_body*math.sin(current_yaw) + vel_x_body*math.cos(current_yaw)
		# vel_cmd.twist.linear.y = vel_y_body*math.cos(current_yaw) - vel_x_body*math.sin(current_yaw)
		vel_cmd.twist.linear.z = 0
		# quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
		print "current yaw", current_yaw
		print "taking action", action, ":: velocity (x,y,z)", vel_cmd.twist.linear.x, vel_cmd.twist.linear.y, vel_cmd.twist.linear.z
		self.vel_pub.publish(vel_cmd)
		time.sleep(0.1)
	
		observation = self._get_frame()
		
		data = None
		while data is None:
			try:
				data = rospy.wait_for_message('/scan', LaserScan, timeout = 5)
			except:
				pass

		# is_terminal = self.check_terminal(data)
		print "min laser", np.min(data.ranges)
		# print "max laser", np.max(data.ranges)
		state, is_terminal = self.discretize_observation(data,len(data.ranges))

		if not is_terminal:
			if action == 0:
				reward = 5
			else:
				reward = 1
		else:
			reward = -200

		return observation, reward, is_terminal, {}	

	def _get_frame(self):
		frame = None;
		while frame is None:
			try:
				frame = rospy.wait_for_message('/camera/rgb/image_raw',Image, timeout = 5)
				cv_image = CvBridge().imgmsg_to_cv2(frame, desired_encoding="passthrough")
				frame = np.asarray(cv_image)
				cv2.imshow('frame', frame)
				cv2.waitKey(1)
				return frame
			except:
				raise ValueError('could not get frame')

	# def _relaunch_apm(self):
	# 	pids = subprocess.check_output(["pidof","ArduCopter.elf"]).split()
	# 	for pid in pids:
	# 		os.system("kill -9 "+str(pid))
	# 	grep_cmd = "ps -ef | grep ardupilot"
	# 	result = subprocess.check_output([grep_cmd], shell=True).split()
	# 	pid = result[1]
	# 	os.system("kill -9 "+str(pid))
	# 	grep_cmd = "ps -af | grep sim_vehicle.sh"
	# 	result = subprocess.check_output([grep_cmd], shell=True).split()
	# 	pid = result[1]
	# 	os.system("kill -9 "+str(pid))
	# 	self._launch_apm()

	# def _reset(self):
	# 	# Resets the state of the environment and returns an initial observation.
	# 	rospy.wait_for_service('/gazebo/reset_world')
	# 	try:
	# 		#reset_proxy.call()
	# 		self.reset_proxy()
	# 	except rospy.ServiceException, e:
	# 		print ("/gazebo/reset_world service call failed")
	# 	# Relaunch autopilot
	# 	self._relaunch_apm()
	# 	self._takeoff(2)
	# 	self.initial_latitude = None
	# 	self.initial_longitude = None
	# 	return self._get_frame()

		
	def _reset(self):
		################# RTL ##################

		# Resets the state of the environment and returns an initial observation.
		rospy.loginfo('Changing mode to RTL')
		# Set RTL mode
		rospy.wait_for_service('/mavros/set_mode')
		try:
			self.mode_proxy(0,'RTL')
		except rospy.ServiceException, e:
			print ("/mavros/set_mode service call failed: %s"%e)

		rospy.loginfo('Waiting to land')
		time.sleep(1)
		alt_msg = None
		erlecopter_alt = float('inf')
		while erlecopter_alt > 0.01:
			try:
				alt_msg = rospy.wait_for_message('/mavros/global_position/rel_alt', Float64, timeout=10)
				erlecopter_alt = alt_msg.data
			except:
				pass

		crash_msg = OverrideRCIn()
		crash_msg.channels[0] = 0
		crash_msg.channels[1] = 0
		crash_msg.channels[2] = 0
		crash_msg.channels[3] = 0
		crash_msg.channels[4] = 0
		crash_msg.channels[5] = 0
		crash_msg.channels[6] = 0
		crash_msg.channels[7] = 0
		rospy.loginfo('Sending RC THROTTLE %d', self.msg.channels[2])
		self.pub.publish(crash_msg)

		rospy.loginfo('Changing mode to STABILIZE')
		# Set STABILIZE mode
		rospy.wait_for_service('/mavros/set_mode')
		try:
			self.mode_proxy(0,'STABILIZE')
		except rospy.ServiceException, e:
			print ("/mavros/set_mode service call failed: %s"%e)

		rospy.loginfo('DISARMing throttle')
		rospy.wait_for_service('/mavros/cmd/arming')
		try:
			self.arm_proxy(False)
		except rospy.ServiceException, e:
			print ("/mavros/set_mode service call failed: %s"%e)

		rospy.loginfo('Gazebo RESET')
		self.reset_proxy()

		self._takeoff(2)

		################# (DE)STABILIZE ##################
		# time.sleep(1)
		# # self.msg.channels[0] = 0
		# # self.msg.channels[1] = 0
		# self.msg.channels[2] = 0
		# # self.msg.channels[3] = 0
		# # self.msg.channels[4] = 0
		# # self.msg.channels[5] = 0
		# # self.msg.channels[6] = 0
		# # self.msg.channels[7] = 0
		# rospy.loginfo('Sending RC THROTTLE %d', self.msg.channels[2])
		# self.pub.publish(self.msg)

		# time.sleep(2)

		# rospy.loginfo('Changing mode to STABILIZE')
		# # Set STABILIZE mode
		# rospy.wait_for_service('/mavros/set_mode')
		# try:
		# 	self.mode_proxy(0,'STABILIZE')
		# except rospy.ServiceException, e:
		# 	print ("/mavros/set_mode service call failed: %s"%e)

		# time.sleep(2)

		# rospy.loginfo('Gazebo RESET')
		# self.reset_proxy()

		# time.sleep(self.reset_time)
		# self._takeoff(2)

		################# GHOST MODE ##################
		# delta_yaw = 1.57
		# target_yaw = self.euler[2] + delta_yaw
		# print "old yaw", self.euler[2], "new yaw", target_yaw
		# target_quat = tf.transformations.quaternion_from_euler(self.euler[0], self.euler[1], target_yaw)

		# setpoint_msg = PoseStamped()
		# now = rospy.get_rostime()
		# setpoint_msg.header.stamp.secs = now.secs
		# setpoint_msg.header.stamp.nsecs = now.nsecs

		# setpoint_msg.pose = self.pose
		# setpoint_msg.pose.orientation.x = target_quat[0]
		# setpoint_msg.pose.orientation.y = target_quat[1]
		# setpoint_msg.pose.orientation.z = target_quat[2]
		# setpoint_msg.pose.orientation.w = target_quat[3]
		# self.setpoint_pub.publish(setpoint_msg)
		# print "sent new yaw. wait for 2 seconds"
		# time.sleep(2)

		return self._get_frame()

	def discretize_observation(self,data,new_ranges):
		# print data
		discretized_ranges = []
		min_range = 2.0
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
			if (min_range > data.ranges[i] > 0):
				done = True
		return discretized_ranges,done