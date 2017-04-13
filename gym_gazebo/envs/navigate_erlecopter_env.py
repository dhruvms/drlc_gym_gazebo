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

			time.sleep(1)

			rospy.loginfo('ARMing throttle')
			# Arm throttle
			rospy.wait_for_service('/mavros/cmd/arming')
			try:
				self.arm_proxy(True)
			except rospy.ServiceException, e:
				print ("/mavros/set_mode service call failed: %s"%e)

			time.sleep(1)
			
			rospy.loginfo('TAKEOFF to %d meters', alt)
			# Takeoff
			rospy.wait_for_service('/mavros/cmd/takeoff')
			try:
				self.takeoff_proxy(0, 0, 0, 0, alt) # 1m altitude
			except rospy.ServiceException, e:
				print ("/mavros/cmd/takeoff service call failed: %s"%e)

			time.sleep(alt)

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

		time.sleep(1)

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

		time.sleep(1)

		rospy.loginfo('Changing mode to ALT_HOLD')
		# Set ALT_HOLD mode
		rospy.wait_for_service('/mavros/set_mode')
		try:
			self.mode_proxy(0,'ALT_HOLD')
		except rospy.ServiceException, e:
			print ("/mavros/set_mode service call failed: %s"%e)

	def _launch_apm(self):
		sim_vehicle_sh = str(os.environ["ARDUPILOT_PATH"]) + "/Tools/autotest/sim_vehicle.sh"
		subprocess.Popen(["xterm","-e",sim_vehicle_sh,"-j4","-f","Gazebo","-v","ArduCopter"])

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

		self.action_space = spaces.Discrete(10) # F, L, R, B
		#self.observation_space = spaces.Box(low=0, high=20) #laser values
		self.reward_range = (-np.inf, np.inf)

		self.initial_latitude = None
		self.initial_longitude = None

		self.current_latitude = None
		self.current_longitude = None

		self.diff_latitude = None
		self.diff_longitude = None

		self.max_distance = 1.6
		self.vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)

		# self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		# self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
		self.mode_proxy = rospy.ServiceProxy('/mavros/set_mode', SetMode)
		self.param_set_proxy = rospy.ServiceProxy('/mavros/param/set', ParamSet)
		self.param_get_proxy = rospy.ServiceProxy('/mavros/param/get', ParamGet)
		self.arm_proxy = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
		self.takeoff_proxy = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)
		self.pub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=1)

		self.rtl_time = 5
		self.reset_time = 3
		self.disarm = False

		# CANNOT SET. ERROR.
		rospy.wait_for_service('/mavros/param/set')
		try:
			info = ParamSet()
			info.param_id = 'RTL_ALT'

			val = ParamValue()
			val.integer = 2
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

	def _state(self, action):
		return discretized_ranges, done

	def _step(self, action):
		vel_cmd = TwistStamped()
		now = rospy.get_rostime()
		vel_cmd.header.stamp.secs = now.secs
		vel_cmd.header.stamp.nsecs = now.nsecs

		speed = 1
		pi = math.pi

		vel_cmd.twist.linear.x = speed*math.cos(action*(pi/10))
		vel_cmd.twist.linear.y = speed*math.sin(action*(pi/10))

		# quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

		self.vel_pub.publish(vel_cmd)
	
		observation = self._get_frame()
		

		# data = None
		# while data is None:
		# 	try:
		# 		data = rospy.wait_for_message('gazebo/circle/physics/contact', ContactState, timeout = 5)
		# 	except:
		# 		pass

		# if len(data.contact_positions) > 0:
		# 	reward = -100
		# 	is_terminal = True
		# else:
		# 	reward = 10
		# 	is_terminal = False


		data = None
		while data is None:
			try:
				data = rospy.wait_for_message('/scan', LaserScan, timeout = 5)
			except:
				pass

		# is_terminal = self.check_terminal(data)
		state, is_terminal = self.discretize_observation(data,len(data.ranges))

		if is_terminal:
			reward = -100
		else:
			reward = 10 

		return observation, reward, is_terminal, {}	


	# def _killall(self, process_name):
	# 	pids = subprocess.check_output(["pidof",process_name]).split()
	# 	for pid in pids:
	# 		os.system("kill -9 "+str(pid))

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

	# def _to_meters(self, n):
	# 	return n * 100000.0

	def _get_frame(self):
		frame = None;
		while frame is None:
			try:
				frame = rospy.wait_for_message('/camera/depth/image_raw',Image, timeout = 5)
				cv_image = CvBridge().imgmsg_to_cv2(frame, desired_encoding="passthrough")
				frame = np.asarray(cv_image)
				return frame
			except:
				raise ValueError('could not get frame')

	# def center_distance(self):
	# 	return math.sqrt(self.diff_latitude**2 + self.diff_longitude**2)

	def _reset(self):
		# Resets the state of the environment and returns an initial observation.
		# rospy.loginfo('Changing mode to RTL')
		# # Set RTL mode
		# rospy.wait_for_service('/mavros/set_mode')
		# try:
		#     self.mode_proxy(0,'RTL')
		# except rospy.ServiceException, e:
		#     print ("/mavros/set_mode service call failed: %s"%e)

		# rospy.loginfo('Waiting to land')
		# time.sleep(self.rtl_time)
		# # alt_msg = None
		# # erlecopter_alt = float('inf')
		# # while erlecopter_alt > 0.3:
		# #     try:
		# #         alt_msg = rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=10)
		# #         erlecopter_index = 0
		# #         for name in alt_msg.name:
		# #             if name == "erlecopter":
		# #                 break
		# #             else:
		# #                 erlecopter_index +=1
		# #         erlecopter_alt = alt_msg.pose[erlecopter_index].position.z
		# #     except:
		# #         pass
		# while not self.disarm:
		#     pass

		# rospy.loginfo('DISARMing throttle')
		# # Disrm throttle
		# rospy.wait_for_service('/mavros/cmd/arming')
		# try:
		#     self.arm_proxy(False)
		#     self.disarm = False
		# except rospy.ServiceException, e:
		#     print ("/mavros/set_mode service call failed: %s"%e)

		# time.sleep(1)

		self.msg.channels[2] = 0
		rospy.loginfo('Sending RC THROTTLE %d', self.msg.channels[2])
		self.pub.publish(self.msg)

		time.sleep(1)

		rospy.loginfo('Changing mode to STABILIZE')
		# Set STABILIZE mode
		rospy.wait_for_service('/mavros/set_mode')
		try:
			self.mode_proxy(0,'STABILIZE')
		except rospy.ServiceException, e:
			print ("/mavros/set_mode service call failed: %s"%e)

		time.sleep(1)

		rospy.loginfo('Gazebo RESET')
		self.reset_proxy()

		time.sleep(self.reset_time)

		self._takeoff(2)

		return self._get_frame()

	# def _update_reward(self, data):
	#     if len(data.contact_positions) >0:
	#          self.b_collision = True

	def check_terminal(delf, data):
		if data.range_min < 0.01:
			return False
		else:
			return True 


	def discretize_observation(self,data,new_ranges):
		discretized_ranges = []
		min_range = 0.2
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