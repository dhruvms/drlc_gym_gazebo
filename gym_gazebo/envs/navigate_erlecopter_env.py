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

from mavros_msgs.msg import OverrideRCIn
from sensor_msgs.msg import LaserScan, NavSatFix, Image
from std_msgs.msg import Float64
from gazebo_msgs.msg import ModelStates

from mavros_msgs.srv import CommandBool, CommandTOL, SetMode
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist

from gazebo_msgs.msg import ContactState

import tf

class GazeboErleCopterHoverEnv(gazebo_env.GazeboEnv):

	def _takeoff(self, altitude):
		print "Waiting for mavros..."
		data = None
		while data is None:
			try:
				data = rospy.wait_for_message('/mavros/global_position/rel_alt', Float64, timeout=5)
			except:
				pass
		
		takeoff_successful = False
		while not takeoff_successful:
			print "Taking off..."
			alt = altitude
			err = alt * 0.1 # 10% error

			#pub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=10)

			#msg = OverrideRCIn()
			#msg.channels[0] = 0 # Roll
			#msg.channels[1] = 0 # Pitch
			#msg.channels[2] = 1500 # Throttle
			#msg.channels[3] = 0    # Yaw
			#msg.channels[4] = 0
			#msg.channels[5] = 0
			#msg.channels[6] = 0
			#msg.channels[7] = 0
			#self.pub.publish(msg)

			# Set GUIDED mode
			rospy.wait_for_service('mavros/set_mode')
			try:
				self.mode_proxy(0,'GUIDED')
			except rospy.ServiceException, e:
				print ("mavros/set_mode service call failed: %s"%e)

			# Wait 2 seconds
			time.sleep(2)

			# Arm throttle
			rospy.wait_for_service('mavros/cmd/arming')
			try:
				self.arm_proxy(True)
			except rospy.ServiceException, e:
				print ("mavros/set_mode service call failed: %s"%e)

			# Takeoff
			rospy.wait_for_service('mavros/cmd/takeoff')
			try:
				self.takeoff_proxy(0, 0, 0, 0, alt) # 1m altitude
			except rospy.ServiceException, e:
				print ("mavros/cmd/takeoff service call failed: %s"%e)

			# Wait 3 seconds
			time.sleep(3)

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
			erlecopter_alt = alt_msg.pose[erlecopter_index].position.z * 2
			if erlecopter_alt > (alt - err):
				takeoff_successful = True
				print "Takeoff successful"
			else:
				print "Takeoff failed, retrying..."

		# Set ALT_HOLD mode
		rospy.wait_for_service('mavros/set_mode')
		try:
			self.mode_proxy(0,'ALT_HOLD')
		except rospy.ServiceException, e:
			print ("mavros/set_mode service call failed: %s"%e)

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
		self._pause(msg)

		# Launch the simulation with the given launchfile name
		gazebo_env.GazeboEnv.__init__(self, "GazeboErleCopterHover-v0.launch")    

		self.action_space = spaces.Discrete(4) # F, L, R, B
		#self.observation_space = spaces.Box(low=0, high=20) #laser values
		self.reward_range = (-np.inf, np.inf)

		self.initial_latitude = None
		self.initial_longitude = None

		self.current_latitude = None
		self.current_longitude = None

		self.diff_latitude = None
		self.diff_longitude = None

		self.max_distance = 1.6

		# self.pub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=10)
		self.pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', Twist, queue_size=10)
		# self.collision = rospy.Subscriber('/gazebo/default/box/link/my_contact', ContactState , self._update_reward(self))


		#self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

		#self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)

		self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)

		self.mode_proxy = rospy.ServiceProxy('mavros/set_mode', SetMode)

		self.arm_proxy = rospy.ServiceProxy('mavros/cmd/arming', CommandBool)
		
		self.takeoff_proxy = rospy.ServiceProxy('mavros/cmd/takeoff', CommandTOL)

		# self.b_collision = False

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
		vel_cmd = Twist()

		speed = 1
		pi = math.pi

		vel_cmd.linear.x = speed*cos(action*(pi/10))
		vel_cmd.linear.x = speed*sin(action*(pi/10))

		# quaternion = tf.transformations.quaternion_from_euler(roll, pitch, yaw)

		self.pub.publish(vel_cmd)
	
		observation = self._get_frame()
	   
		

		data = None
		while data is None:
			try:
				data = rospy.wait_for_message('gazebo/circle/physics/contact', ContactState, timeout = 5)
			except:
				pass

		if len(data.contact_positions) > 0:
			reward = -100
			is_terminal = True
		else:
			reward = 10
			is_terminal = False


		# data = None
		# while data is None:
		#     try:
		#         data = rospy.wait_for_message('/scan', LaserScan, timeout = 5)
		#     except:
		#         pass

		# # is_terminal = self.check_terminal(data)
		# state,is_terminal = self.discretize_observation(data,len(data.ranges))

		# if is_terminal:
		#     reward = -100
		# else:
		#     reward = 10 

		return observation, reward, is_terminal, {}


	def _killall(self, process_name):
		pids = subprocess.check_output(["pidof",process_name]).split()
		for pid in pids:
			os.system("kill -9 "+str(pid))

	def _relaunch_apm(self):
		pids = subprocess.check_output(["pidof","ArduCopter.elf"]).split()
		for pid in pids:
			os.system("kill -9 "+str(pid))
		
		grep_cmd = "ps -ef | grep ardupilot"
		result = subprocess.check_output([grep_cmd], shell=True).split()
		pid = result[1]
		os.system("kill -9 "+str(pid))

		grep_cmd = "ps -af | grep sim_vehicle.sh"
		result = subprocess.check_output([grep_cmd], shell=True).split()
		pid = result[1]
		os.system("kill -9 "+str(pid))  

		self._launch_apm()

	def _to_meters(self, n):
		return n * 100000.0


	def _get_frame(self):
		frame = None;
		while frame is None:
			try:
				frame = rospy.wait_for_message('/camera/depth/image_raw',Image, timeout = 5)

		cv_image = CvBridge().imgmsg_to_cv2(frame, desired_encoding="passthrough")
		frame = np.asarray(cv_image)

		return frame


	def center_distance(self):
		return math.sqrt(self.diff_latitude**2 + self.diff_longitude**2)

	def _reset(self):
		# Resets the state of the environment and returns an initial observation.
		rospy.wait_for_service('/gazebo/reset_world')
		try:
			#reset_proxy.call()
			self.reset_proxy()
		except rospy.ServiceException, e:
			print ("/gazebo/reset_world service call failed")

		# Relaunch autopilot
		self._relaunch_apm()

		self._takeoff(2)

		self.initial_latitude = None
		self.initial_longitude = None
		# self.b_collision = False
		
		return self._get_position()


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