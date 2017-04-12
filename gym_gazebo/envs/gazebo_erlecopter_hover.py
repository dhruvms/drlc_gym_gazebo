import gym
import numpy as np
import os
import rospy
import roslaunch
import subprocess
import time
import math

from gym import utils, spaces
from gym_gazebo.envs import gazebo_env
from gym.utils import seeding

from mavros_msgs.msg import OverrideRCIn, ParamValue
from sensor_msgs.msg import LaserScan, NavSatFix
from std_msgs.msg import Float64
from gazebo_msgs.msg import ModelStates

from mavros_msgs.srv import CommandBool, CommandTOL, SetMode, ParamSet, ParamGet
from std_srvs.srv import Empty

import pdb


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
        # sim_vehicle_sh = '/home/shohin/Libraries/simulation/ardupilot/Tools/autotest/sim_vehicle.sh'
        subprocess.Popen(["xterm","-e",sim_vehicle_sh,"-j4","-f","Gazebo","-v","ArduCopter"]) # 

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
        time.sleep(3)

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

        # self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        # self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.reset_proxy = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        self.mode_proxy = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.param_set_proxy = rospy.ServiceProxy('/mavros/param/set', ParamSet)
        self.param_get_proxy = rospy.ServiceProxy('/mavros/param/get', ParamGet)
        self.arm_proxy = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.takeoff_proxy = rospy.ServiceProxy('/mavros/cmd/takeoff', CommandTOL)
        self.pub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=1)
        self.alt_sub = rospy.Subscriber('/mavros/global_position/rel_alt', Float64, self.alt_callback)

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

            rospy.loginfo('Changed RTL_CLIMB_MIN to %d', val.integer)
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
        self.msg = OverrideRCIn()

        if action == 0: #FORWARD
            self.msg.channels[0] = 1500 # Roll
            self.msg.channels[1] = 1450 # Pitch
        elif action == 1: #LEFT
            self.msg.channels[0] = 1450 # Roll
            self.msg.channels[1] = 1500 # Pitch
        elif action == 2: #RIGHT
            self.msg.channels[0] = 1550 # Roll
            self.msg.channels[1] = 1500 # Pitch
        elif action == 3: #BACKWARDS
            self.msg.channels[0] = 1500 # Roll
            self.msg.channels[1] = 1550 # Pitch

        self.msg.channels[2] = 1500  # Throttle
        self.msg.channels[3] = 0     # Yaw
        self.msg.channels[4] = 0
        self.msg.channels[5] = 0
        self.msg.channels[6] = 0
        self.msg.channels[7] = 0

        self.pub.publish(self.msg)
    
        observation = self._get_position()

        dist = self.center_distance()
        done = dist > self.max_distance

        reward = 0
        if done:
            reward = -100
        else:
            reward = 10 - dist * 8

        return observation, reward, done, {}


    def _killall(self, process_name):
        pids = subprocess.check_output(["pidof",process_name]).split()
        for pid in pids:
            os.system("kill -9 "+str(pid))

    def _relaunch_apm(self):
        pids = subprocess.check_output(["pidof","ArduCopter.elf"]).split()
        for pid in pids:
            os.system("kill -9 "+str(pid))
        
        grep_cmd = "ps -ef | grep ardupilot"
        # result = subprocess.check_output([grep_cmd], shell=True).split()
        # pid = result[1]
        pid = os.popen(grep_cmd).read().split()[1]
        os.system("kill -9 "+str(pid))

        grep_cmd = "ps -af | grep sim_vehicle.sh"
        # result = subprocess.check_output([grep_cmd], shell=True).split()
        # pid = result[1]
        pid = os.popen(grep_cmd).read().split()[1]
        os.system("kill -9 "+str(pid))  

        self._launch_apm()

    def _to_meters(self, n):
        return n * 100000.0

    def _get_position(self):
        #read position data
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('/mavros/global_position/global', NavSatFix, timeout=5)
            except:
                pass

        self.current_latitude = self._to_meters(data.latitude)
        self.current_longitude = self._to_meters(data.longitude)

        if self.initial_latitude == None and self.initial_longitude == None:
            self.initial_latitude = self.current_latitude
            self.initial_longitude = self.current_longitude
            print "Initial latitude : %f, Initial Longitude : %f" % (self.initial_latitude,self.initial_longitude,)

        print "Current latitude : %f, Current Longitude : %f" % (self.current_latitude,self.current_longitude,)

        self.diff_latitude = self.current_latitude - self.initial_latitude
        self.diff_longitude = self.current_longitude - self.initial_longitude

        print "Diff latitude: %f, Diff Longitude: %f" % (self.diff_latitude,self.diff_longitude,)

        return self.diff_latitude, self.diff_longitude

    def center_distance(self):
        return math.sqrt(self.diff_latitude**2 + self.diff_longitude**2)

    def alt_callback(self, data):
        if data.data < 0.3:
            self.disarm = True

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

        self.initial_latitude = None
        self.initial_longitude = None
        
        return self._get_position()

# Param load command:
# param load /home/shohin/Libraries/simulation/ardupilot/Tools/Frame_params/Erle-Copter.param
# param load /home/vaibhav/madratman/projects/gym_gazebo_deps/simulation/ardupilot/Tools/Frame_params/Erle-Copter.param