"""Main DQN agent."""

import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Reshape, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import keras

from objectives import *
import gym
import numpy as np
from policy import *
import preprocessors
from core import *
import matplotlib.pyplot as plt
import cPickle as pkl
import os
from gym import wrappers

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#K.set_session(tf.Session(config=config))

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.

    Feel free to change the functions and funciton parameters that the
    class provides.

    We have provided docstrings to go along with our suggested API.

    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """
    def __init__(self,
                 env,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 mode,
                 resume_dir,
                 log_parent_dir = '/home/vaibhav/madratman/logs/project/dqn'):

        self.env_string = env
        self.env = gym.make(env)
        self.num_actions = self.env.action_space.n
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        if resume_dir is None:
            self.train_iter_ctr = 0

        self.eval_episode_ctr = 0
        self.preprocessor = preprocessors.PreprocessorSequence()

        # loggers
        self.qavg_list = np.array([0])
        self.reward_list = []
        self.loss_log = []
        self.loss_last = None
        self.mode = mode
        self.log_parent_dir = log_parent_dir
        # self.make_log_dir() # makes empty dir and logfiles based on current timestamp inside self.log_parent_dir
        if resume_dir is not None:
            print "resuming from ", resume_dir
            self.resume_from_log_dir(resume_dir)
            self.is_resume=True
        print "self.is_resume", self.is_resume

    def create_model(self):  # noqa: D103
        """Create the Q-network model.

        Use Keras to construct a keras.models.Model instance (you can also
        use the SequentialModel class).

        We highly recommend that you use tf.name_scope as discussed in
        class when creating the model and the layers. This will make it
        far easier to understnad your network architecture if you are
        logging with tensorboard.

        Parameters
        ----------
        window: int
          Each input to the network is a sequence of frames. This value
          defines how many frames are in the sequence.
        input_shape: tuple(int, int)
          The expected input image size.
        num_actions: int
          Number of possible actions. Defined by the gym environment.
        model_name: str
          Useful when debugging. Makes the model show up nicer in tensorboard.

        Returns
        -------
        keras.models.Model
          The Q-model.
        """
        # reference for creation of the model https://yilundu.github.io/2016/12/24/Deep-Q-Learning-on-Space-Invaders.html
        model=Sequential()
        model.add(Convolution2D (32,8,8, subsample = (4,4), input_shape=(84,84,4) ))
        model.add(Activation('relu'))
        model.add(Convolution2D (64,4,4, subsample = (2,2) ))
        model.add(Activation('relu'))
        model.add(Convolution2D (64,3,3, subsample = (1,1) ))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.num_actions)) 

        return model

    def compile(self, is_resum=False):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        # create both networks
        self.q_network = self.create_model()
        self.target_q_network = self.create_model()

        # set loss function in both 
        adam = Adam(lr=1e-4)
        self.q_network.compile(loss=mean_huber_loss, optimizer=adam) 
        self.target_q_network.compile(loss=mean_huber_loss, optimizer=adam)
        RED = '\033[91m'
        BOLD = '\033[1m'
        ENDC = '\033[0m'        
        LINE = "%s%s##############################################################################%s" % (RED, BOLD, ENDC)

        if self.is_resume:
            last_weight_file = sorted(os.listdir(os.path.join(self.log_dir, 'weights')))[-1]
            # second_last_weight_file = sorted(os.listdir(os.path.join(self.log_dir, 'weights')))[-2]
            str_1 = "Loading q_net from " + os.path.join(self.log_dir, 'weights', last_weight_file)
            # str_1 = str_1 + '\n' + "Loading target_q_net from" + os.path.join(self.log_dir, 'weights', second_last_weight_file)
            # str_1 = str_1 + '\n' + "Loading target_q_net from" + os.path.join(self.log_dir, 'weights', last_weight_file)
            msg = "\n%s\n" % (LINE)
            msg += "%s%s\n" % (BOLD, str_1)
            msg += "%s\n" % (LINE)
            print(str(msg))
            # self.target_q_network.load_weights(os.path.join(self.log_dir, 'weights',second_last_weight_file))
            self.q_network.load_weights(os.path.join(self.log_dir, 'weights',last_weight_file))
            # self.target_q_network.load_weights(os.path.join(self.log_dir, 'weights',last_weight_file))

        # set the same weights for both initially
        self.target_q_network.set_weights(self.q_network.get_weights())
        print self.q_network.summary()

    def calc_q_values(self, state):
        """Given a state (or batch of states) calculate the Q-values.
        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """ 
        q_vals = self.q_network.predict(np.swapaxes(state,0,3),batch_size=1)
        return q_vals

    def make_log_dir(self):
        import datetime, os
        current_timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(self.log_parent_dir, self.env_string, self.mode, current_timestamp)
        os.makedirs(self.log_dir)
        os.makedirs(os.path.join(self.log_dir, 'weights'))
        os.makedirs(os.path.join(self.log_dir, 'replay_memory'))
        os.makedirs(os.path.join(self.log_dir, 'gym_monitor'))
        # create empty logfiles now
        self.log_files = {
                            'train_loss': os.path.join(self.log_dir, 'train_loss.txt'),
                            'train_episode_reward': os.path.join(self.log_dir, 'train_episode_reward.txt'),
                            'test_episode_reward': os.path.join(self.log_dir, 'test_episode_reward.txt')
                          }

        for key in self.log_files:
            open(os.path.join(self.log_dir, self.log_files[key]), 'a').close()

    def resume_from_log_dir(self, resume_dir):
        self.log_dir = resume_dir
        self.log_files = {
                            'train_loss': os.path.join(self.log_dir, 'train_loss.txt'),
                            'train_episode_reward': os.path.join(self.log_dir, 'train_episode_reward.txt'),
                            'test_episode_reward': os.path.join(self.log_dir, 'test_episode_reward.txt')
                          }
        #picks the last replay memory(with most iters acc to file name)
        latest_replay_memory_file = sorted(os.listdir(os.path.join(self.log_dir, 'replay_memory')))[-1]
        with open(os.path.join(self.log_dir, 'replay_memory', latest_replay_memory_file), 'rb') as f:
            self.replay_memory = pkl.load(f)
        self.train_iter_ctr = int(latest_replay_memory_file.split('.')[0].split('_')[-1])
        print "self.train_iter_ctr", self.train_iter_ctr 

    def dump_train_loss(self, loss):
        self.loss_last = loss
        with open(self.log_files['train_loss'], "a") as f:
            f.write(str(self.train_iter_ctr) + ' ' + str(self.train_episode_ctr) + ' ' + str(loss) + '\n')

    def dump_train_episode_reward(self, episode_reward):
        with open(self.log_files['train_episode_reward'], "a") as f:
            f.write(str(self.train_iter_ctr) + ' ' + str(self.train_episode_ctr) + ' ' + str(episode_reward) + '\n')

    def dump_test_episode_reward(self, episode_reward):
        with open(self.log_files['test_episode_reward'], "a") as f:
            f.write(str(self.train_iter_ctr) + ' ' + str(self.train_episode_ctr) + ' ' + str(episode_reward) + '\n')

    # ref http://stackoverflow.com/questions/37902705/how-to-manually-create-a-tf-summary 
    # https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514#file-tensorboard_logging-py-L41
    def tf_log_scaler(self, tag, value, step):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.tf_summary_writer.add_summary(summary, step)

    def update_policy(self):
        """Update your policy.

        Behavior may differ based on what stage of training your
        in. If you're in training mode then you should check if you
        should update your network parameters based on the current
        step and the value you set for train_freq.

        Inside, you'll want to sample a minibatch, calculate the
        target values, update your network, and then update your
        target values.

        You might want to return the loss and other metrics as an
        output. They can help you monitor how training is going.
        """
        # this is update_policy 
        # sample batch of 32 from the memory
        batch_of_samples = self.replay_memory.sample(batch_size=32)
        current_state_samples = batch_of_samples['current_state_samples']
        next_state_samples = batch_of_samples['next_state_samples']
        #print type(current_state_samples[0])

        # fetch stuff we need from samples 32*84*84*4
        current_state_images = np.zeros([32, 84, 84, 4])
        for (idx, each_list_of_samples) in enumerate(current_state_samples):
            current_state_images[idx, ...] = np.dstack([sample.state for sample in each_list_of_samples])

        next_state_images = np.zeros([32, 84, 84, 4])
        for (idx, each_list_of_samples) in enumerate(next_state_samples):
            next_state_images[idx, ...] = np.dstack([sample.state for sample in each_list_of_samples])

        # preprocess
        current_state_images = self.preprocessor.process_batch(current_state_images)
        next_state_images = self.preprocessor.process_batch(next_state_images)
        # print "current_state_images {} max {} ".format(current_state_images.shape, np.max(current_state_images))

        q_current = self.q_network.predict(current_state_images,batch_size=self.batch_size) # 32*num_actions
        q_next = self.target_q_network.predict(next_state_images,batch_size=self.batch_size)

        # targets
        y_targets_all = q_current #32*num_actions

        for (idx, each_list_of_samples) in enumerate(current_state_samples):
            last_sample = each_list_of_samples[-1]
            if last_sample.is_terminal:
                y_targets_all[idx, last_sample.action] = last_sample.reward
            else:
                if self.mode == 'vanilla':
                    y_targets_all[idx, last_sample.action] = np.float32(last_sample.reward) + self.gamma*np.max(q_next[idx])
                if self.mode == 'double':               
                    y_targets_all[idx, last_sample.action] = np.float32(last_sample.reward) + self.gamma*q_next[idx, np.argmax(q_current[idx])] 

        loss = self.q_network.train_on_batch(current_state_images, np.float32(y_targets_all))

        with tf.name_scope('summaries'):
            self.tf_log_scaler(tag='train_loss', value=loss, step=self.train_iter_ctr)

        if not (self.train_iter_ctr % self.log_loss_every_nth):
            self.dump_train_loss(loss)

        if (self.train_iter_ctr > (self.num_burn_in+1)) and not(self.train_iter_ctr%self.target_update_freq):
            # copy weights
            print "Iter {} Updating target Q network".format(self.train_iter_ctr)
            self.target_q_network.set_weights(self.q_network.get_weights())
            # [self.target_q_network.trainable_weights[i].assign(self.q_network.trainable_weights[i]) \
            #     for i in range(len(self.target_q_network.trainable_weights))]

    def fit(self, num_iterations, max_episode_length=250, eval_every_nth=1000, save_model_every_nth=1000, log_loss_every_nth=1000, video_every_nth=20000):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: gym.Env
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        self.compile(self.is_resume)
        self.policy = LinearDecayGreedyEpsilonPolicy(start_value=1., end_value=0.1, num_steps=100000, num_actions=self.num_actions) # for training
        self.policy.epsilon = 1.0 - ((1 - 0.1) / 100000 * self.train_iter_ctr)
        self.log_loss_every_nth = log_loss_every_nth
        random_policy = UniformRandomPolicy(num_actions=self.num_actions) # for burn in 
        self.train_episode_ctr = 0

        # tf logging # todo how to resume
        self.tf_session = K.get_session()
        self.tf_summary_writer = tf.summary.FileWriter(self.log_dir, self.tf_session.graph)

        while self.train_iter_ctr < num_iterations:
            state = self.env.reset()
            self.preprocessor.reset_history_memory()

            num_timesteps_in_curr_episode = 0
            total_reward_curr_episode = 0       

            while num_timesteps_in_curr_episode < max_episode_length:
                self.train_iter_ctr+=1 # number of steps overall
                num_timesteps_in_curr_episode += 1 # number of steps in the current episode
                RED = '\033[91m'
                BOLD = '\033[1m'
                ENDC = '\033[0m'        
                LINE = "%s%s##############################################################################%s" % (RED, BOLD, ENDC)

                # # logging
                # if not self.train_iter_ctr % 10:
                #     RED = '\033[91m'
                #     BOLD = '\033[1m'
                #     ENDC = '\033[0m'        
                #     LINE = "%s%s##############################################################################%s" % (RED, BOLD, ENDC)
                #     str_1 = "iter_ctr {}, self.train_episode_ctr : {} num_timesteps_in_curr_episode {}".format(self.train_iter_ctr, self.train_episode_ctr, num_timesteps_in_curr_episode)
                #     msg = "\n%s\n" % (LINE)
                #     msg += "%s%s\n" % (BOLD, str_1)
                #     msg += "%s\n" % (LINE)
                #     print(str(msg))

                # this appends to uint8 history and also returns stuff ready to be spit into the  network
                state_network = self.preprocessor.process_state_for_network(state) #shape is (4,84,84,1). axis are swapped in cal_q_vals
                # print "shape {}, max {}, min {}, type {} ".format(state_network.shape, np.max(state_network), np.min(state_network), state_network.dtype)

                # burning in 
                if self.train_iter_ctr < self.num_burn_in:
                    action = random_policy.select_action() # goes from 0 to n-1
                    # print "\ntaking action", action, "\n"

                    next_state, reward, is_terminal, _ = self.env.step(action)
                    total_reward_curr_episode += reward
                    state_proc_memory = self.preprocessor.process_state_for_memory(state)
                    # atari_preprocessor.process_state_for_memory converts it to grayscale, resizes it to (84, 84) and converts to uint8
                    self.replay_memory.append(state_proc_memory, action, reward, is_terminal)

                    if is_terminal or (num_timesteps_in_curr_episode > max_episode_length-1):
                        # state = self.env.reset()
                        self.train_episode_ctr += 1
                        with tf.name_scope('summaries'):
                            self.tf_log_scaler(tag='train_reward_per_episode_wrt_no_of_episodes', value=total_reward_curr_episode, step=self.train_episode_ctr)
                            self.tf_log_scaler(tag='train_reward_per_episode_wrt_iterations', value=total_reward_curr_episode, step=self.train_iter_ctr)
                        str_1 = "iter_ctr {}, self.train_episode_ctr : {}, episode_reward : {}, loss : {}, episode_timesteps : {}, epsilon : {}".format\
                                (self.train_iter_ctr, self.train_episode_ctr, total_reward_curr_episode, self.loss_last, num_timesteps_in_curr_episode, self.policy.epsilon)
                        msg = "\n%s\n" % (LINE)
                        msg += "%s%s\n" % (BOLD, str_1)
                        msg += "%s\n" % (LINE)
                        print(str(msg))
                        num_timesteps_in_curr_episode = 0
                        self.dump_train_episode_reward(total_reward_curr_episode)
                        # this should be called when num_timesteps_in_curr_episode > max_episode_length, but we can call it in is_terminal as well. 
                        # it won't change anything as it just sets the last entry's is_terminal to True
                        self.replay_memory.end_episode() 
                        break

                # training
                else:
                    # print "iter_ctr {}, self.train_episode_ctr : {} num_timesteps_in_curr_episode {}".format(self.train_iter_ctr, self.train_episode_ctr, num_timesteps_in_curr_episode)
                    q_values = self.calc_q_values(state_network)
                    # print "q_values {} q_values.shape {}".format(q_values, q_values.shape)
                    #print "q_values.shape ", q_values.shape
                    action = self.policy.select_action(q_values=q_values, is_training=True)
                    next_state, reward, is_terminal, _ = self.env.step(action)
                    total_reward_curr_episode += reward
                    state_proc_memory = self.preprocessor.process_state_for_memory(state)
                    self.replay_memory.append(state_proc_memory, action, reward, is_terminal)

                    # validation. keep this clause before the breaks!
                    if not(self.train_iter_ctr%eval_every_nth):
                        print "\n\nEvaluating at iter {}".format(self.train_iter_ctr)
                        if not(self.train_iter_ctr%video_every_nth):
                            # self.evaluate(self.train_episode_ctr=20, max_episode_length=max_episode_length, gen_video=True)
                            self.evaluate(num_episodes=5, max_episode_length=max_episode_length, gen_video=False)
                        else:
                            self.evaluate(num_episodes=5, max_episode_length=max_episode_length, gen_video=False)
                        print "Done Evaluating\n\n"

                    # save model
                    if not(self.train_iter_ctr%save_model_every_nth):
                        self.q_network.save(os.path.join(self.log_dir, 'weights/q_network_{}.h5'.format(str(self.train_iter_ctr).zfill(7))))
                        output = open(os.path.join(self.log_dir, 'replay_memory/iter_{}.pkl'.format(str(self.train_iter_ctr).zfill(7))), 'wb')
                        
                        # THIS LINE GAVE ERROR BECAUSE MYDICT DOES NOT EXIST
                        # pkl.dump(mydict, output)
                        
                        pkl.dump(self.replay_memory, output)
                        output.close()

                    if is_terminal or (num_timesteps_in_curr_episode > max_episode_length-1):
                        # state = self.env.reset()
                        self.train_episode_ctr += 1
                        with tf.name_scope('summaries'):
                            self.tf_log_scaler(tag='train_reward_per_episode_wrt_no_of_episodes', value=total_reward_curr_episode, step=self.train_episode_ctr)
                            self.tf_log_scaler(tag='train_reward_per_episode_wrt_iterations', value=total_reward_curr_episode, step=self.train_iter_ctr)
                        str_1 = "iter_ctr {}, self.train_episode_ctr : {}, episode_reward : {}, loss : {}, episode_timesteps : {}, epsilon : {}".format\
                                (self.train_iter_ctr, self.train_episode_ctr, total_reward_curr_episode, self.loss_last, num_timesteps_in_curr_episode, self.policy.epsilon)
                        msg = "\n%s\n" % (LINE)
                        msg += "%s%s\n" % (BOLD, str_1)
                        msg += "%s\n" % (LINE)
                        print(str(msg))
                        num_timesteps_in_curr_episode = 0
                        self.dump_train_episode_reward(total_reward_curr_episode)
                        self.replay_memory.end_episode() 
                        break

                    if not(self.train_iter_ctr % self.train_freq):
                        self.update_policy()

                state = next_state

    def evaluate(self, num_episodes, max_episode_length=None, gen_video=False):
        evaluation_policy = GreedyPolicy()
        eval_preprocessor = preprocessors.PreprocessorSequence()
        # env_valid = gym.make(self.env_string)

        iter_ctr_valid = 0
        Q_sum = 0
        eval_episode_ctr_valid = 0
        total_reward_all_episodes = []
  
        # https://github.com/openai/gym/blob/master/gym/wrappers/monitoring.py video_callable takes function as arg. so we hack with true lambda
        # https://github.com/openai/gym/issues/494  
        if gen_video:
            video_dir = os.path.join(self.log_dir, 'gym_monitor', str(self.train_iter_ctr).zfill(7))
            os.makedirs(video_dir)
            env_valid = wrappers.Monitor(env_valid, video_dir, video_callable=lambda x:True, mode='evaluation')
        RED = '\033[91m'
        BOLD = '\033[1m'
        ENDC = '\033[0m'        
        LINE = "%s%s##############################################################################%s" % (RED, BOLD, ENDC)

        while eval_episode_ctr_valid < num_episodes:
            state = self.env.reset()
            eval_preprocessor.reset_history_memory()
            num_timesteps_in_curr_episode = 0
            total_reward_curr_episode = 0.0

            while num_timesteps_in_curr_episode < max_episode_length:
                num_timesteps_in_curr_episode += 1
                iter_ctr_valid += 1

                state_network = self.preprocessor.process_state_for_network(state)
                q_values = self.calc_q_values(state_network)
                Q_sum += np.max(q_values) # todo fix this

                action = evaluation_policy.select_action(q_values)
                next_state, reward, is_terminal, _ = self.env.step(action)
                total_reward_curr_episode += reward
                # print "Evalution : timestep {}, episode {}, action {}, reward {}, total_reward {}"\
                        # .format(iter_ctr_valid, eval_episode_ctr_valid, action, reward, total_reward_curr_episode)

                if is_terminal or (num_timesteps_in_curr_episode > max_episode_length-1):
                    eval_episode_ctr_valid += 1
                    str_1 = "Evaluate() : iter_ctr_valid {}, eval_episode_ctr_valid : {}, total_reward_curr_episode : {}, num_timesteps_in_curr_episode {}"\
                            .format(iter_ctr_valid, eval_episode_ctr_valid, total_reward_curr_episode, num_timesteps_in_curr_episode)
                    msg = "\n%s\n" % (LINE)
                    msg += "%s%s\n" % (BOLD, str_1)
                    msg += "%s\n" % (LINE)
                    print(str(msg))

                    total_reward_all_episodes.append(total_reward_curr_episode)
                    # num_timesteps_in_curr_episode = 0
                    break

                state = next_state

        Q_avg = Q_sum/float(iter_ctr_valid)
        print " sum(total_reward_all_episodes) : {} , float(len(total_reward_all_episodes)) : {}".format\
                (sum(total_reward_all_episodes), float(len(total_reward_all_episodes)))
        all_episode_avg_reward = sum(total_reward_all_episodes)/float(len(total_reward_all_episodes))
        with tf.name_scope('summaries'):
            self.tf_log_scaler(tag='test_mean_avg_reward', value=all_episode_avg_reward, step=self.train_iter_ctr)
            self.tf_log_scaler(tag='test_mean_Q_max', value=Q_avg, step=self.train_iter_ctr)
        self.dump_test_episode_reward(all_episode_avg_reward)
        self.qavg_list = np.append(self.qavg_list, Q_avg)
        self.reward_list.append(all_episode_avg_reward)

        print "all_episode_avg_reward ", all_episode_avg_reward
        print "\n\n\n self.reward_list \n\n\n", self.reward_list

