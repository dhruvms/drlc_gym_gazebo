"""Main DQN agent."""

import pdb
import objectives
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
import moviepy.editor as mpy
from gym import wrappers
from keras.layers import (Activation, Lambda, Conv2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model


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
      The preprocessor class. See th
      e associated classes for more
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
                 q_network,
                 preprocessor,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 alpha,
                 model_name,
                 network_type = 'single',
                 b_target_fix = True,
                 save_frequency = 1000):
        self.q_network = q_network
        self.num_networks = len(q_network)
        self.preprocessor = preprocessor
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.b_target_fix = b_target_fix
        self.alpha = alpha
        self.episode_num = 0
        self.global_step = 0
        self.save_frequency = save_frequency
        self.network_type = network_type
        self.model_name = model_name

    def compile(self, optimizer, loss_func):
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
        for i in range(self.num_networks):
          self.q_network[i].compile(optimizer=optimizer, loss=loss_func)

        if (self.num_networks==1) and (self.b_target_fix == True):
          if self.model_name == 'dueling dqn':
            def getTiledAdvantage(a):
              num_actions = K.params['actions']
              a_mean = tf.reduce_mean(a, axis = 1)
              a_mean = K.expand_dims(a_mean, axis = 1)
              a_mean = tf.tile(a_mean, [1,num_actions])
              a_norm = tf.subtract(a,a_mean)
              return a_norm 


            def getTiledValue(v):
              num_actions = K.params['actions']
              v_tiled = tf.tile(v,[1,num_actions])
              return v_tiled

            input = Input(shape = (84,84,4), name = 'input')
            y = Conv2D(32, (8,8), strides=(4,4), activation='relu')(input)
            y = Conv2D(64, (4,4), strides=(2,2), activation='relu')(y)
            y = Conv2D(64, (3,3), strides=(1,1), activation='relu')(y)
            y = Flatten()(y)

            value_out = Dense(512, activation='relu')(y)
            value_out = Dense(1)(y)
            value_out = Lambda(getTiledValue)(value_out)

            adv_out = Dense(512, activation='relu')(y)
            adv_out = Dense(K.params['actions'])(y)
            adv_out = Lambda(getTiledAdvantage)(adv_out)


            out = Lambda(lambda outs: outs[0] + outs[1])
            net_output = out([value_out, adv_out])

            model_1 = Model(inputs = [input], outputs = [net_output])
            self.target_q_network = model_1

          else:
            config = self.q_network[0].get_config()
            self.target_q_network = Sequential.from_config(config)
            self.target_q_network.set_weights(self.q_network[0].get_weights())
            self.target_q_network.compile(optimizer=optimizer, loss=loss_func)

    def calc_q_values(self, state, model_idx, batch=None):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        if batch == None:
          batch = self.batch_size

        if model_idx == -1:
          q_state = self.target_q_network.predict(state, batch_size=batch)
        else:
          q_state = self.q_network[model_idx].predict(state, batch_size=batch)


        return q_state

    def select_action(self, state, **kwargs):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        # self.preprocessor is SequencePreprocessor. It calls Atari -> History, and returns
        # the latest sequence of processed frames.
        state_p = self.preprocessor.process_state_for_network(state) # state is uint8
        

        q_values = self.calc_q_values(state_p, 0, state_p.shape[0])
        if self.num_networks == 2:
          q_values += self.calc_q_values(state_p, 1, state_p.shape[0])

        action = self.policy.select_action(q_values, kwargs['stage'])
        return action

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

        batch_states,batch_actions,batch_rewards,batch_terminal = self.memory.sample(self.batch_size)


        window = batch_states.shape[-1]-1
        batch_curr_states = batch_states[..., 0:window]
        batch_next_states = batch_states[..., 1:window+1]

        batch_curr_states_proc = self.preprocessor.process_batch_for_network(batch_curr_states)
        batch_next_states_proc = self.preprocessor.process_batch_for_network(batch_next_states)


        if self.num_networks == 1:
          model_idx = 0
          if self.network_type == 'single':
            if self.b_target_fix == True:        # For DQN and linear with target fixing
              q_s_a_curr = self.calc_q_values(batch_curr_states_proc,0)
              q_s_a_next = self.calc_q_values(batch_next_states_proc,-1)
            else:     # For linear with no target fixing (also should be no replay memory)
              q_s_a_curr = self.calc_q_values(batch_curr_states_proc,0)
              q_s_a_next = self.calc_q_values(batch_next_states_proc,0)
          elif self.network_type == 'double':  # For DDQN
            q_s_a_curr = self.calc_q_values(batch_curr_states_proc,model_idx)
            q_s_a_next_source = self.calc_q_values(batch_next_states_proc,model_idx)
            q_s_a_next_target = self.calc_q_values(batch_next_states_proc,-1)
        elif self.num_networks == 2:   # For double linear:double Q learning
          model_idx = random.randint(0, 1)
          q_s_a_curr = self.calc_q_values(batch_curr_states_proc,model_idx)
          q_s_a_next = self.calc_q_values(batch_next_states_proc,1-model_idx)




        target = q_s_a_curr[:]
        for i in range(self.batch_size):
          if self.network_type == 'single' or self.model_name == 'double linear':
            target[i,batch_actions[i]] = batch_terminal[i].astype(float)*batch_rewards[i] + (1-batch_terminal[i].astype(float))*(batch_rewards[i] + self.gamma*np.max(q_s_a_next[i,:]))
          elif self.network_type == 'double':
            target[i,batch_actions[i]] = batch_terminal[i].astype(float)*batch_rewards[i] + (1-batch_terminal[i].astype(float))*(batch_rewards[i] + self.gamma*q_s_a_next_target[i, np.argmax(q_s_a_next_source[i,:])])



        loss = self.q_network[model_idx].train_on_batch(batch_curr_states_proc, target)

        if ((self.num_networks==1) and (self.b_target_fix == True) and (len(self.memory) > self.num_burn_in) and (self.global_step%self.target_update_freq == 0)): 
          weights = self.q_network[model_idx].get_weights()
          self.target_q_network.set_weights(weights)

        return loss

    def add_scalar_summary(self, name, value):
      summary = tf.Summary()
      summary_value = summary.value.add()
      summary_value.simple_value = value
      summary_value.tag = name

      return summary

    def save_model(self):
      folder = 'logs/'
      for the_file in os.listdir(folder):
          file_path = os.path.join(folder, the_file)
          try:
              if os.path.isfile(file_path) and file_path[-3:] == '.h5':
                  os.unlink(file_path)
          except Exception as e:
              print(e)

      filepath = 'logs/' + '_'.join([self.model_name, self.network_type, 'Step']) + str(self.episode_num) + '.h5'
      self.q_network[self.num_networks-1].save(filepath)

    def fit(self, env, num_iterations, max_episode_length=None):
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
        self.sess = K.get_session()
        self.writer = tf.summary.FileWriter('logs', self.sess.graph)

        optimizer = Adam(lr=self.alpha, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        loss_func = objectives.mean_huber_loss
        self.compile(optimizer, loss_func)

        while self.global_step < num_iterations:
          self.preprocessor.reset()
          curr_state = env.reset()
          # env.render()

          episode_reward = 0

          for j in range(max_episode_length):
            self.global_step += 1
            if len(self.memory) <= self.num_burn_in:
              action = self.select_action(curr_state, **{'stage': 'random'})
            else:
              action = self.select_action(curr_state, **{'stage': 'training'})
            
            if not (self.global_step % self.target_update_freq):
              print('Episode No = ', self.episode_num, 'Episode steps = ', j, 'Total Steps = ', self.global_step, 'Memory size = ', len(self.memory), 'Current eps = ', self.policy.decay_greedy_epsilon.epsilon)
            next_state, reward, is_terminal, debug_info = env.step(action)
            reward = self.preprocessor.process_reward(reward)
            # env.render()


            episode_reward += reward

            if self.b_target_fix == False and self.network_type == 'single':
              if is_terminal:
                break

              curr_state_proc = self.preprocessor.process_state_for_network(curr_state)
              next_state_proc = self.preprocessor.process_state_for_network(next_state)
              q_s_a_curr = self.calc_q_values(curr_state_proc,0)
              q_s_a_next = self.calc_q_values(next_state_proc,0)
              target = q_s_a_curr[:]
              target[0, action] = float(is_terminal)*reward+(1-float(is_terminal))*(reward + self.gamma*np.max(q_s_a_next[0,:]))
              loss = self.q_network[0].train_on_batch(curr_state_proc, target)

              with tf.name_scope('summaries'):
                pass
                # self.writer.add_summary(self.add_scalar_summary('training_loss', loss), self.global_step)

              if (not (self.global_step % self.save_frequency)):
                self.save_model()

            else:
              curr_state_proc_memory = self.preprocessor.process_state_for_memory(curr_state)
              self.memory.append(curr_state_proc_memory, action, reward, is_terminal)
              
              if is_terminal:
                self.memory.end_episode(curr_state_proc_memory, is_terminal=True)
                break
              if j == max_episode_length-1:
                self.memory.end_episode(curr_state_proc_memory, is_terminal=False)
              
              if (len(self.memory) > self.num_burn_in) and (self.global_step%self.train_freq == 0):
                loss = self.update_policy()

                with tf.name_scope('summaries'):
                 self.writer.add_summary(self.add_scalar_summary('training_loss', loss), self.global_step)
              
              # print('Loss = ', loss)
        
              if ((len(self.memory) > self.num_burn_in) and (self.global_step % self.save_frequency == 0)):
                self.save_model()
               
            
            curr_state = next_state
            if (not (self.global_step % (num_iterations/3)) or self.global_step == 1):
              self.evaluate(env, 20, max_episode_length, save_vid=True)
            elif (not self.global_step % (self.target_update_freq)):
              self.evaluate(env, 20, max_episode_length, save_vid=False)
         
          self.episode_num += 1
          self.writer.add_summary(self.add_scalar_summary('episode_reward', episode_reward), self.episode_num)


    def evaluate(self, env, num_episodes, max_episode_length=None, save_vid=False):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """

        print('Evaluating at global step {}'.format(self.global_step))
        if save_vid:
          print('SAVING FRAMES TO DISK')
          directory = '_'.join(['eval', 'frames', self.model_name, self.network_type]) + '/'
          subfolder = 'GStep{}'.format(str(self.global_step)) + '/'

          if not os.path.exists(directory):
            os.makedirs(directory)
          if not os.path.exists(directory+subfolder):
            os.makedirs(directory+subfolder)

          env = wrappers.Monitor(env, directory + subfolder, mode='evaluation', video_callable=lambda _: True, uid='_'.join([self.model_name, self.network_type]))

        else:
          print('NOT SAVING FRAMES TO DISK')

        total_steps = 0
        total_reward = 0
        for episode in range(num_episodes):
          self.preprocessor.reset()
          curr_state = env.reset()
          # env.render()

          episode_reward = 0
          for step in range(max_episode_length):
            total_steps += 1
            action = self.select_action(curr_state, **{'stage': 'testing'})
            next_state, reward, is_terminal, debug_info = env.step(action)

            # env.render()
            episode_reward += reward
            if is_terminal:
              break
            curr_state = next_state

          total_reward += episode_reward

        with tf.name_scope('summaries'):
          self.writer.add_summary(self.add_scalar_summary('Evaluate/Avg Episode Reward', total_reward/num_episodes), self.global_step)

        env.close()