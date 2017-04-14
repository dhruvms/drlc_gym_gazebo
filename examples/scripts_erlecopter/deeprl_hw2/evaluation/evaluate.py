import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Lambda, Conv2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.models import (Sequential, load_model)
import gym, pdb

# from ..deeprl_hw2.policy import Policy_Set

# from ..deeprl_hw2.objectives import mean_huber_loss

# from ..deeprl_hw2.preprocessors import PreprocessorSequence

from policy import Policy_Set

from objectives import mean_huber_loss

from preprocessors import PreprocessorSequence

import argparse
import os
import random

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


def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
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
    if model_name=='linear':
        model_1 = Sequential(name=model_name)
        # with tf.name_scope('linear'):
        # model_1.add(Dense(num_actions, input_shape=(np.prod(input_shape)*window,)))
        model_1.add(Flatten(input_shape=input_shape+(window,)))
        model_1.add(Dense(num_actions))
        model = [model_1]

    elif model_name == 'double linear':
        model_1 = Sequential()
        model_1.add(Flatten(input_shape=input_shape+(window,)))
        model_1.add(Dense(num_actions, input_shape=input_shape+(window,)))

        # model = [model_1]        
        model_2 = Sequential()
        model_2.add(Flatten(input_shape=input_shape+(window,)))
        model_2.add(Dense(num_actions, input_shape=input_shape+(window,)))
        model = [model_1,model_2]

       

    elif model_name == 'dqn':
        # with tf.name_scope('dqn'):
        model_1 = Sequential(name=model_name)
        model_1.add(Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=input_shape+(window,)))
        model_1.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))
        model_1.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
        model_1.add(Flatten())
        model_1.add(Dense(512, activation ='relu'))
        model_1.add(Dense(num_actions))
        print(model_1.summary())
        model = [model_1]

    elif model_name == 'double dqn':
        # with tf.name_scope('double dqn'):
        model_1 = Sequential(name=model_name)
        model_1.add(Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=input_shape+(window,)))
        model_1.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))
        model_1.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
        model_1.add(Flatten())
        model_1.add(Dense(512, activation ='relu'))
        model_1.add(Dense(num_actions))

        print(model_1.summary())

        model = [model_1]
        
        # model_2 = Sequential(name=model_name)
        # model_2.add(Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=input_shape+(window,)))
        # model_2.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))
        # model_2.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
        # model_2.add(Flatten())
        # model_2.add(Dense(512, activation ='relu'))
        # model_2.add(Dense(num_actions))
        
        # print(model_2.summary())
        
        # model = [model_1, model_2]

    elif model_name == 'dueling dqn':
        input = Input(shape = input_shape+(window,), name = 'input')
        y = Conv2D(32, (8,8), strides=(4,4), activation='relu')(input)
        y = Conv2D(64, (4,4), strides=(2,2), activation='relu')(y)
        y = Conv2D(64, (3,3), strides=(1,1), activation='relu')(y)
        y = Flatten()(y)

        value_out = Dense(512, activation='relu')(y)
        value_out = Dense(1)(y)
        value_out = Lambda(getTiledValue)(value_out)

        adv_out = Dense(512, activation='relu')(y)
        adv_out = Dense(num_actions)(y)
        adv_out = Lambda(getTiledAdvantage)(adv_out)

        # net_output = add([value_out, adv_out])
        out = Lambda(lambda outs: outs[0] + outs[1])
        net_output = out([value_out, adv_out])

        model_1 = Model(inputs = [input], outputs = [net_output])
        model = [model_1]
    else:
        pass


    return model




def main():
	parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
	parser.add_argument('--env', default='Enduro-v0', help='Atari env name')
	parser.add_argument('-o', '--output', default='atari-v0', help='Directory to save data to')

	parser.add_argument('--window', default=4, type=int, help='Number of frames to stack')
	parser.add_argument('--input_shape', nargs=2, action='append', type=int, help='Scaled input image size')
	# parser.add_argument('--history_length', default=4, type=int, help='Number of frames to stack')
	parser.add_argument('--memory_limit', default=1000000, type=int, help='Replay memory size')
	parser.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
	parser.add_argument('--target_update_freq', default=10000, type=int, help='Steps before target network update')
	parser.add_argument('--num_burn_in', default=50000, type=int, help='Random samples to populate memory')
	parser.add_argument('--train_freq', default=4, type=int, help='Iterations before policy update')
	parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
	parser.add_argument('--b_target_fix', dest='b_target_fix', action='store_true')
	parser.add_argument('--no_b_target_fix', dest='b_target_fix', action='store_false')
	parser.set_defaults(b_target_fix=True)
	parser.add_argument('--start_value', default=1.0, type=float, help='Initial exploration probability')
	parser.add_argument('--end_value', default=0.10, type=float, help='Final exploration probability')
	parser.add_argument('--num_steps', default=1000000, type=int, help='Steps for exploration decay')
	parser.add_argument('--epsilon', default=0.05, type=float, help='Exploration probability')
	parser.add_argument('--num_iterations', default=5000000, type=int, help='Number of sampled interactions with environmemt')
	parser.add_argument('--max_episode_length', default=250000, type=int, help='Maximum steps in an episode')
	parser.add_argument('--alpha', default=0.0001, type=float, help='Learning rate')

	args = parser.parse_args()

	args.input_shape = tuple(args.input_shape[0])
    # args.output = get_output_folder(args.output, args.env)

	height, width = args.input_shape[0], args.input_shape[1]



	models = ['linear.h5','linearOnline.h5','doubleLinear.h5','DQN.h5','doubleDQN.h5','duelingDQN.h5']
	model_type = ['linear', 'linear','double linear','dqn','double dqn','dueling dqn']


	# models = ['duelingDQN.h5']
	# model_type = ['dueling dqn']

	episodeNum = 100
	episodeLength = 250000

	batch_size = 32
	env_name = ['SpaceInvaders-v0','SpaceInvaders-v0','SpaceInvaders-v0','SpaceInvaders-v0','SpaceInvaders-v0','Enduro-v0']
	# env_name = ['Enduro-v0']

	mean_reward = np.zeros(len(models))
	std_reward = np.zeros(len(models))

	preprocessor = PreprocessorSequence(height, width, args.input_shape, args.window)
	

	for i in  [1]:#range(len(models)):
		print('Model name: '+ models[i])
		print('Env name: '+ env_name[i])
		episode_reward = np.zeros(episodeNum)
		env = gym.make(env_name[i])
		params = {'actions':env.action_space.n}
		setattr(K, 'params', params)
		policy = Policy_Set(env.action_space.n, args.start_value, args.end_value, args.num_steps, args.epsilon)

		# model = load_model(models[i], custom_objects={'mean_huber_loss': mean_huber_loss, 'getTiledAdvantage': getTiledAdvantage, 'getTiledValue':getTiledValue})
		
		model = create_model(args.window, args.input_shape, env.action_space.n, model_name=model_type[i])[0]
		model.load_weights(models[i])

		preprocessor.reset()
		curr_state = env.reset()

		for j in range(episodeNum):
			print('Episode number: {}'.format(j))
			preprocessor.reset()
			curr_state = env.reset()

			for k in range(episodeLength):
				
				state_p = preprocessor.process_state_for_network(curr_state)
				q_values = model.predict(state_p, batch_size = 32)
				action = policy.select_action(q_values, **{'stage': 'testing'})
				next_state, reward, is_terminal, debug_info = env.step(action)
				# env.render()
				episode_reward[j] += reward

				if is_terminal:
					break
				else:
					curr_state = next_state


		mean_reward[i] = np.mean(episode_reward)
		std_reward[i] = np.std(episode_reward)
		print('Model: {} | Mean reward: {}| Standard deviation reward: {}|'.format(model,mean_reward[i],std_reward[i]))
		pdb.set_trace()
	
	pdb.set_trace()


	
if 	__name__ == '__main__':
	main()

















