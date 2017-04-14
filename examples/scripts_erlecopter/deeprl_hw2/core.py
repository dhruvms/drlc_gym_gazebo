"Core classes."""

import numpy as np
import time
import pdb

class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    pass


class Preprocessor:
    """Preprocessor base class.

    This is a suggested interface for the preprocessing steps. You may
    implement any of these functions. Feel free to add or change the
    interface to suit your needs.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def process_state_for_network(self, state):
        """Preprocess the given state before giving it to the network.

        Should be called just before the action is selected.

        This is a different method from the process_state_for_memory
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory is a lot more efficient thant float32, but the
        networks work better with floating point images.

        Parameters
        ----------
        state: np.ndarray
          Generally a numpy array. A single state from an environment.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in anyway.

        """
        return state

    def process_state_for_memory(self, state):
        """Preprocess the given state before giving it to the replay memory.

        Should be called just before appending this to the replay memory.

        This is a different method from the process_state_for_network
        because the replay memory may require a different storage
        format to reduce memory usage. For example, storing images as
        uint8 in memory and the network expecting images in floating
        point.

        Parameters
        ----------
        state: np.ndarray
          A single state from an environmnet. Generally a numpy array.

        Returns
        -------
        processed_state: np.ndarray
          Generally a numpy array. The state after processing. Can be
          modified in any manner.

        """
        return state

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass


class ReplayMemory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw saamples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, height, width, max_size, window_length):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.max_size = max_size
        self.window_length = window_length
        self.height = height
        self.width = width

        self.states = np.zeros((self.max_size, self.height, self.width), dtype='uint8')
        self.actions = np.zeros(self.max_size, dtype='int32')
        self.rewards = np.zeros(self.max_size, dtype='float32')
        self.terminal = np.zeros(self.max_size, dtype='bool')

        self.oldest = 0
        self.newest = 0
        self.size = 0

        self.rand_gen = np.random.RandomState()

        self.shape = [self.height, self.width, self.window_length]

    def __len__(self):
        return max(0, self.size - self.window_length + 1)

    def __iter__(self):
        raise NotImplementedError('This method should be overridden')

    def __getitem__(self, index):
        raise NotImplementedError('This method should be overridden')

    def append(self, state, action, reward, is_terminal):
        self.states[self.newest] = state
        self.actions[self.newest] = action
        self.rewards[self.newest] = reward
        self.terminal[self.newest] = is_terminal

        if self.size == self.max_size:
            self.oldest = (self.oldest + 1) % self.max_size
        else:
            self.size += 1
        self.newest = (self.newest + 1) % self.max_size

    # What does this even do?!
    def end_episode(self, final_state, is_terminal):
        if np.array_equal(self.states[self.newest-1], final_state):
            self.terminal[self.newest-1] = is_terminal
        else:
            print('Bad final state supplied.')

    def sample(self, batch_size, indexes=None):
        batch_states = np.zeros((batch_size, self.window_length + 1, self.height, self.width), dtype='uint8')
        batch_actions = np.zeros((batch_size, 1), dtype='int32')
        batch_rewards = np.zeros((batch_size, 1), dtype='float32')
        batch_terminal = np.zeros((batch_size, 1), dtype='bool')

        if not indexes:
            count = 0
            while count < batch_size:
                index = self.rand_gen.randint(self.oldest, self.oldest + self.size - self.window_length)
                all_frames = range(index, index + self.window_length + 1) # oldest to newest
                # all_frames.reverse() # newest to oldest

                if self.terminal.take(all_frames[-2], mode='wrap'): 
                    continue

                sample = self.states.take(all_frames, axis=0, mode='wrap')
                
                try:
                    final_terminal = np.max(np.where(self.terminal.take(all_frames[0:-1], mode='wrap') == True)[0])
                except ValueError:
                    final_terminal = None
                
                if final_terminal != None:
                    sample[0:final_terminal] = np.zeros((self.height, self.width))
                
                batch_states[count] = sample
                batch_actions[count] = self.actions.take(all_frames[-2], mode='wrap')
                batch_rewards[count] = self.rewards.take(all_frames[-2], mode='wrap')
                batch_terminal[count] = self.terminal.take(all_frames[-1], mode='wrap')
                
                count += 1

        # TODO: Check if this makes sens, and isn't too slow.
        else:
            for index in indexes:
                all_frames = range(index, index + self.window_length + 1) # oldest to newest

                if self.terminal.take(all_frames[-2], mode='wrap'): 
                    continue

                sample = self.states.take(all_frames, axis=0, mode='wrap')
                
                try:
                    final_terminal = np.where(self.terminal.take(all_frames[0:-2], mode='wrap') == True)[0][-1]
                except IndexError:
                    final_terminal = None

                if final_terminal != None:
                    sample[0:final_terminal+1] = np.zeros((self.height, self.width))

                batch_states[count] = self.states.take(sample, axis=0, mode='wrap')
                batch_actions[count] = self.actions.take(all_frames[-2], mode='wrap')
                batch_rewards[count] = self.rewards.take(all_frames[-2], mode='wrap')
                batch_terminal[count] = self.terminal.take(all_frames[-1], mode='wrap')
                
                count += 1

        # Do we need something like this?
        # try:
        #   assert len(batch_states) == len(batch_actions) == len(batch_rewards) == len(batch_terminal)
        # except:
        #   raise AssertionError('Incorrect sampling - lists are of different lengths.')

        batch_states = np.swapaxes(np.swapaxes(batch_states, 1, 3), 1, 2)

        return batch_states, batch_actions, batch_rewards, batch_terminal

    def clear(self):
        self.states = np.zeros((self.max_size, self.height, self.width), dtype='uint8')
        self.actions = np.zeros(self.max_size, dtype='int32')
        self.rewards = np.zeros(self.max_size, dtype='float32')
        self.terminal = np.zeros(self.max_size, dtype='bool')

        del self.states
        del self.actions
        del self.rewards
        del self.terminal

        self.oldest = 0
        self.newest = 0
        self.size = 0