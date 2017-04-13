"""Core classes."""

import numpy as np

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
    def __init__(self, state, action, reward, is_terminal):
        self.state = state
        self.action = action
        self.reward = reward
        self.is_terminal = is_terminal

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

    def __init__(self):
        raise NotImplementedError

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
        raise NotImplementedError
        # return state.astype('float32')        

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
        raise NotImplementedError
        # return state.astype('uint8')

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
        # return samples
        raise NotImplementedError

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
        raise NotImplementedError

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
    def __init__(self, max_size):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        #self.max_size = 1000 # for debugging
        self.max_size = max_size
        self.experience = [] # list of tuples
        self.index_for_insertion = 0

    def append(self, state, action, reward, is_terminal):
        # pseudo ring buffer. Keep appending stuff till it reaches max size. 
        # once, it has reach max size, we start to replace the oldest items. 
        
        new_sample = Sample(state, action, reward, is_terminal)

        if len(self.experience) < self.max_size:
            self.experience.append(new_sample)
        else: # Replay Memory already has max size
            if self.index_for_insertion==self.max_size:
                self.index_for_insertion = 0
            self.experience[self.index_for_insertion] = new_sample
        self.index_for_insertion+=1

    def end_episode(self):
        # make the is_terminal (last element of tuple) of the last inserted (SAR+is_terminal) sequence True
        self.experience[self.index_for_insertion-1].is_terminal = True

    def sample(self, batch_size):
        import random
        # sample 32 indices. but don't sample 0,1,2. 
        # TODO: if there is a terminal state in the middle of the sample, then resample
        # if len(self.experience) < 1000000:

        # indices = random.sample(range(len(self.experience))[4:], batch_size)
        # indices = np.random.randint(4,len(self.experience),batch_size)

        if self.index_for_insertion-1 < 4:
            index = 4
        else:
            index = self.index_for_insertion-1

        # else:
            # indices = random.sample(range(len(self.experience)), batch_size)
        # print "indices {}".format(indices)

        # list of list of samples. (32 outside and 4 inside)
        current_state_samples = self.experience[index-4:index]  
        next_state_samples = self.experience[index-3:index+1]
        #print index, current_state_samples
        # for index in indices:
        #     current_state_samples = [self.experience[index-4:index]]
        #     for (idx, each_list_of_samples) in enumerate(current_state_samples):
        #         print "sample 0 :: index ", index, each_list_of_samples[0].state.shape
        #         print "sample 1 :: index ", index, each_list_of_samples[1].state.shape
        #         print "sample 2 :: index ", index, each_list_of_samples[2].state.shape
        #         print "sample 3 :: index ", index, each_list_of_samples[3].state.shape

        return {
                    'current_state_samples':current_state_samples,
                    'next_state_samples':next_state_samples
                } 

    def clear(self):
        self.experience=[]
        self.index_for_insertion = 0
