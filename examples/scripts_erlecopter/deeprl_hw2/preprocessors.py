"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor

import pdb


class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of previous states to prepend to state being processed.

    """

    def __init__(self, height, width, history_length=1):
        self.history_length = history_length
        self.height = height
        self.width = width

        self.history = np.zeros((self.history_length, self.height, self.width), dtype='float32')

        # self.oldest = 0
        # self.newest = 0
        # self.size = 0

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""
        # Both state and self.history[self.newest] are (84,84) float32 arrays, scaled between 0 and 1
        # self.history[self.newest] = state.astype('float32')

        # state = np.roll(self.history, self.newest-1, axis=0)[::-1]
        # state = np.swapaxes(np.swapaxes(state, 0, 2), 0, 1)
        # state = state[np.newaxis, ...]
        # # state is now (1,84,84,4) float32, with the newest->oldest frames at positions 0->history_length

        # if self.size == self.history_length:
        #     self.oldest = (self.oldest + 1) % self.history_length
        # else:
        #     self.size += 1
        # self.newest = (self.newest + 1) % self.history_length

        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = state
        state = np.swapaxes(np.swapaxes(self.history, 0, 2), 0, 1)
        state = state[np.newaxis, ...]
        # state is now (1,84,84,4) float32, with the oldest->newest frames at positions 0->history_length

        return state

    def reset(self):
        """Reset the history sequence.

        Useful when you start a new episode.
        """
        self.history = np.zeros((self.history_length, self.height, self.width), dtype='float32')

        self.oldest = 0
        self.newest = 0
        self.size = 0

    def get_config(self):
        return {'history_length': self.history_length, 'img_height': self.height, 'img_width': self.width,\
                    'size': self.size, 'newest': self.newest}


class AtariPreprocessor(Preprocessor):
    """Converts images to greyscale and downscales.

    Based on the preprocessing step described in:

    @article{mnih15_human_level_contr_throug_deep_reinf_learn,
    author =	 {Volodymyr Mnih and Koray Kavukcuoglu and David
                  Silver and Andrei A. Rusu and Joel Veness and Marc
                  G. Bellemare and Alex Graves and Martin Riedmiller
                  and Andreas K. Fidjeland and Georg Ostrovski and
                  Stig Petersen and Charles Beattie and Amir Sadik and
                  Ioannis Antonoglou and Helen King and Dharshan
                  Kumaran and Daan Wierstra and Shane Legg and Demis
                  Hassabis},
    title =	 {Human-Level Control Through Deep Reinforcement
                  Learning},
    journal =	 {Nature},
    volume =	 518,
    number =	 7540,
    pages =	 {529-533},
    year =	 2015,
    doi =        {10.1038/nature14236},
    url =	 {http://dx.doi.org/10.1038/nature14236},
    }

    You may also want to max over frames to remove flickering. Some
    games require this (based on animations and the limited sprite
    drawing capabilities of the original Atari).

    Parameters
    ----------
    new_size: 2 element tuple
      The size that each image in the state should be scaled to. e.g
      (84, 84) will make each image in the output have shape (84, 84).
    """

    def __init__(self, new_size, window_length):
        self.final_size = new_size
        self.window_length = window_length

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        state = np.array(Image.fromarray(state).resize(self.final_size).convert('L')).astype('uint8')
        return state

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images. <- NO.
        """
        # proc_state = np.zeros((self.window_length, state.shape[0], state.shape[1]), dtype='float32')
        # for i in range(self.window_length):
        #     old_frame = state[self.frame_skip*i]
        #     new_frame = state[self.frame_skip*i + 1]

        #     proc_state[i] = np.maximum(old_frame, new_frame)

        # return np.swapaxes(np.swapaxes(proc_state, 0, 2), 0, 1)
        state = (self.process_state_for_memory(state).astype('float32') / 255.0)
        return state

    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        samples = (samples.astype('float32') / 255.0)
        return samples

    def process_reward(self, reward):
        """Clip reward between -1 and 1."""
        return np.clip(reward, -1, 1)


class PreprocessorSequence(Preprocessor):
    """You may find it useful to stack multiple prepcrocesosrs (such as the History and the AtariPreprocessor).

    You can easily do this by just having a class that calls each preprocessor in succession.

    For example, if you call the process_state_for_network and you
    have a sequence of AtariPreproccessor followed by
    HistoryPreprocessor. This this class could implement a
    process_state_for_network that does something like the following:

    state = atari.process_state_for_network(state)
    return history.process_state_for_network(state)
    """
    def __init__(self, height, width, new_size, window_length):
        History = HistoryPreprocessor(height, width, window_length)
        Atari = AtariPreprocessor(new_size, window_length)
        self.preprocessors = {'history': History, 'atari': Atari}

    def process_state_for_network(self, state):
        state = self.preprocessors['atari'].process_state_for_network(state)
        return self.preprocessors['history'].process_state_for_network(state)

    def process_state_for_memory(self, state):
        return self.preprocessors['atari'].process_state_for_memory(state)

    def process_batch_for_network(self, batch):
        batch = self.preprocessors['atari'].process_batch(batch)
        return batch

    def process_reward(self, reward):
        return self.preprocessors['atari'].process_reward(reward)

    def reset_history(self):
        self.preprocessors['history'].reset()
