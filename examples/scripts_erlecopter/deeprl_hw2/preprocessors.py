"""Suggested Preprocessors."""

import numpy as np
from PIL import Image

from deeprl_hw2 import utils
from deeprl_hw2.core import Preprocessor

class HistoryPreprocessor(Preprocessor):
    """Keeps the last k states.

    Useful for domains where you need velocities, but the state
    contains only positions.

    When the environment starts, this will just fill the initial
    sequence values with zeros k times.

    Parameters
    ----------
    history_length: int
      Number of states to feed to CNN (4 acc to paper).

    """
    def __init__(self, history_length=4):
        self.history_length = history_length
        self.history = np.zeros([history_length, 84, 84, 1], dtype='float32')

    def process_state_for_network(self, state):
        """You only want history when you're deciding the current action to take."""
        self.history = np.roll(self.history, -1, axis=0) # makes something like [0,1,2,3] -> [1,2,3,0]
        self.history[-1] = state[..., np.newaxis] # replaces last item by 4. => [1,2,3,4]
        return self.history

    def reset(self):    
        """Reset the history sequence.4
        Useful when you start a new episode.
        """
        self.history = np.zeros([self.history_length, 84, 84, 1], dtype='float32')

    def get_config(self):
        return self.history_length

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

    def __init__(self):
        pass

    def process_state_for_memory(self, state):
        """Scale, convert to greyscale and store as uint8.

        We don't want to save floating point numbers in the replay
        memory. We get the same resolution as uint8, but use a quarter
        to an eigth of the bytes (depending on float32 or float64)

        We recommend using the Python Image Library (PIL) to do the
        image conversions.
        """
        # if not (state.shape == (84,84)):
            # raise ValueError('AtariPreprocessor.process_state_for_memory : input state is not 84*84')
        state_gray = Image.fromarray(state).convert('L')
        state_gray = state_gray.resize((110,84)).crop((0, 0, 84, 84)) # section 4.1
        #state_gray = state_gray.crop((0,26,84,110)) #crops looking at the bottom of the image, not at the score
        return np.uint8(np.asarray(state_gray))

    def process_state_for_network(self, state):
        """Scale, convert to greyscale and store as float32.

        Basically same as process state for memory, but this time
        outputs float32 images.
        """
        state_gray = self.process_state_for_memory(state)
        return np.float32(state_gray)/255.

    # todo check 
    def process_batch(self, samples):
        """The batches from replay memory will be uint8, convert to float32.

        Same as process_state_for_network but works on a batch of
        samples from the replay memory. Meaning you need to convert
        both state and next state values.
        """
        return np.float32(samples)/255.

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
    def __init__(self):
        self.history_preprocessor = HistoryPreprocessor()
        self.atari_preprocessor = AtariPreprocessor()

    def process_state_for_network(self, state):
        state = self.atari_preprocessor.process_state_for_network(state)
        return self.history_preprocessor.process_state_for_network(state)

    def process_state_for_memory(self, state):
        return self.atari_preprocessor.process_state_for_memory(state)

    def process_batch(self, batch):
        return self.atari_preprocessor.process_batch(batch)

    def process_reward(self, reward):
        return self.atari_preprocessor.process_reward(reward)

    def reset_history_memory(self):
        self.history_preprocessor.reset()
