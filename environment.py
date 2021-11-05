import cv2
import gym
import gym_super_mario_bros
import numpy as np
from gym import spaces
from gym.wrappers import FrameStack
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from constants import ENV_HEIGHT, ENV_WIDTH, SKIP_AND_STACK_AMOUNT, WOLRD

cv2.ocl.setUseOpenCL(False)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=ENV_HEIGHT, height=ENV_WIDTH, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 3), dtype=np.uint8
            )

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = info["life"]
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, info = self.env.step(0)
            self.lives = info["life"]
        return obs


def create_mario_env():
    env = gym_super_mario_bros.make(WOLRD)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = SkipFrame(env, skip=SKIP_AND_STACK_AMOUNT)
    env = WarpFrame(env)
    env = EpisodicLifeEnv(env)
    env = FrameStack(env, SKIP_AND_STACK_AMOUNT)  # tre frames

    return env
