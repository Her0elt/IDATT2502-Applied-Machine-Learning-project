import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import gym

class MarioEnvironment(gym.Wrapper):
    def __init__(self, world='SuperMarioBros-v0'):
        self.env = gym_super_mario_bros.make(world)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)
        gym.Wrapper.__init__(self, self.env)
        self._current_score = 0
    
    def step(self, action):
        state, reward, done, info = self.env.step(action) 
        reward += (info['score'] - self._current_score) / 40.0
        self._current_score = info['score']
        if done:
            if info['flag_get']:
                reward += 350.0
            else:
                reward -= 50.0
        return state, reward / 10.0, done, info
