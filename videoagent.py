import numpy as np
import gym
from gym import spaces
import Streaming_ENV
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.results_plotter import load_results, plot_results
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch as th


bandwidths = [3, 5, 10, 15, 20]
qualities = [1, 2, 3, 4, 5, 6]
user_bandwidth = 20

chunk_important = np.array([1,1,2,3,4,5,6,5,4,3,2,1,1,1,1,1])

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-50, high=100, shape=(103,), dtype=np.float32)
        self.env = Streaming_ENV.VideoStreamingEnv(bandwidths, chunk_play_time=4)
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return np.array(observation, dtype=np.float32), reward, done, info
    def reset(self):
        observation = self.env.reset()
        return np.array(observation, dtype=np.float32)
    def close (self):
        print("close")

# Main 함수
if __name__ == '__main__':
    env = CustomEnv()
    env = Monitor(env)
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    eval_callback = EvalCallback(env, eval_freq=100, deterministic=True, render=False)
    model.learn(total_timesteps=1000000, callback=[eval_callback])
    model.save("dqn_v1")
    results = load_results("logs/")
    plot_results(results, title="My Training Results")