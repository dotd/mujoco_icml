import gym
import numpy as np

from stable_baselines3 import PPO

from stable_baselines3.ppo.policies import MlpPolicy

env = gym.make('CartPole-v1')

model = PPO(MlpPolicy, env, verbose=0)

