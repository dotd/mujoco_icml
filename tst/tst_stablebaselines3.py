import gym

from stable_baselines3 import A2C

env = gym.make('CartPole-v1')

model = A2C('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(10):
    if i % 100 == 0:
        print(f"i={i}")
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # env.render()
    if done:
      obs = env.reset()