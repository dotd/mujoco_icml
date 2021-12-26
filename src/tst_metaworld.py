import metaworld
import random

print("\n".join([str(x) + y for (x,y) in enumerate(metaworld.ML1.ENV_NAMES)]))

ml1 = metaworld.ML1('pick-place-v2') # Construct the benchmark, sampling tasks

env = ml1.train_classes['pick-place-v2']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

obs = env.reset()  # Reset environment
a = env.action_space.sample()  # Sample an action
obs, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
print(f"obs={obs}\nreward={reward}\ndone={done}\ninfo={info}")
