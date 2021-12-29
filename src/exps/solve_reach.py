import random

from definitions import ROOT_DIR
import metaworld
from stable_baselines3 import SAC
from wandb.integration.sb3 import WandbCallback
import wandb

wandb.init(project="my-test-project",
           entity="dotd",
           dir=f"{ROOT_DIR}/")
config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": -1,
    "env_name": "pick-place-v2"
}
run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


ml1 = metaworld.ML1(config["env_name"])  # Construct the benchmark, sampling tasks
env = ml1.train_classes[config["env_name"]]()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task
print(f"{env.max_path_length}")
config["total_timesteps"] = env.max_path_length

model = SAC(config["policy_type"],
            env,
            verbose=1,
            tensorboard_log=f"{ROOT_DIR}/runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"{ROOT_DIR}/models/{run.id}",
        verbose=2,
    ),
)
run.finish()
