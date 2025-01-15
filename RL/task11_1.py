from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from clearml import Task
import gym
import os

task = Task.init(project_name='Pendulum-v1/YourName', task_name='Experiment1')

task.set_base_docker('deanis/2023y2b-rl:latest')

task.execute_remotely(queue_name="default")

env = gym.make('Pendulum-v1', g=9.81)

model = PPO(
    policy='MlpPolicy',
    env=env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
)

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback

    os.environ['WANDB_API_KEY'] = 'INSERT_API_KEY_HERE'
    run = wandb.init(project="pendulum_training", sync_tensorboard=True)

    wandb_callback = WandbCallback(
        model_save_freq=1000,
        model_save_path=f"models/{run.id}",
        verbose=2
    )
except ImportError:
    wandb_callback = None

time_steps = 1000  
iterations = 5  

for i in range(iterations):
    print(f"Starting iteration {i + 1} of {iterations}")
    model.learn(
        total_timesteps=time_steps,
        callback=wandb_callback,
        reset_num_timesteps=False,
        progress_bar=True
    )

    model.save(f"models/pendulum_model_{(i + 1) * time_steps}")

print("Training complete. Model saved.")
