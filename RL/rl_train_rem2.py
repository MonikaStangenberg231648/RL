import os
import argparse
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env  
from clearml import Task
import wandb
import gymnasium as gym
from wandb.integration.sb3 import WandbCallback



os.environ['WANDB_API_KEY'] = '81e90cda052e4ff4b2e6d490e7c614a9b48a3307' 

run = wandb.init(project="sb3_pendulum_demo", sync_tensorboard=True)

task = Task.init(project_name='Pendulum-v1/Monika Stangenberg', 
                    task_name='Experiment1')


task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")


parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for the optimizer")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per environment rollout")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs for policy updates")
parser.add_argument("--time_steps", type=int, default=100000, help="Number of timesteps per iteration")
parser.add_argument("--iterations", type=int, default=10, help="Number of iterations to train")
args = parser.parse_args()

env = OT2Env(render=False, num_agents=1)

model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    tensorboard_log=f"runs/{run.id}",
    device='cpu',  
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs
)

model.learn(
    total_timesteps=args.time_steps * args.iterations,
    callback=WandbCallback,            
    progress_bar=True,                  
    tb_log_name=f"runs/{run.id}"        
)

env.close()
wandb.finish()
print("Training complete")

