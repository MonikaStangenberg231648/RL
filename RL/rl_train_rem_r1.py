import os
import argparse
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env
from clearml import Task
import wandb
from wandb.integration.sb3 import WandbCallback

# OT2Env already contains the reward logic
class OT2EnvWithReward(OT2Env):
    def __init__(self, render=True, num_agents=1):
        super().__init__(render, num_agents)

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # Custom reward logic
        distance_to_target = np.linalg.norm(self.robot_position - self.target_position)
        action_penalty = -0.1 * np.sum(np.square(action[:3]))  # Penalize large actions
        reward = -distance_to_target + action_penalty
        
        return obs, reward, done, info


# Setup WandB
os.environ['WANDB_API_KEY'] = '81e90cda052e4ff4b2e6d490e7c614a9b48a3307'
run = wandb.init(project="sb3_ot2_demo", sync_tensorboard=True)

# Initialize ClearML Task
task = Task.init(project_name='OT2_Training', task_name='Experiment_with_reward')
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate for the optimizer")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
parser.add_argument("--n_steps", type=int, default=2048, help="Number of steps per environment rollout")
parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs for policy updates")
parser.add_argument("--time_steps", type=int, default=100000, help="Number of timesteps per iteration")
parser.add_argument("--iterations", type=int, default=10, help="Number of iterations to train")
args = parser.parse_args()

# Use the custom environment
env = OT2EnvWithReward(render=False, num_agents=1)

# Initialize PPO model
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

# Setup WandB callback
wandb_callback = WandbCallback(
    model_save_freq=10000,
    model_save_path=f"models/{run.id}",
    verbose=2,
    log_reward=True
)

# Train the model
model.learn(
    total_timesteps=args.time_steps * args.iterations,
    callback=wandb_callback,
    progress_bar=True,
    tb_log_name=f"runs/{run.id}"
)

# Save the trained model
model.save("ppo_ot2_model")

# Close environment and finalize WandB
env.close()
wandb.finish()
print("Training complete")


