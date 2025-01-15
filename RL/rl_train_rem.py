import os
import argparse
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env  
from clearml import Task

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

model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs)

time_steps = args.time_steps  
iterations = args.iterations 

model_dir = "models/ot2_model"
os.makedirs(model_dir, exist_ok=True)

for i in range(iterations):
    print(f"Starting iteration {i + 1}/{iterations}")
    model.learn(total_timesteps=time_steps, progress_bar=True, reset_num_timesteps=False)
    
    model.save(f"{model_dir}/ot2_model_{time_steps * (i + 1)}")
    print(f"Model saved after {time_steps * (i + 1)} timesteps")

env.close()
print("Training complete")
