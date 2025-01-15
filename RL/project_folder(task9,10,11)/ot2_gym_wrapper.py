import gym
from gym import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=True, num_agents=1):
        super(OT2Env, self).__init__()
        
        # initializing simulation
        self.sim = Simulation(num_agents=num_agents, render=render)
        self.workspace_limits = {
            "x": [-0.26, 0.18],
            "y": [-0.26, 0.13],
            "z": [0.08, 0.20]
        }

        # defining action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0]), 
            high=np.array([1.0, 1.0, 1.0, 1]), 
            dtype=np.float32
        )

        # defining observation space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(6,), 
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        # randomly generating new target position
        self.target_position = np.random.uniform(
            [self.workspace_limits['x'][0], self.workspace_limits['y'][0], self.workspace_limits['z'][0]],
            [self.workspace_limits['x'][1], self.workspace_limits['y'][1], self.workspace_limits['z'][1]]
        )

        # setting the initial pipette position
        self.sim.set_start_position(0.0, 0.0, 0.1)
        self.robot_position = np.array([0.0, 0.0, 0.1])

        # returning the initial observation 
        return np.concatenate((self.robot_position, self.target_position))

    def step(self, action):
        velocity_x, velocity_y, velocity_z, _ = action
        print(f"Action received: {action}")

        self.robot_position += np.array([velocity_x, velocity_y, velocity_z])
        print(f"Updated pipette position: {self.robot_position}")

        self.robot_position = np.clip(
            self.robot_position,
            [self.workspace_limits["x"][0], self.workspace_limits["y"][0], self.workspace_limits["z"][0]],
            [self.workspace_limits["x"][1], self.workspace_limits["y"][1], self.workspace_limits["z"][1]]
        )
        print(f"Clipped pipette position: {self.robot_position}")

        distance_to_target = np.linalg.norm(self.robot_position - self.target_position)
        print(f"Distance to target: {distance_to_target}")

        reward = -distance_to_target

        done = distance_to_target < 0.01

        observation = np.concatenate([self.robot_position, self.target_position])
        return observation, reward, done, {}
    
    # rendering the environment 
    def render(self, mode='human'):
        if mode == 'human':
            pass
        elif mode == 'rgb_array':
            return self.sim.current_frame
        
    # closing the simulation
    def close(self):
        self.sim.close()