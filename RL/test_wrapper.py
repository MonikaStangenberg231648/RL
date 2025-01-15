from ot2_gym_wrapper_wersja1 import OT2Env


if __name__ == "__main__":
    # Test the environment with random actions
    env = OT2Env(render=True, num_agents=1)

    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  
        obs, reward, done, _ = env.step(action)

        if done:
            print("Target reached!")
            break

    env.close()