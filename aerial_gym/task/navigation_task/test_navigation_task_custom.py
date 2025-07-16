from aerial_gym.task.navigation_task.navigation_task_custom_gym_wrapper import NavigationTaskCustomEnv
import numpy as np

if __name__ == "__main__":
    # Initialize Gym wrapper
    env = NavigationTaskCustomEnv()
    obs, info = env.reset()
    print("Obs keys:", list(obs.keys()))
    for k, v in obs.items():
        print(f"{k}: {np.array(v).shape}, dtype={np.array(v).dtype}")

    # Step with random actions
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}:")
        for k, v in obs.items():
            print(f"  {k}: {np.array(v).shape}, dtype={np.array(v).dtype}")
        print("  Reward:", reward)
        print("  Terminated:", terminated)
        print("  Truncated:", truncated)
    env.close() 