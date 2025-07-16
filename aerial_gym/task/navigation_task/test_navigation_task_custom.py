from aerial_gym.task.navigation_task.navigation_task_custom import NavigationTaskCustom
from aerial_gym.config.task_config.navigation_task_config import task_config
import torch

if __name__ == "__main__":
    # Set up device and number of envs
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    num_envs = 2
    task_config.device = device_str
    task_config.num_envs = num_envs

    # Initialize environment
    env = NavigationTaskCustom(task_config)
    obs, reward, terminated, truncated, info = env.reset()
    print("Obs keys:", list(obs.keys()))
    for k, v in obs.items():
        print(f"{k}: {v.shape}, dtype={v.dtype}")
    print("Reward shape:", reward.shape)
    print("Terminated shape:", terminated.shape)
    print("Truncated shape:", truncated.shape)

    # Step with random actions
    for i in range(3):
        actions = torch.randn((num_envs, env.action_space.shape[0]), device=device)
        obs, reward, terminated, truncated, info = env.step(actions)
        print(f"Step {i+1}:")
        for k, v in obs.items():
            print(f"  {k}: {v.shape}, dtype={v.dtype}")
        print("  Reward:", reward)
        print("  Terminated:", terminated)
        print("  Truncated:", truncated)
    env.close() 