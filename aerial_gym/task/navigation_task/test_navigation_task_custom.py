from aerial_gym.registry.task_registry import task_registry
import torch

if __name__ == "__main__":
    # Set up device and number of envs
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    num_envs = 2

    # Get the task config and update it
    task_config = task_registry.get_task_config("navigation_task_custom")
    task_config.device = device_str
    task_config.num_envs = num_envs

    # Initialize environment using task registry
    env = task_registry.make_task("navigation_task_custom", num_envs=num_envs)
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