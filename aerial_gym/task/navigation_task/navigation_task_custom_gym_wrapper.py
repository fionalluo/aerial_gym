import gymnasium as gym
import numpy as np
from aerial_gym.task.navigation_task.navigation_task_custom import NavigationTaskCustomBase
from aerial_gym.config.task_config.navigation_task_config import task_config

class NavigationTaskCustomEnv(gym.Env):
    """
    Gymnasium-compatible environment for the navigation task.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, task_config_override=None):
        # Use a copy of the config to avoid side effects
        cfg = task_config_override if task_config_override is not None else task_config
        # Force single environment for Gym compatibility
        cfg.num_envs = 1
        self.env = NavigationTaskCustomBase(cfg)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._last_obs = None

    def reset(self, *, seed=None, options=None):
        # Gymnasium API: returns obs, info
        obs, *_ , info = self.env.reset()
        # Convert obs dict of tensors to dict of numpy arrays
        obs_np = {k: v[0].cpu().numpy() for k, v in obs.items()}
        self._last_obs = obs_np
        return obs_np, info

    def step(self, action):
        # Accepts action as numpy array, converts to torch
        import torch
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        action = torch.from_numpy(action).float().unsqueeze(0).to(self.env.device)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_np = {k: v[0].cpu().numpy() for k, v in obs.items()}
        reward = float(reward[0].cpu().item())
        terminated = bool(terminated[0].cpu().item())
        truncated = bool(truncated[0].cpu().item())
        self._last_obs = obs_np
        return obs_np, reward, terminated, truncated, info

    def render(self, mode="human"):
        # Optionally implement rendering
        return self.env.render()

    def close(self):
        self.env.close()

# Gymnasium registration
try:
    import gymnasium as gym
    gym.register(
        id="NavigationTaskCustomEnv-v0",
        entry_point="aerial_gym.task.navigation_task.navigation_task_custom_gym_wrapper:NavigationTaskCustomEnv",
        max_episode_steps=1000,
    )
except Exception as e:
    # Registration may fail if already registered or if gymnasium is not available
    pass 