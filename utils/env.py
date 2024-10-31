import torch

class RecordEpisodeStatistics:
    def __init__(self, env):
        self.env = env
        self.n_envs = getattr(env, "n_envs", 1)
        self.device = env.device
        self.success = torch.zeros(self.n_envs, dtype=torch.float32, device=self.device)
        self.episode_length = torch.zeros(self.n_envs, dtype=torch.int, device=self.device)
        
    def __getattr__(self, name: str):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)
    
    def step(self, action):
        state, loss, terminated, extra = self.env.step(action)
        n_resets = extra["reset_indicies"].size(0)
        if n_resets > 0:
            self.success = torch.roll(self.success, -n_resets, 0)
            self.success[-n_resets:] = extra["success"][extra["reset"]]
            self.episode_length = torch.roll(self.episode_length, -n_resets, 0)
            self.episode_length[-n_resets:] = extra["l"][extra["reset"]]
        extra["stats"] = {
            "success_rate": self.success.mean().item(),
            "l": self.episode_length,
        }
        return state, loss, terminated, extra