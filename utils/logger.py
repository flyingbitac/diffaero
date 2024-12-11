import os
import copy

import hydra
from omegaconf import OmegaConf, DictConfig
import torch
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(
        self,
        cfg: DictConfig,
        type: str = 'tensorboard',
        run_name: str = ""
    ):
        assert type.lower() in ['tensorboard', 'wandb']
        self.cfg = copy.deepcopy(cfg)
        self.logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        if run_name != "":
            run_name = "__" + run_name
        env_name = {"position_control": "PC", "obstacle_avoidance": "OA"}[cfg.env.name]
        run_name = f"{cfg.dynamics.name}__{env_name}__{cfg.algo.name}__{cfg.network.name}{run_name}__{cfg.seed}"
        if type.lower() == 'tensorboard':
            print("Using Tensorboard Logger.")
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.logdir, run_name)
            )
            self.log_hparams()
        else:
            print("Using W&B Logger.")
            assert "wandb" in list(dict(cfg).keys())
            import wandb
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                dir=self.logdir,
                sync_tensorboard=True,
                config=dict(cfg),
                name=run_name
            )
            self.writer = wandb
        print(f"Output directory  : {self.logdir}")
    
    def log_scalar(self, tag, value, step):
        if isinstance(self.writer, SummaryWriter):
            self.writer.add_scalar(tag, value, step)
        else:
            self.writer.log({tag: value}, step=step)
    
    def log_scalars(self, value_dict, step):
        for k, v in value_dict.items():
            if isinstance(v, dict):
                self.log_scalars({k+"/"+k_: v_ for k_, v_ in v.items()}, step)
            else:
                self.log_scalar(k, v, step)
    
    def log_histogram(self, tag, values, step):
        if isinstance(self.writer, SummaryWriter):
            self.writer.add_histogram(tag, values, step)
        else:
            self.writer.log({tag: values}, step=step)
    
    def log_image(self, tag, img, step):
        if isinstance(self.writer, SummaryWriter):
            self.writer.add_image(tag, img, step, dataformats='CHW')
        else:
            self.writer.log({tag: img}, step=step)
    
    def log_video(self, tag, video, step, fps):
        if isinstance(self.writer, SummaryWriter):
            self.writer.add_video(tag, video, step, fps=fps)
        else:
            self.writer.log({tag: video}, step=step)
            
    def close(self):
        if isinstance(self.writer, SummaryWriter):
            self.writer.flush()
            self.writer.close()
        else:
            self.writer.finish()

    def log_hparams(self):
        if isinstance(self.writer, SummaryWriter):
            to_yaml = lambda x: OmegaConf.to_yaml(x, resolve=True).replace("  ", "- ").replace("\n", "  \n")
            if hasattr(self.cfg.env, "render"):
                delattr(self.cfg.env, "render")
            self.writer.add_text("Env HParams", to_yaml(self.cfg.env), 0)
            self.writer.add_text("Train HParams", to_yaml(self.cfg.algo), 0)
            overrides_path = os.path.join(self.logdir, ".hydra", "overrides.yaml")
            if os.path.exists(overrides_path):
                with open(overrides_path, "r") as f:
                    overrides = [line.strip('- ') for line in f.readlines()]
                    self.writer.add_text("Overrides", ' '.join(overrides), 0)

class RecordEpisodeStatistics:
    def __init__(self, env):
        self.env = env
        self.n_envs = getattr(env, "n_envs", 1)
        self.device = env.device
        self.success = torch.zeros(self.n_envs, dtype=torch.float, device=self.device)
        self.arrive_time = torch.full((self.n_envs,), env.max_steps*env.dt, dtype=torch.float, device=self.device)
        self.episode_length = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
        
    def __getattr__(self, name: str):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)
    
    def step(self, action):
        state, loss, terminated, extra = self.env.step(action)
        n_resets = extra["reset_indicies"].size(0)
        if n_resets > 0:
            n_success = int(extra["success"].sum().item())
            if n_success > 0:
                self.arrive_time = torch.roll(self.arrive_time, -n_success, 0)
                self.arrive_time[-n_success:] = extra["arrive_time"][extra["success"]]
            self.success = torch.roll(self.success, -n_resets, 0)
            self.success[-n_resets:] = extra["success"][extra["reset"]]
            self.episode_length = torch.roll(self.episode_length, -n_resets, 0)
            self.episode_length[-n_resets:] = extra["l"][extra["reset"]]
        extra["stats"] = {
            "success_rate": self.success.mean().item(),
            "l": self.episode_length.float().mean().item(),
            "arrive_time": self.arrive_time.mean().item()
        }
        return state, loss, terminated, extra