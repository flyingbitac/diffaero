import os

import hydra
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(
        self,
        cfg: DictConfig,
        type: str = 'tensorboard',
        run_name: str = ""
    ):
        assert type.lower() in ['tensorboard', 'wandb']
        self.cfg = cfg
        self.logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        if run_name != "":
            run_name = "__" + run_name
        env_name = {"position_control": "PC", "obstacle_avoidance": "OA"}[cfg.env.name]
        run_name = f"{env_name}__{cfg.algo.name}{run_name}__{cfg.seed}"
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
    
    def log_video(self, tag, video, step):
        if isinstance(self.writer, SummaryWriter):
            self.writer.add_video(tag, video, step)
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
            self.writer.add_text("Env HParams", to_yaml(self.cfg.env), 0)
            self.writer.add_text("Train HParams", to_yaml(self.cfg.algo), 0)
            overrides_path = os.path.join(self.logdir, ".hydra", "overrides.yaml")
            if os.path.exists(overrides_path):
                with open(overrides_path, "r") as f:
                    overrides = [line.strip('- ') for line in f.readlines()]
                    self.writer.add_text("Overrides", ' '.join(overrides), 0)
