from typing import Union
import os
import copy
import inspect
import logging
from pathlib import Path
from time import time

import hydra
from omegaconf import OmegaConf, DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from quaddif import QUADDIF_ROOT_DIR

class TensorBoardLogger:
    def __init__(
        self,
        cfg: DictConfig,
        logdir: str,
        run_name: str = ""
    ):
        self.cfg = cfg
        self.logdir = logdir
        Logger.info("Using Tensorboard Logger.")
        self.writer = SummaryWriter(log_dir=os.path.join(self.logdir, run_name))
        self.log_hparams()
    
    def log_scalar(self, tag, value, step):
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, value_dict, step):
        for k, v in value_dict.items():
            if isinstance(v, dict):
                self.log_scalars({k+"/"+k_: v_ for k_, v_ in v.items()}, step)
            else:
                self.log_scalar(k, v, step)
    
    def log_histogram(self, tag, values, step):
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag, img, step):
        self.writer.add_image(tag, img, step, dataformats='CHW')
    
    def log_images(self, tag, img, step):
        self.writer.add_images(tag, img, step)
            
    def log_video(self, tag, video, step, fps):
        self.writer.add_video(tag, video, step, fps=fps)
            
    def close(self):
        self.writer.close()

    def log_hparams(self):
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


class WandBLogger:
    def __init__(
        self,
        cfg: DictConfig,
        logdir: str,
        run_name: str = ""
    ):
        self.cfg = cfg
        self.logdir = logdir
        Logger.info("Using WandB Logger.")
        
        overrides_path = os.path.join(self.logdir, ".hydra", "overrides.yaml")
        if os.path.exists(overrides_path):
            with open(overrides_path, "r") as f:
                overrides = " ".join([line.strip('- ') for line in f.readlines()])
        import wandb
        wandb.init(
            project=cfg.logger.project,
            entity=cfg.logger.entity,
            dir=self.logdir,
            sync_tensorboard=False,
            config={**dict(cfg), "overrides": overrides},
            name=run_name
        )
        self.writer = wandb
    
    def log_scalar(self, tag, value, step):
        self.writer.log({tag: value}, step=step)
    
    def log_scalars(self, value_dict, step):
        for k, v in value_dict.items():
            if isinstance(v, dict):
                self.log_scalars({k+"/"+k_: v_ for k_, v_ in v.items()}, step)
            else:
                self.log_scalar(k, v, step)
    
    def log_histogram(self, tag, values, step):
        self.writer.log({tag: values}, step=step)
    
    def log_image(self, tag, img, step):
        self.writer.log({tag: img}, step=step)
    
    def log_images(self, tag, img, step):
        self.writer.log({tag: img}, step=step)
            
    def log_video(self, tag, video, step, fps):
        self.writer.log({tag: video}, step=step)
            
    def close(self):
        self.writer.finish()

class Logger:
    logging = logging.getLogger()
    def __init__(
        self,
        cfg: DictConfig,
        run_name: str = ""
    ):
        logger_alias = {
            "tensorboard": TensorBoardLogger,
            "wandb": WandBLogger
        }
        self.cfg = copy.deepcopy(cfg)
        assert str(cfg.log_level).upper() in logging._nameToLevel.keys()
        Logger.logging.setLevel(logging._nameToLevel[str(cfg.log_level).upper()])
        self.logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        run_name = f"{cfg.dynamics.name}__{cfg.env.name}__{cfg.algo.name}__{cfg.network.name}{run_name}__{cfg.seed}"
        type = cfg.logger.name.lower()
        self._logger: Union[TensorBoardLogger, WandBLogger] = logger_alias[type](self.cfg, self.logdir, run_name)
        Logger.info(f"Output directory  : {self.logdir}")
    
    @staticmethod
    def _get_logger(inspect_stack):
        rel_path = Path(inspect_stack[1].filename).resolve().relative_to(QUADDIF_ROOT_DIR)
        Logger.logging.name = str(rel_path)
        return Logger.logging
    
    @staticmethod
    def debug(msg):
        with tqdm.external_write_mode():
            Logger._get_logger(inspect.stack()).debug(str(msg))

    @staticmethod
    def info(msg):
        with tqdm.external_write_mode():
            Logger._get_logger(inspect.stack()).info(str(msg))

    @staticmethod
    def warning(msg):
        with tqdm.external_write_mode():
            Logger._get_logger(inspect.stack()).warning(str(msg))

    @staticmethod
    def error(msg):
        with tqdm.external_write_mode():
            Logger._get_logger(inspect.stack()).error(str(msg))

    @staticmethod
    def critical(msg):
        with tqdm.external_write_mode():
            Logger._get_logger(inspect.stack()).critical(str(msg))

    def log_scalar(self, tag, value, step):
        return self._logger.log_scalar(tag, value, step)
    
    def log_scalars(self, value_dict, step):
        return self._logger.log_scalars(value_dict, step)
    
    def log_histogram(self, tag, values, step):
        return self._logger.log_histogram(tag, values, step)
    
    def log_image(self, tag, img, step):
        return self._logger.log_image(tag, img, step)
    
    def log_images(self, tag, img, step):
        return self._logger.log_images(tag, img, step)
            
    def log_video(self, tag, video, step, fps):
        return self._logger.log_video(tag, video, step, fps)
            
    def close(self):
        return self._logger.close()