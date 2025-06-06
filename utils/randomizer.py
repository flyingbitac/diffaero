from typing import Tuple, Union, List, Optional

from omegaconf import DictConfig
import torch

class RandomizerBase:
    def __init__(
        self,
        shape: Union[int, List[int], torch.Size],
        default_value: Union[float, bool],
        device: torch.device,
        dtype: torch.dtype = torch.float,
    ):
        self.value = torch.zeros(shape, device=device, dtype=dtype)
        self.default_value = default_value
        self.randomize()

    def __str__(self) -> str:
        return str(self.value)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.value.shape
    
    def randomize(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError
    
    def default(self) -> torch.Tensor:
        self.value = torch.full_like(self.value, self.default_value)
        return self.value
    
    def __add__(self, other):
        return self.value + other
    def __sub__(self, other):
        return self.value - other
    def __mul__(self, other):
        return self.value * other
    def __div__(self, other):
        return self.value / other
    def __neg__(self):
        return -self.value
    def reshape(self, shape: Union[int, List[int], torch.Size]):
        return self.value.reshape(shape)
    def squeeze(self, dim: int = -1):
        return self.value.squeeze(dim)
    def unsqueeze(self, dim: int = -1):
        return self.value.unsqueeze(dim)

class UniformRandomizer(RandomizerBase):
    def __init__(
        self,
        shape: Union[int, List[int], torch.Size],
        default_value: Union[float, bool],
        device: torch.device,
        enabled: bool = True,
        low: float = 0.0,
        high: float = 1.0,
        dtype: torch.dtype = torch.float,
    ):
        self.low = low
        self.high = high
        self.enabled = enabled
        super().__init__(shape, default_value, device, dtype)
    
    def randomize(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.enabled:
            return self.default()
        if idx is not None:
            mask = torch.zeros_like(self.value, dtype=torch.bool)
            mask[idx] = True
            new = torch.rand_like(self.value) * (self.high - self.low) + self.low
            self.value = torch.where(mask, new, self.value)
        else:
            self.value.uniform_(self.low, self.high)
        return self.value
    
    def __repr__(self) -> str:
        return f"UniformRandomizer(low={self.low}, high={self.high}, default={self.default_value}, shape={self.value.shape}, device={self.value.device}, dtype={self.value.dtype})"

class NormalRandomizer(RandomizerBase):
    def __init__(
        self,
        shape: Union[int, List[int], torch.Size],
        default_value: Union[float, bool],
        device: torch.device,
        enabled: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
        dtype: torch.dtype = torch.float,
    ):
        self.mean = mean
        self.std = std
        self.enabled = enabled
        super().__init__(shape, default_value, device, dtype)
    
    def randomize(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.enabled:
            return self.default()
        if idx is not None:
            mask = torch.zeros_like(self.value, dtype=torch.bool)
            mask[idx] = True
            new = torch.randn_like(self.value) * self.std + self.mean
            self.value = torch.where(mask, new, self.value)
        else:
            self.value.normal_(self.mean, self.std)
        return self.value
    
    def __repr__(self) -> str:
        return f"NormalRandomizer(mean={self.mean}, std={self.std}, default={self.default_value}, shape={self.value.shape}, device={self.value.device}, dtype={self.value.dtype})"

class RandomizerManager:
    randomizers: List[Union[UniformRandomizer, NormalRandomizer]] = []
    def __init__(
        self, 
        cfg: DictConfig,
    ):
        self.enabled: bool = cfg.enabled
        
    def randomize(self, idx: Optional[torch.Tensor] = None):
        if self.enabled:
            for randomizer in self.randomizers:
                randomizer.randomize(idx)
        else:
            for randomizer in self.randomizers:
                randomizer.default()
    
    def __str__(self) -> str:
        return (
            "RandomizeManager(\n\t" + 
            f"Enabled: {self.enabled},\n\t" +
            ",\n\t".join([randomizer.__repr__() for randomizer in self.randomizers]) + 
            "\n)"
        )

def build_randomizer(
    cfg: DictConfig,
    shape: Union[int, List[int], torch.Size],
    device: torch.device,
    dtype: torch.dtype = torch.float,
) -> Union[UniformRandomizer, NormalRandomizer]:
    if hasattr(cfg, "min") and hasattr(cfg, "max"):
        randomizer = UniformRandomizer(
            shape=shape,
            default_value=cfg.default,
            device=device,
            enabled=cfg.enabled,
            low=cfg.min,
            high=cfg.max,
            dtype=dtype,
        )
    elif hasattr(cfg, "mean") and hasattr(cfg, "std"):
        randomizer = NormalRandomizer(
            shape=shape,
            default_value=cfg.default,
            device=device,
            enabled=cfg.enabled,
            mean=cfg.mean,
            std=cfg.std,
            dtype=dtype,
        )
    else:
        raise ValueError("Invalid randomizer configuration. Must contain 'min' and 'max' for UniformRandomizer or 'mean' and 'std' for NormalRandomizer.")
    RandomizerManager.randomizers.append(randomizer)
    return randomizer

if __name__ == "__main__":
    # Example usage
    print(UniformRandomizer([2, 3], 0.5, torch.device("cpu"), low=0.0, high=1.0).randomize(torch.tensor([0])))
    print(build_randomizer(DictConfig({"defalut": 0.5, "min": 0, "max": 1}), [2, 3], torch.device("cpu")).randomize())
    print(build_randomizer(DictConfig({"defalut": 0.5, "mean": 0, "std": 1}), [2, 3], torch.device("cpu")).randomize())
    print(RandomizerManager(DictConfig({"enable": False})))