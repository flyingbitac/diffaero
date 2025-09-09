from typing import Tuple, Dict, Union, Optional, List

import torch
import torch.nn as nn

class module(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([
            obs["pos"],
            obs["vel"],
            obs["acc"]
        ], dim=-1)
    
    def export(self):
        dummy_input = {
            "pos": torch.randn(1, 3),
            "vel": torch.randn(1, 3),
            "acc": torch.randn(1, 3)
        }
        torch.onnx.export(
            model=self,
            kwargs={"obs": dummy_input},
            f="/home/zxh/ws/diffaero/script/exported_model.onnx",
        )

def forward(obs: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([
        obs["pos"],
        obs["vel"],
        obs["acc"]
    ], dim=-1)

if __name__ == "__main__":
    module().export()
    dummy_input = {
        "pos": torch.randn(1, 3),
        "vel": torch.randn(1, 3),
        "acc": torch.randn(1, 3)
    }
    torch.onnx.export(
        model=torch.jit.script(forward),
        kwargs={"obs": dummy_input},
        f="/home/zxh/ws/diffaero/script/exported_model_function.onnx",
    )