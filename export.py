import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        x = torch.sign(x) * torch.log(1 + torch.abs(x))  # 这是合法的
        return x

# 使用 torch.jit.script 编译
scripted_model = torch.jit.script(MyModel())

# 测试
x = torch.randn(10, 10)
output = scripted_model(x)
