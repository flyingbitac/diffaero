import torch
import torch.nn as nn

device = 'cuda:1'
model = nn.Sequential(nn.Linear(10,3)).to(device)
a = torch.randn(5,10).to(device)
b = torch.randn(5,3).to(device)
optim = torch.optim.Adam(model.parameters(), lr=0.01)
loss = torch.mean(torch.sum((b - model(a))**2, dim=-1))
loss.backward(retain_graph=True)
optim.step()
optim.zero_grad()
loss.backward()

###############################################################################
# 1. collect datas to train world model
# 2. when train world model, use retain graph
# 3. freeze the world model, train the policy model