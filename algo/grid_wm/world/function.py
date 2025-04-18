import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

@torch.no_grad()
def symlog(x:torch.Tensor):
    return torch.sign(x) * torch.log1p(torch.abs(x))

@torch.no_grad()
def symexp(x:torch.Tensor):
    return torch.sign(x) * torch.expm1(torch.abs(x))

def mse(pred:torch.Tensor, target:torch.Tensor):
    mse_loss = (pred - target) ** 2
    mse_loss = reduce(mse_loss, 'b l ... -> b l', 'sum')
    return mse_loss.mean()

class SymLogTwoHotLoss(nn.Module):
    def __init__(self, num_classes, lower_bound, upper_bound):
        super().__init__()
        self.num_classes = num_classes
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bin_length = (upper_bound - lower_bound) / (num_classes-1)

        # use register buffer so that bins move with .cuda() automatically
        self.bins: torch.Tensor
        self.register_buffer(
            'bins', torch.linspace(-20, 20, num_classes), persistent=False)

    def forward(self, output, target):
        target = symlog(target)
        assert target.min() >= self.lower_bound and target.max() <= self.upper_bound

        index = torch.bucketize(target, self.bins)
        diff = target - self.bins[index-1]  # -1 to get the lower bound
        weight = diff / self.bin_length
        weight = torch.clamp(weight, 0, 1)
        weight = weight.unsqueeze(-1)

        target_prob = (1-weight)*F.one_hot(index-1, self.num_classes) + weight*F.one_hot(index, self.num_classes)

        loss = -target_prob * F.log_softmax(output, dim=-1)
        loss = loss.sum(dim=-1)
        return loss.mean()

    def decode(self, output):
        return symexp(F.softmax(output, dim=-1) @ self.bins)

class EMAScalar():
    def __init__(self, decay) -> None:
        self.scalar = 0.0
        self.decay = decay

    def __call__(self, value):
        self.update(value)
        return self.get()

    def update(self, value):
        self.scalar = self.scalar * self.decay + value * (1 - self.decay)

    def get(self):
        return self.scalar

def get_unimix_logits(logits:torch.Tensor, unimix:float=0.01):
    probs = F.softmax(logits, dim=-1)
    probs = (1-unimix) * probs + unimix / probs.size(-1)
    return torch.log(probs)

class CategoricalLossWithFreeBits:
    def __init__(self, free_bits:float=1.):
        self.free_bits = free_bits
    
    def kl_loss(self, post_probs:torch.Tensor, prior_probs:torch.Tensor):
        post_dist = torch.distributions.OneHotCategorical(logits=post_probs)
        prior_dist = torch.distributions.OneHotCategorical(logits=prior_probs)
        real_kl_loss = torch.distributions.kl_divergence(post_dist, prior_dist)
        real_kl_loss = reduce(real_kl_loss, '... D -> ...', 'sum')
        real_kl_loss = torch.mean(real_kl_loss)
        kl_loss = torch.max(self.free_bits*torch.ones_like(real_kl_loss), real_kl_loss)
        return kl_loss