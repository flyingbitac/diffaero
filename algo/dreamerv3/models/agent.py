import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
from dataclasses import dataclass
from torch.cuda.amp import autocast

from .blocks import SymLogTwoHotLoss, MLP

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


class ContDist:
    def __init__(self, dist=None, absmax=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean
        self.absmax = absmax

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        out = self._dist.mean
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def sample(self, sample_shape=()):
        out = self._dist.rsample(sample_shape)
        if self.absmax is not None:
            out *= (self.absmax / torch.clip(torch.abs(out), min=self.absmax)).detach()
        return out

    def log_prob(self, x):
        return self._dist.log_prob(x)


def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage*len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per


def calc_lambda_return(rewards, values, termination, gamma, lam, device, dtype=torch.float32):
    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    gamma_return = torch.zeros((batch_size, batch_length+1), dtype=dtype, device=device)
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = \
            rewards[:, t] + \
            gamma * inv_termination[:, t] * (1-lam) * values[:, t] + \
            gamma * inv_termination[:, t] * lam * gamma_return[:, t+1]
    return gamma_return[:, :-1]

@dataclass
class ActorCriticConfig:
    feat_dim: int
    num_layers: int
    hidden_dim: int
    action_dim: int
    gamma: float
    lambd: float
    entropy_coef: float
    device: torch.device
    max_std: float=1.0
    min_std: float=0.1


class ActorCriticAgent(nn.Module):
    def __init__(self, cfg: ActorCriticConfig, envs) -> None:
        super().__init__()
        self.gamma = cfg.gamma
        self.lambd = cfg.lambd
        self.entropy_coef = cfg.entropy_coef
        self.use_amp = False
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self._min_std = cfg.min_std
        self._max_std = cfg.max_std
        self.register_buffer('min_action',torch.tensor(-1.))
        self.register_buffer('max_action', torch.tensor(1.))
        self.min_action: torch.Tensor; self.max_action: torch.Tensor
        
        self.device = cfg.device
        feat_dim = cfg.feat_dim
        hidden_dim = cfg.hidden_dim
        num_layers = cfg.num_layers
        action_dim = cfg.action_dim

        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)

        self.actor_mean_std = nn.Sequential(
            MLP(feat_dim, hidden_dim, hidden_dim, num_layers, 'ReLU', 'LayerNorm',bias=False),
            nn.Linear(hidden_dim, action_dim*2),
        )
        self.critic = nn.Sequential(
            MLP(feat_dim, hidden_dim, hidden_dim, num_layers, 'ReLU', 'LayerNorm', bias=False),
            nn.Linear(hidden_dim, 255),
        )
        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, eps=1e-5)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))
    
    def dist(self,mean,std):
        return torch.distributions.Normal(mean,std)

    def policy(self, x):
        LOG_STD_MAX = 3
        LOG_STD_MIN = -5
        mean_std = self.actor_mean_std(x)
        mean, std = torch.chunk(mean_std, 2, dim=-1)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            torch.tanh(std) + 1)
        # std = torch.exp(log_std).expand_as(mean)
        std = torch.exp(log_std)
        return mean,std

    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_dist_raw_value(self, x):
        mean,std = self.policy(x[:,:-1])
        dist = self.dist(mean,std)
        raw_value = self.critic(x)
        return dist, raw_value

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            mean,std = self.policy(latent)
            dist = self.dist(mean,std)
            if greedy:
                sample = mean
            else:
                sample = dist.sample()
            action = (self.max_action - self.min_action) * (torch.tanh(sample)*0.5 + 0.5) + self.min_action
        return action,sample

    def sample_as_env_action(self, latent, greedy=False):
        action = self.sample(latent, greedy)
        return action.to(torch.float32).detach().cpu().squeeze(0).numpy()

    def update(self, latent, action, reward, termination, logger=None):
        '''
        Update policy and value model
        '''
        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            dist, raw_value = self.get_dist_raw_value(latent)
            log_prob = dist.log_prob(action)
            log_prob = log_prob.sum(-1)
            entropy = dist.entropy().sum(-1)

            # decode value, calc lambda return
            slow_value = self.slow_value(latent)
            slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd,self.device)
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd,self.device)

            # update value function with slow critic regularization
            value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
            slow_value_regularization_loss = self.symlog_twohot_loss(raw_value[:, :-1], slow_lambda_return.detach())

            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound-lower_bound
            norm_ratio = torch.max(torch.ones(1).to(self.device), S)  # max(1, S) in the paper
            norm_advantage = (lambda_return-value[:, :-1]) / norm_ratio
            policy_loss = -(log_prob * norm_advantage.detach()).mean()

            entropy_loss = entropy.mean()

            loss = policy_loss + value_loss + slow_value_regularization_loss - self.entropy_coef * entropy_loss

        # gradient descent
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        gradnorm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()

        if logger is not None:
            logger.log('ActorCritic/policy_loss', policy_loss.item())
            logger.log('ActorCritic/value_loss', value_loss.item())
            logger.log('ActorCritic/entropy_loss', entropy_loss.item())
            logger.log('ActorCritic/S', S.item())
            logger.log('ActorCritic/gradnorm', gradnorm.item())
            logger.log('ActorCritic/total_loss', loss.item())

        agent_info = {
            'ActorCritic/policy_loss': policy_loss.item(),
            'ActorCritic/value_loss': value_loss.item(),
            'ActorCritic/entropy_loss': entropy_loss.item(),
            'ActorCritic/S': S.item(),
            'ActorCritic/gradnorm': gradnorm.item(),
            'ActorCritic/total_loss': loss.item()
        }

        return agent_info