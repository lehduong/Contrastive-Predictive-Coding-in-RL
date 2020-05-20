import numpy as np
import torch
import torch.nn as nn

from core.distributions import Bernoulli, Categorical, DiagGaussian
from core.utils import init
from core.networks import CNNBase, MLPBase
from .pg import PolicyGradientAgent

class CPCPolicyGradientAgent(PolicyGradientAgent):
    """
        Contrastive Predictive Coding on top of Policy Gradient
        The difference between this class and PolicyGradientAgent is that when acting, the instances of this class 
            also return state_feat and action_feat
    """
    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, state_feat, rnn_hxs = self.state_encoder(inputs, rnn_hxs, masks)
        dist = self.dist(state_feat)

        # if deterministic greedily choose the most optimal solution otherwise sampling with probability proportional to cummulate reward
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_feat = self.action_encoder(action)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs, state_feat, action_feat
