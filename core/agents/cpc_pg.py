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
        The difference between this class and PolicyGradientAgent is that when EVALUATING, the instances of this class 
            also return state_feat and action_feat
    """
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super().__init__(obs_shape, action_space, base, base_kwargs)
        self.state_encoder = self.base
        self.action_encoder = nn.Embedding(action_space.n, self.state_encoder.output_size)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, state_feat, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(state_feat)

        action_feat = self.action_encoder(action.view(-1))
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs, state_feat, action_feat