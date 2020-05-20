import torch
import torch.nn as nn
import torch.optim as optim

from core.algorithms.kfac import KFACOptimizer
from .a2c_acktr import A2C_ACKTR

class CPC_A2C_ACKTR(A2C_ACKTR):
    """
        Incorporate contrastive predictive coding to a2c/acktr algorithm
            this class return an auxiliary loss at update method for learning representation of state and action.
    """
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 num_steps=200):
        super().__init__(actor_critic, value_loss_coef, entropy_coef, lr, eps, alpha)
        self.num_steps = num_steps  # number of steps per gradient update (trade off between bias and variance)
        hidden_dim = actor_critic.recurrent_hidden_state_size
        self.Wk_state  = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(num_steps)])
        self.Wk_state_action  = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(num_steps)])
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()

    def update(self, rollouts):
        """
        :param rollouts: CPCRolloutStorage object
        """
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(
                -1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()
        
        _, _, nce_state, nce_state_action = self.cpc(rollouts)
        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Compute fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean()

            fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        self.optimizer.zero_grad()
        (value_loss * self.value_loss_coef + action_loss -
         dist_entropy * self.entropy_coef + nce_state + nce_state_action).backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                     self.max_grad_norm)

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item(), nce_state.item(), nce_state_action.item()
    
    def cpc(self, rollouts):
        """
            Contrastive Predictive Coding for learning representation and density ratio
            Learn h(a_t|s_t, s_k)/p(a_t|s_t) by estimating TWO density ratio
                p(s_t,a_t|s_k)/p(s_t,a_t) and p(s_t|s_k)/p(s_t)
        """
        obs_feat = rollouts.obs_feat  # of shape: (timestep, n_processes, hidden_size)
        action_feat = rollouts.action_feat  # of shape: (timestep, n_processes, hidden_size)
        _, n_processes, hidden_size = action_feat.shape

        # create vector combining both action and state
        # used for learning p(s_t,a_t | s_k)/p(s_t,a_t)
        obs_action_feat = obs_feat+action_feat
        
        # s_t to compute p(s_t|s_k)
        state_condition = obs_feat[0].view(n_processes, hidden_size)
        # (s_t,a_t) to compute p(s_t,a_t|s_k)
        state_action_condition = obs_action_feat[0].view(n_processes, hidden_size)

        # compute W_i*c_t
        # num_steps * n_processes * hidden_size
        pred_state = torch.empty(self.num_steps, n_processes, hidden_size).float()
        pred_state_action = torch.empty(self.num_steps, n_processes, hidden_size).float()
        for i in range(self.num_steps):
            # condition s_t
            linear_state = self.Wk_state[i]
            pred_state[i] = linear_state(state_condition)
            
            # condition s_t, a_t
            linear_state_action = self.Wk_state_action[i]
            pred_state_action[i] = linear_state_action(state_action_condition)
        
        # transpose pred_state and pred_state_action to num_steps, hidden_size, n_processes
        pred_state = pred_state.permute(0, 2, 1)
        pred_state_action = pred_state_action.permute(0, 2, 1)
        
        # compute nce
        for i in range(self.num_steps):
            state_total = torch.mm(obs_feat[i], pred_state[i])
            state_action_total = torch.mm(obs_action_feat[i], pred_state_action[i])
            # accuracy
            correct_state = torch.sum(torch.eq(torch.argmax(self.softmax(state_total), dim=0), torch.arange(0, n_processes)))
            correct_state_action = torch.sum(torch.eq(torch.argmax(self.softmax(state_action_total), dim=0), torch.arange(0, n_processes)))

            # nce
            nce_state += torch.sum(torch.diag(self.lsoftmax(total_state)))
            nce_state_action += torch.sum(torch.diag(self.lsoftmax(total_state_action)))
        # infonce loss
        nce_state /= -1*n_processes*self.num_steps
        nce_state_action /= -1*n_processes*self.num_steps
        # accuracy
        accuracy_state = 1.*correct_state.item()/n_processes
        accuracy_state_action = 1.*correct_state_action.item()/n_processes

        return accuracy_state, accuracy_state_action, nce_state, nce_state_action
