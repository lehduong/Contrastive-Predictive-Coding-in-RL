from .base_storage import RolloutStorage
import torch 


class CPCRolloutStorage(RolloutStorage):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size, hidden_size=512):
        super().__init__(num_steps, num_processes, obs_shape, action_space,
                         recurrent_hidden_state_size)
        # Importance: obs_feat[i] is not the embedded vector of obs[i]
        # it's the embedded of obs[i-1] 
        # more intuitive way to understand, action[i] is based on obs_feat[i] !
        self.obs_feat = torch.zeros(num_steps, num_processes, hidden_size) 
        self.action_feat = torch.zeros(num_steps, num_processes, hidden_size) 
        
    def to(self, device):
        self.obs_feat = self.obs_feat.to(device)
        self.action_feat = self.action_feat.to(device)
        super().to(device)
        
    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, obs_feat, action_feat):
        """
        store features of obs and action for contrastive learning
        """
        self.obs_feat[self.step].copy_(obs_feat)
        self.action_feat[self.step].copy_(action_feat)
        super().insert(obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks)

