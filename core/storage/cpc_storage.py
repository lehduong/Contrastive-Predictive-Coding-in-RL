from .base_storage import RolloutStorage


class CPCRolloutStorage(RolloutStorage):
    def __init__(self, num_steps, num_processes, obs_shape, action_space,
                 recurrent_hidden_state_size):
        super().__init__(num_steps, num_processes, obs_shape, action_space,
                         recurrent_hidden_state_size)
        # recurrent_hidden_state has same size as features of observation/action
        self.obs_feat = torch.zeros(num_steps+1, num_processes, recurrent_hidden_state_size) 
        self.action_feat = torch.zeros(num_steps+1, num_processes, recurrent_hidden_state_size) 
        
    def to(self, device):
        self.obs_feat = self.obs_feat.to(device)
        self.action_feat = self.action_feat.to(device)
        super().to(device)
        
    def insert(self, obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks, obs_feat, action_feat):
        """
        store features of obs and action for contrastive learning
        """
        self.obs_feat[self.step + 1].copy_(obs_feat)
        self.action_feat[self.step + 1].copy_(action_feat)
        super().insert(obs, recurrent_hidden_states, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks)

    def after_update(self):
        self.obs_feat[0].copy_(self.obs_feat[-1])
        self.action_feat[0].copy_(self.act_feat[-1])
        super().after_update()

