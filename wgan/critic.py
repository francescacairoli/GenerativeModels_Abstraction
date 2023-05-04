import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Critic(nn.Module):
    def __init__(self, x_dim, traj_len):
        super(Critic, self).__init__()

        def critic_block(in_filters, out_filters, L):
            padd = 1
            n_filters = 2*padd+2            
            block = [nn.Conv1d(in_filters, out_filters, n_filters, stride=2, padding=padd), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.2)]
            block.append(nn.LayerNorm([out_filters, L]))

            return block

        self.model = nn.Sequential(
            *critic_block(x_dim, 64, traj_len//2),
            *critic_block(64, 64, traj_len//4)
        )

        # The height and width of downsampled image
        ds_size = (traj_len +1) // (2**2)
        self.adv_layer = nn.Sequential(nn.Linear(64 * ds_size, 1))
        
    def forward(self, trajs, conditions):
        d_in = torch.cat((conditions, trajs), 2)
        out = self.model(d_in)
        out_flat = out.view(out.shape[0], -1)
        validity = self.adv_layer(out_flat)
        return validity




class ParamCritic(nn.Module):
    def __init__(self, x_dim, traj_len):
        super(ParamCritic, self).__init__()
        self.traj_len = traj_len
        def critic_block(in_filters, out_filters, L):
            padd = 1
            n_filters = 2*padd+2            
            block = [nn.Conv1d(in_filters, out_filters, n_filters, stride=2, padding=padd), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.2)]
            block.append(nn.LayerNorm([out_filters, L]))

            return block

        self.model = nn.Sequential(
            *critic_block(x_dim, 64, traj_len//2),
            *critic_block(64, 64, traj_len//4)
        )

        # The height and width of downsampled image
        ds_size = (traj_len +1) // (2**2)
        self.adv_layer = nn.Sequential(nn.Linear(64 * ds_size, 1))
        

    def forward(self, traj, init_state, param):

        full_traj = torch.cat((init_state, traj), 2)
        param_rep = param.repeat(1, 1, self.traj_len+1)

        d_in = torch.cat((full_traj, param_rep), 1)

        out = self.model(d_in)
        out_flat = out.view(out.shape[0], -1)
        validity = self.adv_layer(out_flat)
        return validity