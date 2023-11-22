import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from torch.utils.data import DataLoader

import sys
sys.path.append(".")

class Generator(nn.Module):
    def __init__(self, x_dim, traj_len, latent_dim):
        super(Generator, self).__init__()

        #self.init_size = traj_len // int(traj_len/2)
        self.x_dim = x_dim
        self.padd = 1
        self.n_filters = 2*self.padd+1
        if traj_len == 64:
            self.Q = 8

        else:
            self.Q = 2

        self.Nch = 512

        self.l1 = nn.Sequential(nn.Linear(latent_dim, self.Nch * self.Q))

        if traj_len == 32:

            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose1d(self.Nch+x_dim, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                
                nn.ConvTranspose1d(128, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose1d(256, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose1d(256, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv1d(128, x_dim, self.n_filters, stride=1, padding=self.padd),
                nn.Tanh(),
            )

        else:
            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose1d(self.Nch+x_dim, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose1d(128, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose1d(256, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv1d(128, x_dim, self.n_filters, stride=1, padding=self.padd),
                nn.Tanh(),
            )


    def forward(self, noise, conditions):
        conds_flat = conditions.view(conditions.shape[0],-1)
        conds_rep = conds_flat.repeat(1,self.Q).view(conditions.shape[0], self.x_dim, self.Q)
        noise_out = self.l1(noise)
        noise_out = noise_out.view(noise_out.shape[0], self.Nch, self.Q)
        gen_input = torch.cat((conds_rep, noise_out), 1)
        traj = self.conv_blocks(gen_input)
        
        return traj


class ParamGenerator(nn.Module):
    def __init__(self, x_dim, y_dim, traj_len, latent_dim):
        super(ParamGenerator, self).__init__()

        self.init_size = traj_len // int(traj_len/2)
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.padd = 1
        self.n_filters = 2*self.padd+1
        self.Q = 1#2
        self.Nch = 1028#512

        self.l1 = nn.Sequential(nn.Linear(latent_dim, self.Nch * self.Q))

        if traj_len == 32:
            if True:
                print("LARGE ARCHITECTURE")
            
                self.conv_blocks = nn.Sequential(
                    nn.ConvTranspose1d(self.Nch+x_dim+y_dim, 128, 4, stride=2, padding=self.padd),
                    nn.BatchNorm1d(128, 0.8),
                    nn.LeakyReLU(0.2, inplace=True),
                    
                    nn.ConvTranspose1d(128, 256, 4, stride=2, padding=self.padd),
                    nn.BatchNorm1d(256, 0.8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.ConvTranspose1d(256, 512, 4, stride=2, padding=self.padd),
                    nn.BatchNorm1d(512, 0.8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.ConvTranspose1d(512,256, 4, stride=2, padding=self.padd),
                    nn.BatchNorm1d(256, 0.8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.ConvTranspose1d(256, 128, 4, stride=2, padding=self.padd),
                    nn.BatchNorm1d(128, 0.8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv1d(128, x_dim, self.n_filters, stride=1, padding=self.padd),
                    nn.Tanh(),
                )
            else:
                self.conv_blocks = nn.Sequential(
                    nn.ConvTranspose1d(self.Nch+x_dim+y_dim, 128, 4, stride=2, padding=self.padd),
                    nn.BatchNorm1d(128, 0.8),
                    nn.LeakyReLU(0.2, inplace=True),
                    
                    nn.ConvTranspose1d(128, 256, 4, stride=2, padding=self.padd),
                    nn.BatchNorm1d(256, 0.8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.ConvTranspose1d(256, 256, 4, stride=2, padding=self.padd),
                    nn.BatchNorm1d(256, 0.8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.ConvTranspose1d(256, 128, 4, stride=2, padding=self.padd),
                    nn.BatchNorm1d(128, 0.8),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv1d(128, x_dim, self.n_filters, stride=1, padding=self.padd),
                    nn.Tanh(),
                )
        else:
            self.conv_blocks = nn.Sequential(
                nn.ConvTranspose1d(self.Nch+x_dim+y_dim, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose1d(128, 256, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(256, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose1d(256, 128, 4, stride=2, padding=self.padd),
                nn.BatchNorm1d(128, 0.8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv1d(128, x_dim, self.n_filters, stride=1, padding=self.padd),
                nn.Tanh(),
            )




    def forward(self, noise, init_state, param):
        init_state_rep = init_state.repeat(1,1,self.Q)
        
        param_rep = param.repeat(1, 1, self.Q)

        noise_out = self.l1(noise)
        noise_out = noise_out.view(noise_out.shape[0], self.Nch, self.Q)
        gen_input = torch.cat((init_state_rep, param_rep, noise_out), 1)

        traj = self.conv_blocks(gen_input)
        #print('GEN ', traj.shape)
        return traj