import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        """
        TODO 2.1 : Fill in self.convs following the given architecture
         Sequential(
                (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (1): ReLU()
                (2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (3): ReLU()
                (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
                (5): ReLU()
                (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            )
        """
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),                # [3, 32, 32] -> [32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),               # [64, 16, 16]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),              # [128, 8, 8]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))              # [256, 4, 4]
        )

        #TODO 2.1: fill in self.fc, such that output dimension is self.latent_dim
        self.last_conv_shape = torch.tensor([256, input_shape[1]//8, input_shape[2]//8])        # [256, 4, 4]
        self.fc = nn.Linear(torch.prod(self.last_conv_shape), self.latent_dim)                  # [256*4*4, 256]
    
    def forward(self, x):
        #TODO 2.1 : forward pass through the network, output should be of dimension : self.latent_dim
        x = self.convs(x)                                                                       # [3, 32, 32] -> [256, 4, 4]
        x = x.view(x.shape[0], -1) # flatten                                                    # [256, 4, 4] -> [256*4*4]
        out = self.fc(x)                                                                        # [256*4*4]   -> [256]
        return out
    
class VAEEncoder(Encoder):
    def __init__(self, input_shape, latent_dim):
        super().__init__(input_shape, latent_dim)
        #TODO 2.4: fill in self.fc, such that output dimension is 2*self.latent_dim
        self.fc = nn.Linear(torch.prod(self.last_conv_shape), 2*self.latent_dim)                # [256*4*4, 256*2]

    def forward(self, x):
        #TODO 2.4: forward pass through the network.
        # should return a tuple of 2 tensors, mu and log_std
        x = self.convs(x)                                                                       # [3, 32, 32] -> [256, 4, 4]
        x = x.view(x.shape[0], -1) # flatten                                                    # [256, 4, 4] -> [256*4*4]
        out = self.fc(x)                                                                        # [256*4*4]   -> [256*2]              
        mu, log_std = out[:, :self.latent_dim], out[:, self.latent_dim:]                        # [256*2]     -> [256, 256]
        
        return mu, log_std


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        #TODO 2.1: fill in self.base_size
        self.base_size = [256, output_shape[1]//8, output_shape[2]//8] # from the last conv layer of the encoder
        """
        TODO 2.1 : Fill in self.deconvs following the given architecture
        Sequential(
                (0): ReLU()
                (1): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (2): ReLU()
                (3): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (4): ReLU()
                (5): ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                (6): ReLU()
                (7): Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        """
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),        # [128, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),         # [64, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),          # [32, 32, 32]
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))                     # [3, 32, 32]
        )
        self.fc = nn.Linear(self.latent_dim, self.base_size[0]*self.base_size[1]*self.base_size[2]) # [256, 256*4*4]

    def forward(self, z):
        #TODO 2.1: forward pass through the network, first through self.fc, then self.deconvs.
        z = self.fc(z)                                                                              # [256]       -> [256*4*4]
        z = z.view(z.shape[0], *self.base_size)                                                     # [256*4*4]   -> [256, 4, 4]
        out = self.deconvs(z)                                                                       # [256, 4, 4] -> [3, 32, 32]
        # out = z.view(z.shape[0], *self.output_shape)                                      no need # [3, 32, 32] -> [3, 32, 32]
        return out

class AEModel(nn.Module):
    def __init__(self, variational, latent_size, input_shape = (3, 32, 32)):
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        if variational:
            self.encoder = VAEEncoder(input_shape, latent_size)
        else:
            self.encoder = Encoder(input_shape, latent_size)
        self.decoder = Decoder(latent_size, input_shape)
    #NOTE: You don't need to implement a forward function for AEModel. For implementing the loss functions in train.py, call model.encoder and model.decoder directly.
