from __future__ import division
import torch.nn as nn


class Encoder(nn.Module):
 
    def __init__(self,x_dim, z_dim):
        """
        Encoder initializer
        :param x_dim: dimension of the input
        :param z_dim: dimension of the latent representation
        """
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(x_dim),512),
            nn.Tanh(),
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512, z_dim),
        )


    def forward(self, img):
        out = self.model(img)
        return out

class Decoder(nn.Module):

    def __init__(self,x_dim, z_dim):
        """
        Encoder initializer
        :param x_dim: dimension of the input
        :param z_dim: dimension of the latent representation
        """
        super(Decoder, self).__init__()
       
        
        self.model = nn.Sequential(
            nn.Linear(z_dim,512),
            nn.Tanh(),
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512,x_dim)
        )

    def forward(self, z):
        img= self.model(z)
        return img     