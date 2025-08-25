import torch
from torch import nn
from torch.nn import functional as F
from stablediffusion.decoder import VAE_ResidualBlock, VAE_AttentionBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (B, 3, H, W) -> (B, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (B, 128, H, W)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # (B, 128, H, W) -> (B, 128, ~H/2, ~W/2)
            # NOTE: use padding=1 if to have exact halves
            # use https://ezyang.github.io/convolution-visualizer/ to visualize the output
            # also see the fwd part func below
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            # (B, 256, ~H/2, ~W/2) -> (B, 256, ~H/4, ~W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),

            ## (B, 512, ~H/4, ~W/4) -> (B, 512, ~H/8, ~W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )


    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (pad left, pad right, pad top, pad bottom)
                x = F.pad(x, (0, 1, 0, 1))
        
            x = module(x)
        
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, min=-30, max=20)
        var = logvar.exp()
        std = var.sqrt()
    
        # Z = N(0, 1) -> N(mean, std) = X
        # X = mean + std * Z
        x = mean + std * noise

        # scale the output by a constant
        # taken from the original repository
        # https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215

        return x

