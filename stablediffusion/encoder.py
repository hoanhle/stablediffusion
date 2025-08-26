import torch
from torch import nn
from torch.nn import functional as F
from stablediffusion.decoder import VAE_ResidualBlock, VAE_AttentionBlock

class VAE_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # stage 1 @ 128
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.res1_block1 = VAE_ResidualBlock(128, 128)
        self.res1_block2 = VAE_ResidualBlock(128, 128)

        # downsample to ~H/2 NOTE: use https://ezyang.github.io/convolution-visualizer/ to visualize the output
        self.down1_conv = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)
        self.down1_res1 = VAE_ResidualBlock(128, 256)
        self.down1_res2 = VAE_ResidualBlock(256, 256)

        # downsample to ~H/4
        self.down2_conv = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0)
        self.down2_res1 = VAE_ResidualBlock(256, 512)
        self.down2_res2 = VAE_ResidualBlock(512, 512)

        # downsample to ~H/8
        self.down3_conv = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0)

        # bottleneck @ 512
        self.bottleneck_res1 = VAE_ResidualBlock(512, 512)
        self.bottleneck_res2 = VAE_ResidualBlock(512, 512)
        self.bottleneck_res3 = VAE_ResidualBlock(512, 512)
        self.bottleneck_attn = VAE_AttentionBlock(512)
        self.bottleneck_res4 = VAE_ResidualBlock(512, 512)

        # output head -> 8 channels (mean+logvar = 4+4)
        self.out_norm = nn.GroupNorm(32, 512)
        self.out_act = nn.SiLU()
        self.out_proj_3x3 = nn.Conv2d(512, 8, kernel_size=3, padding=1)
        self.out_proj_1x1 = nn.Conv2d(8, 8, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # stem
        x = self.conv_in(x)
        x = self.res1_block1(x)
        x = self.res1_block2(x)

        # downsample 1
        x = F.pad(x, (0, 1, 0, 1))
        x = self.down1_conv(x)
        x = self.down1_res1(x)
        x = self.down1_res2(x)

        # downsample 2
        x = F.pad(x, (0, 1, 0, 1))
        x = self.down2_conv(x)
        x = self.down2_res1(x)
        x = self.down2_res2(x)

        # downsample 3
        x = F.pad(x, (0, 1, 0, 1))
        x = self.down3_conv(x)

        # bottleneck
        x = self.bottleneck_res1(x)
        x = self.bottleneck_res2(x)
        x = self.bottleneck_res3(x)
        x = self.bottleneck_attn(x)
        x = self.bottleneck_res4(x)

        # head
        x = self.out_norm(x)
        x = self.out_act(x)
        x = self.out_proj_3x3(x)
        x = self.out_proj_1x1(x)
        
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

