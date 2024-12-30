import torch
import math

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


from typing import List
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MyBlock(nn.Module):
    def __init__(self,shape, in_c, out_c,  kernel_size=3, stride=1, padding=1, normalize = False, activation=None, groups=8):
        super(MyBlock, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.groups = groups
        self.shape = shape
        # Group Normalization
        self.gn1 = nn.GroupNorm(groups, out_c)
        self.gn2 = nn.GroupNorm(groups, out_c)

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)

        # Activation
        self.activation = nn.SiLU() if activation is None else activation

    def forward(self, x):
        # First convolution + GroupNorm + activation
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.activation(x)

        # Second convolution + GroupNorm + activation
        x = self.conv2(x)
        x = self.gn2(x)
        x = self.activation(x)

        return x
    
class UNet(nn.Module):
    def __init__(self, device=None, in_channels: int = 1, 
                down_channels: List[int] =[64, 128, 128, 128],
                up_channels: List[int] = [128, 128, 128, 64],
                time_emb_dim: int = 128,
                num_classes: int = 10) -> None:
        super(UNet, self).__init__()
        down_channels=[64, 128, 128, 128]
        up_channels = [128, 128, 128, 64]
        # embeddings
        self.device = device
        self.time_mlp = SinusoidalPositionEmbeddings(time_emb_dim)
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)
        self.num_classes = num_classes

        # First half (Downsampling)
        self.te1 = self._make_te(time_emb_dim, in_channels)
        self.b1 = MyBlock((in_channels, 32, 32), in_channels, down_channels[0])
        self.down1 = nn.Conv2d(down_channels[0], down_channels[0], kernel_size=4, stride=2, padding=1)

        self.te2 = self._make_te(time_emb_dim, down_channels[0])
        self.b2 = MyBlock((down_channels[0], 16, 16), down_channels[0], down_channels[1])
        self.down2 = nn.Conv2d(down_channels[1], down_channels[1], kernel_size=4, stride=2, padding=1)

        self.te3 = self._make_te(time_emb_dim, down_channels[1])
        self.b3 = MyBlock((down_channels[1], 8, 8), down_channels[1], down_channels[2])
        self.down3 = nn.Conv2d(down_channels[2], down_channels[2], kernel_size=4, stride=2, padding=1)

        self.te4 = self._make_te(time_emb_dim, down_channels[2])
        self.b4 = MyBlock((down_channels[2], 4, 4), down_channels[2], down_channels[3])
        self.down4 = nn.Conv2d(down_channels[3], down_channels[3], kernel_size=4, stride=2, padding=1)

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, down_channels[3])
        self.b_mid = MyBlock((down_channels[3], 2, 2), down_channels[3], 128)
        self.up1 = nn.ConvTranspose2d(128, up_channels[0], kernel_size=4, stride=2, padding=1)


        # Second half (Upsampling)
        self.te5 = self._make_te(time_emb_dim, up_channels[0])
        self.b5 = MyBlock((up_channels[0]*2, 4, 4), up_channels[0] + down_channels[3], up_channels[0])
        self.up2 = nn.ConvTranspose2d(up_channels[0], up_channels[1], kernel_size=4, stride=2, padding=1)
        
        self.te6 = self._make_te(time_emb_dim, up_channels[1])
        self.b6 = MyBlock((up_channels[1]*2, 8, 8), up_channels[1] + down_channels[2], up_channels[1])
        self.up3 = nn.ConvTranspose2d(up_channels[1], up_channels[2], kernel_size=4, stride=2, padding=1)
        
        self.te7 = self._make_te(time_emb_dim, up_channels[2])
        self.b7 = MyBlock((up_channels[2]*2, 16, 16), up_channels[2] + down_channels[1], up_channels[2])
        self.up4 = nn.ConvTranspose2d(up_channels[2], up_channels[3], kernel_size=4, stride=2, padding=1)

        self.te8 = self._make_te(time_emb_dim, up_channels[3])
        self.b8 =MyBlock((up_channels[3]*2, 32, 32),up_channels[3] + down_channels[0], up_channels[3])


        self.b_out = nn.Conv2d(up_channels[3], 1, kernel_size=1)


    def forward(self, x: torch.Tensor, timestep: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        time_embd = self.time_mlp(timestep).to(self.device)
        label_embd = self.class_emb(label).to(self.device)
        batch_size = x.size(0)
        if batch_size != 128:
            time_embd= time_embd[:x.size(0)]
            label_embd =label_embd[:x.size(0)]
        
        embd = time_embd + label_embd

        embd1 = self.te1(embd).view(batch_size, -1, 1, 1)
        x = x + embd1
        down1 = self.b1(x)  # (N, down_channels[0], 32, 32)
        out1 = self.down1(down1)  # (N, down_channels[0], 28, 28)\

        embd2 = self.te2(embd).view(batch_size, -1, 1, 1)
        out1 = out1 + embd2
        down2 = self.b2(out1)  # (N, down_channels[1], 16, 16)
        out2 = self.down2(down2)  # (N, down_channels[1], 14, 14)

        embd3 = self.te3(embd).view(batch_size, -1, 1, 1)
        out2 = out2 + embd3
        down3 = self.b3(out2)  # (N, down_channels[2], 7, 7)
        out3 = self.down3(down3)  # (N, down_channels[2], 7, 7)

        embd4 = self.te4(embd).view(batch_size, -1, 1, 1)
        out3 = out3 + embd4
        down4 = self.b4(out3)  # (N, down_channels[3], 3, 3)
        out4 = self.down4(down4)  # (N, down_channels[3], 3, 3)

        te_mid = self.te_mid(embd).view(batch_size, -1, 1, 1)
        out_mid = self.b_mid(out4 + te_mid)  # (N, 40, 3, 3)
        
        up1 = self.up1(out_mid)
        up_te5 = self.te5(embd).view(batch_size, -1, 1, 1)
        up1 = up1 + up_te5
        up1 = torch.cat([up1, down4], dim=1)
        up_b5 = self.b5(up1)

        up2 = self.up2(up_b5)
        up_te6 = self.te6(embd).view(batch_size, -1, 1, 1)
        up2 = up2 + up_te6
        up2 = torch.cat([up2, down3], dim=1)
        up_b6 = self.b6(up2)
        up3 = self.up3(up_b6)
        up_te7 = self.te7(embd).view(batch_size, -1, 1, 1)
        up3 = up3 + up_te7
        up3 = torch.cat([up3, down2], dim=1)
        up_b7 = self.b7(up3)

        up4 = self.up4(up_b7)
        up_te8 = self.te8(embd).view(batch_size, -1, 1, 1)
        up4 = up4 + up_te8
        up4 = torch.cat([up4, down1], dim=1)
        up_b8 = self.b8(up4)

        # Final Classification Layer
        output = self.b_out(up_b8)

        return output

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )


class VarianceScheduler:
    def __init__(self, beta_start: int=0.0001, beta_end: int=0.02, num_steps: int=1000, interpolation: str='linear') -> None:
        self.num_steps = num_steps

        # find the beta valuess by linearly interpolating from start beta to end beta
        if interpolation == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_steps,requires_grad=False)
        elif interpolation == 'quadratic':
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_steps,requires_grad=False)
            self.betas = self.betas**2
        else:
            raise Exception('[!] Error: invalid beta interpolation encountered...')
        
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        # send the tensors to the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alpha_bars = self.alpha_bars.to(self.device)

    def add_noise(self, x:torch.Tensor, time_step:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        # Ensure time_step is of type torch.long
        time_step = time_step.long().to(device)
        alpha_bar_t = self.alpha_bars[time_step].view(-1, 1, 1, 1).to(device)  # Reshape for broadcasting
        noise = torch.randn_like(x)
        noisy_input = (alpha_bar_t**0.5) * x + ((1 - alpha_bar_t)**0.5) * noise
        return noisy_input, noise


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # TODO: compute the sinusoidal positional encoding of the time
        embeddings = self.time_embedding(time)
        return embeddings
    def time_embedding(self, time: torch.Tensor) -> torch.Tensor:
        if not isinstance(time, torch.Tensor):
            time = torch.tensor(time, dtype=torch.float32, device=next(self.parameters()).device)

        device = time.device
        half_dim = self.dim // 2
        emb = torch.arange(half_dim, device=device).float() / half_dim
        emb = time[:, None] * (10000 ** -emb)
        embeddings = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return embeddings


def debug_tensor(tensor, name):
    if torch.isnan(tensor).any():
        print(f"{name} has NaN!")
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"{name} has Inf!")
        raise ValueError(f"Inf detected in {name}")


class VAE(nn.Module):
    def __init__(self, 
                in_channels: int, 
                height: int=32, 
                width: int=32, 
                mid_channels: List=[32, 32, 32], 
                latent_dim: int=32, 
                num_classes: int=10) -> None:
        
        super().__init__()

        self.height = height
        self.width = width
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.mid_channels = mid_channels 

        # NOTE: self.mid_size specifies the size of the image [C, H, W] in the bottleneck of the network
        self.mid_size = [mid_channels[-1], height // (2 ** (len(mid_channels)-1)), width // (2 ** (len(mid_channels)-1))]

        # NOTE: You can change the arguments of the VAE as you please, but always define self.latent_dim, self.num_classes, self.mid_size
        
        # TODO: handle the label embedding here
        self.class_emb = nn.Embedding(num_classes, latent_dim)
        
        # TODO: define the encoder part of your network
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels[0], kernel_size=4, stride=2, padding=1),  # Downsample
            nn.BatchNorm2d(mid_channels[0]),
            nn.ReLU(),
            nn.Conv2d(mid_channels[0], mid_channels[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels[1]),
            nn.ReLU(),
            nn.Conv2d(mid_channels[1], mid_channels[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels[2]),
            nn.ReLU()
        )
        
        flattened_size = int(self.mid_size[0] * self.mid_size[1] * self.mid_size[2] / 4)
        self.mean_net = nn.Linear(flattened_size*2, latent_dim)        
        self.logvar_net = nn.Linear(flattened_size*2, latent_dim)

        self.label_convert1 = nn.Linear(2*self.latent_dim, mid_channels[-1]*10)
        self.label_convert2 = nn.Linear(mid_channels[-1]*10, flattened_size*2)
        
        # TODO: define the decoder part of your network
        self.decoder1 = nn.Sequential(
            #nn.Unflatten(1, (self.mid_size[0]//2, self.mid_size[1], self.mid_size[2])),
            nn.ConvTranspose2d(mid_channels[-1]//2, mid_channels[-2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels[-2]),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels[-2], mid_channels[-3], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(mid_channels[-3]),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels[-3], in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid())
        
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: compute the output of the network encoder
        out = self.encoder(x)
        out = out.flatten(start_dim=1)
        # TODO: estimating mean and logvar
        mean = self.mean_net(out)
        logvar = self.logvar_net(out)
        # TODO: computing a sample from the latent distribution
        sample = self.reparameterize(mean, logvar)
        # TODO: decoding the sample
        out = self.decode(sample, label)
        return out, mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: implement the reparameterization trick: sample = noise * std + mean
        std = torch.exp(0.5 * logvar)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        noise = torch.randn_like(std).to(device)
        sample = noise * std + mean

        return sample

    @staticmethod
    def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # TODO: compute the binary cross entropy between the pred (reconstructed image) and the traget (ground truth image)
        loss = F.binary_cross_entropy(pred, target, reduction='sum')

        return loss

    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # TODO: compute the KL divergence
        kl_div = -.5 * (logvar.flatten(start_dim=1) + 1 - torch.exp(logvar.flatten(start_dim=1)) - mean.flatten(start_dim=1).pow(2)).sum()

        return kl_div

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device=torch.device('cuda'), labels: torch.Tensor = None):
      if labels is not None:
          assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
          labels = labels.to(device)
      else:
          # Randomly generate labels
          labels = torch.randint(0, self.num_classes, [num_samples,], device=device)
      # Sample from the standard normal distribution
      noise = torch.randn(num_samples, self.latent_dim, device=device)
      # Decode the noise based on the given labels
      out = self.decode(noise, labels)

      return out



    def decode(self, sample: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Embed the label into the latent space
        batch_size = sample.shape[0]
        label_embedding = self.class_emb(labels).to(sample.device)
        
        # concat the label embedding with the sample
        le = torch.cat([sample, label_embedding], dim=1)
        sample = F.relu(self.label_convert2(F.relu(self.label_convert1(le))))
        # Pass through the decoder layers
        sample = sample.reshape((batch_size, self.mid_size[0]//2, 4, 4))
        out = self.decoder1(sample)


        return out


class LDDPM(nn.Module):
    def __init__(self, network: nn.Module, vae: VAE, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.add_noise = var_scheduler.add_noise
        self.vae = vae
        self.network = network

        # freeze vae
        self.vae.requires_grad_(False)
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # TODO: uniformly sample as many timesteps as the batch size
        batch_size = x.size(0)
        t = torch.randint(0, self.var_scheduler.num_steps, (batch_size,), device=x.device).long()


        # TODO: generate the noisy input
        noisy_input, noise = self.add_noise(x, t)

        # TODO: estimate the noise
        estimated_noise = self.network(noisy_input, t, label)

        # compute the loss (either L1 or L2 loss)
        loss = F.mse_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: implement the sample recovery strategy of the DDPM
        alpha_t = self.var_scheduler.alphas[timestep]
        alpha_bar_t = self.var_scheduler.alpha_bars[timestep]
        if timestep == 0:
            alpha_bar_t_1 = torch.tensor(1.0, dtype=alpha_bar_t.dtype, device=alpha_bar_t.device)
        else:
            alpha_bar_t_1 = self.var_scheduler.alpha_bars[timestep - 1]
        
        beta_t = self.var_scheduler.betas[timestep]
        X_t = noisy_sample
        
        sample = (1/alpha_t**0.5) * ( X_t - (beta_t/(1-alpha_bar_t)**0.5) * estimated_noise )
        if timestep>0:
            sigma_t = torch.sqrt( ( (1 - alpha_bar_t_1)/(1-alpha_bar_t) ) * beta_t )
            epsilon = torch.randn_like(X_t)     
            sample += sigma_t * epsilon

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        else:
            labels = torch.randint(0, self.vae.num_classes, [num_samples,], device=device)
        
        # TODO: using the diffusion model generate a sample inside the latent space of the vae
        # NOTE: you need to recover the dimensions of the image in the latent space of your VAE
        
        if labels is None:
            labels = torch.randint(0, self.vae.num_classes, (num_samples,), device=device)
        
        latent_sample = torch.randn((num_samples, self.vae.latent_dim), device=device)
        for t in reversed(range(self.var_scheduler.num_steps)):
            estimated_noise = self.network(latent_sample, torch.tensor([t], device=device), labels)
            sample = self.recover_sample(latent_sample, estimated_noise, torch.tensor([t], device=device))

        sample = self.vae.decode(sample, labels)
        
        return sample


class DDPM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network
        self.num_steps = var_scheduler.num_steps


    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: uniformly sample as many timesteps as the batch size
        batch_size, channel, height, width = x.shape
        t = torch.randint(0, self.num_steps, (batch_size,), device=x.device)

        # TODO: generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)
        # TODO: estimate the noise
        estimated_noise = self.network(noisy_input, t, label.to(x.device))
        # TODO: compute the loss (either L1, or L2 loss)
        loss = F.l1_loss(estimated_noise, noise)

        return loss

    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: implement the sample recovery strategy of the DDPM
        alpha_t = self.var_scheduler.alphas[timestep]
        alpha_bar_t = self.var_scheduler.alpha_bars[timestep]
        if timestep == 0:
            alpha_bar_t_1 = torch.tensor(1.0, dtype=alpha_bar_t.dtype, device=alpha_bar_t.device)
        else:
            alpha_bar_t_1 = self.var_scheduler.alpha_bars[timestep - 1]
        
        beta_t = self.var_scheduler.betas[timestep]
        X_t = noisy_sample
        
        sample = (1/alpha_t**0.5) * ( X_t - (beta_t/(1-alpha_bar_t)**0.5) * estimated_noise )
        if timestep>0:
            sigma_t = torch.sqrt( ( (1 - alpha_bar_t_1)/(1-alpha_bar_t) ) * beta_t )
            epsilon = torch.randn_like(X_t)     
            sample += sigma_t * epsilon

        return sample

    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None
        self.num_samples = num_samples
        # TODO: apply the iterative sample generation of the DDPM
        sample = torch.randn((num_samples,1,32,32), device = device)
        for t in reversed(range(0,self.var_scheduler.num_steps)):
            time = torch.full((num_samples,),t, dtype=torch.long, device=device)  # Convert timestep to tensor
            estimated_noise = self.network(sample, time, labels)
            sample = self.recover_sample(sample, estimated_noise, t)
        return sample


class DDIM(nn.Module):
    def __init__(self, network: nn.Module, var_scheduler: VarianceScheduler) -> None:
        super().__init__()

        self.var_scheduler = var_scheduler
        self.network = network
        self.num_steps = var_scheduler.num_steps
    
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: uniformly sample as many timesteps as the batch size
        t = torch.randint(0, self.num_steps, (x.size(0),), device=x.device)  
        # TODO: generate the noisy input
        noisy_input, noise = self.var_scheduler.add_noise(x, t)

        
        # TODO: estimate the noise
        estimated_noise = self.network(noisy_input, t, label)

        # TODO: compute the loss (either L1, or L2 loss)
        loss = F.l1_loss(estimated_noise, noise)
        return loss
    
    @torch.no_grad()
    def recover_sample(self, noisy_sample: torch.Tensor, estimated_noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # TODO: apply the sample recovery strategy of the DDIM
        alpha_bar_t = self.var_scheduler.alpha_bars[timestep]
        
        if timestep == 0:
            alpha_bar_t_1 = torch.tensor(1.0, dtype=alpha_bar_t.dtype, device=alpha_bar_t.device)
        else:
            alpha_bar_t_1 = self.var_scheduler.alpha_bars[timestep - 1]
        beta_t = self.var_scheduler.betas[timestep]
        #prediction_of_X0= ( (noisy_sample - (((1-alpha_bar_t)**0.5)*estimated_noise) )/ (alpha_bar_t)**0.5)
        prediction_of_X0 = (noisy_sample - (1 - alpha_bar_t).sqrt() * estimated_noise) / alpha_bar_t.sqrt()
        direction_to_Xt = ((1-alpha_bar_t_1 )**0.5) * estimated_noise
        epsilon = torch.randn_like(direction_to_Xt)   
        #sample = (alpha_bar_t_1**0.5)* prediction_of_X0 + direction_to_Xt 
        sample = alpha_bar_t_1.sqrt() * prediction_of_X0  + ((1 - alpha_bar_t_1)).sqrt()  * estimated_noise

        return sample
    @torch.no_grad()
    def generate_sample(self, num_samples: int, device: torch.device=torch.device('cuda'), labels: torch.Tensor=None):
        if labels is not None and self.network.num_classes is not None:
            assert len(labels) == num_samples, 'Error: number of labels should be the same as number of samples!'
            labels = labels.to(device)
        elif labels is None and self.network.num_classes is not None:
            labels = torch.randint(0, self.network.num_classes, [num_samples,], device=device)
        else:
            labels = None

        # TODO: apply the iterative sample generation of the DDPM
        sample = torch.randn((num_samples,1,32,32), device = device)
        for t in reversed(range(0,self.var_scheduler.num_steps)):
            time = torch.full((num_samples,),t, dtype=torch.long, device=device)  # Convert timestep to tensor
            estimated_noise = self.network(sample, time, labels)
            sample = self.recover_sample(sample, estimated_noise, t)
        return sample