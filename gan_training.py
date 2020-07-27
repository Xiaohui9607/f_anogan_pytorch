import argparse
import os
import numpy as np
import math
import sys
import random
import torchvision.transforms as transforms
from torchvision.utils import save_image
from dataloader.dataloader import load_data
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from networks import Generator, Discriminator
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval betwen image samples")
parser.add_argument("--dataroot", default="", help="path to dataset")
parser.add_argument("--dataset", default="cifar10", help="folder | cifar10 | mnist")
parser.add_argument("--abnormal_class", default="airplane", help="Anomaly class idx for mnist and cifar datasets")
parser.add_argument("--device", default="cuda", help="device: cuda | cpu")
parser.add_argument("--out", default="ckpts", help="checkpoint directory")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.out, exist_ok=True)
img_shape = (opt.channels, opt.img_size, opt.img_size)


# Loss weight for gradient penalty
lambda_gp = 10

# Initialize generator and discriminator
generator = Generator(dim = 64, zdim=opt.latent_dim, nc=opt.channels)
discriminator = Discriminator(dim = 64, zdim=opt.latent_dim, nc=opt.channels)

generator.to(opt.device)
discriminator.to(opt.device)

# Configure data loader
dataloader = load_data(opt)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if opt.device == 'cuda' else torch.FloatTensor

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# ----------
#  Training
# ----------

batches_done = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader.train):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        fake_imgs = generator(z)
        # Loss measures generator's ability to fool the discriminator
        # Train on fake images`
        fake_validity = discriminator(fake_imgs)
        g_loss = -torch.mean(fake_validity)

        g_loss.backward()
        optimizer_G.step()

        if batches_done % opt.sample_interval == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader.train), d_loss.item(), g_loss.item())
            )
            save_image(fake_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

        batches_done += 1

    torch.save(generator.state_dict(), os.path.join(opt.out, 'G_epoch{}.pt'.format(epoch)))
    torch.save(discriminator.state_dict(), os.path.join(opt.out, 'D_epoch{}.pt'.format(epoch)))
