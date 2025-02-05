import os

import torch
import torch.nn.functional as F
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.4.1: Implement LSGAN loss for discriminator.
    Do not use discrim_interp, interp, lamb. They are placeholders for Q1.5.
    """
    labels_real = torch.ones_like(discrim_real)
    labels_fake = torch.zeros_like(discrim_fake)
    loss = 0.5 * F.mse_loss(discrim_real, labels_real) + \
           0.5 * F.mse_loss(discrim_fake, labels_fake)
    return loss


def compute_generator_loss(discrim_fake):
    """
    TODO 1.4.1: Implement LSGAN loss for generator.
    """
    labels_real = torch.ones_like(discrim_fake)
    loss = 0.5 * F.mse_loss(discrim_fake, labels_real)
    return loss

if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_ls_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.4.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=100,#256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
