import os

import torch
from utils import get_args

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.5.1: Implement WGAN-GP loss for discriminator.
    loss = E[D(fake_data)] - E[D(real_data)] + lambda * E[(|| grad wrt interpolated_data (D(interpolated_data))|| - 1)^2]
    """
    labes_real = torch.ones_like(discrim_real)
    grad = torch.autograd.grad(outputs=discrim_interp, inputs=interp, \
                               grad_outputs=labes_real, create_graph=True, \
                               retain_graph=True)[0]
    grad = grad.view(grad.size(0), -1)
    grad_penality = torch.mean((torch.norm(grad, dim=1) - 1) ** 2)
    
    loss = torch.mean(discrim_fake) - \
           torch.mean(discrim_real) + \
           lamb * grad_penality
    return loss


def compute_generator_loss(discrim_fake):
    """
    TODO 1.5.1: Implement WGAN-GP loss for generator.
    loss = - E[D(fake_data)]
    """
    loss = - torch.mean(discrim_fake)
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_wgan_gp/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.5.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=32,#256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
