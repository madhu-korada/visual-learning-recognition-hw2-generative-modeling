import argparse
import torch
import torchvision
from cleanfid import fid
from matplotlib import pyplot as plt


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    # TODO 1.2: Generate and save out latent space interpolations.
    # 1. Generate 100 samples of 128-dim vectors
    # 2. Do so by linearly interpolating for 10 steps across each of the first two dimensions between -1 and 1.
    # 3. torch.linspace to generate the interpolation steps.
    # 4. Keep the rest of the z vector for the samples to be some fixed value (e.g. 0).
    # 5. Forward the samples through the generator.
    # 6. Save out an image holding all 100 samples.
    # 7. Use torchvision.utils.save_image to save out the visualization.
    
    
    # With mean 0 and standard deviation 1.
    sample = torch.normal(mean=0, std=1, size=(1, 128)).cuda()         # 1. use repeat, why??
    samples = sample.repeat(100, 1)                                    # 1. shape: (100, 128)
    # samples = torch.zeros(100, 128)
    
    interp_steps = torch.linspace(-1, 1, 10)                           # 3. shape: (10,)
    grid_x, grid_y = torch.meshgrid(interp_steps, interp_steps)        # 3. shape: (10, 10)
    grid_x, grid_y = grid_x.flatten(), grid_y.flatten()                # 3. shape: (100,)
    
    samples[:, 0], samples[:, 1] = grid_x, grid_y                      # 4. shape: (100, 128)                     
                
    gen_images = gen.forward_given_samples(samples)                    # 5. shape: (100, 3, 32, 32)
    torchvision.utils.save_image(gen_images, path, nrow=10)            # 6. image with 10 rows and 10 columns


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Utils Testings: ")
    interpolate_latent_space(torch.jit.load("generator.pt"), "interpolations.png")
