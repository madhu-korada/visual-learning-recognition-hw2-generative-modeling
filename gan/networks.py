import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSampleConv2D(torch.jit.ScriptModule):
    # TODO 1.1: Implement nearest neighbor upsampling + conv layer

    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        # TODO 1.1: Setup the network layers
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size, padding=padding)
        
    # @torch.jit.unused
    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling
        # 1. Repeat x channel wise upscale_factor^2 times
        # 2. Use pixel shuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle)
        # to form a (batch x channel x height*upscale_factor x width*upscale_factor) output
        # 3. Apply convolution and return output
        print(x.shape)
        x_repeated = x.repeat(1, self.upscale_factor**2, 1, 1)
        print(x_repeated.shape)
        output_pixelsuffle = F.pixel_shuffle(x_repeated, self.upscale_factor)
        print(output_pixelsuffle.shape)
        return self.conv(output_pixelsuffle)
        

class DownSampleConv2D(torch.jit.ScriptModule):
    # TODO 1.1: Implement spatial mean pooling + conv layer

    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        # TODO 1.1: Setup the network layers
        self.downscale_ratio = downscale_ratio
        self.conv = nn.Conv2d(input_channels, n_filters, kernel_size, padding=padding)


    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling
        # 1. Use pixel unshuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#torch.nn.PixelUnshuffle)
        # to form a (batch x channel * downscale_factor^2 x height x width) output
        # 2. Then split channel wise into (downscale_factor^2xbatch x channel x height x width) images
        # 3. Average across dimension 0, apply convolution and return output
        output_pixelunshuffle = F.pixel_unshuffle(x, self.downscale_ratio)
        output_split = output_pixelunshuffle.view(-1, x.shape[1], x.shape[2], x.shape[3])
        output = torch.mean(output_split, dim=0)
        return self.conv(output)

class ResBlockUp(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        # TODO 1.1: Setup the network layers
        self.layers = nn.Sequential(
            nn.BatchNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            UpSampleConv2D(n_filters, kernel_size, n_filters=n_filters, padding=1)
        )
        self.upsample_residual = UpSampleConv2D(input_channels, kernel_size=1, n_filters=n_filters, padding=0)
        
    # @torch.jit.unused
    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Make sure to upsample the residual before adding it to the layer output.
        residual = x
        x = self.layers(x)
        out = self.upsample_residual(residual) + x
        return out

class ResBlockDown(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Downsampler.
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        # TODO 1.1: Setup the network layers
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size, padding=1),
            nn.ReLU(),
            DownSampleConv2D(n_filters, kernel_size, n_filters=n_filters, padding=1)
        )
        self.downsample_residual = DownSampleConv2D(input_channels, kernel_size=1, n_filters=n_filters, padding=0)

    # @torch.jit.unused
    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through self.layers and implement a residual connection.
        # Make sure to downsample the residual before adding it to the layer output.
        residual = x
        x = self.layers(x)
        out = self.downsample_residual(residual) + x
        return out


class ResBlock(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        # TODO 1.1: Setup the network layers
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(input_channels, n_filters, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        )

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        residual = x
        x = self.layers(x)
        out = residual + x
        return out


class Generator(torch.jit.ScriptModule):
    # TODO 1.1: Impement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        # TODO 1.1: Setup the network layers
        self.linear = nn.Linear(128, 2048, bias=True)
        self.layers = nn.Sequential(
            ResBlockUp(input_channels=128, n_filters=128),
            ResBlockUp(input_channels=128, n_filters=128),
            ResBlockUp(input_channels=128, n_filters=128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    # @torch.jit.unused
    @torch.jit.script_method
    def forward_given_samples(self, z):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        x = self.linear(z)
        x = x.view(-1, 128, 4, 4)
        out = self.layers(x)
        return out
    
    # @torch.jit.unused
    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents and forward through the network.
        rand_samples = torch.randn(n_samples, 128)
        out = self.forward_given_samples(rand_samples)
        return out
        

class Discriminator(torch.jit.ScriptModule):
    # TODO 1.1: Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (3): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        # TODO 1.1: Setup the network layers
        self.layers = nn.Sequential(
            ResBlockDown(input_channels=3, n_filters=128),
            ResBlockDown(input_channels=128, n_filters=128),
            ResBlock(input_channels=128, n_filters=128),
            ResBlock(input_channels=128, n_filters=128),
            nn.ReLU(),
        )
        self.linear = nn.Linear(128, 1, bias=True)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to sum across the image dimensions after passing x through self.layers.
        x = self.layers(x)
        x = x.sum(dim=(2,3))
        out = self.linear(x)
        return out

if __name__ == "__main__":
    print("Network Testings: ")
    rand_input = torch.randn(100, 128)
    rand_output = torch.randn(1024, 128)
    
    model_gen = Generator()
    print(model_gen)
    model_dis = Discriminator()
    print(model_dis)
    # print(model.forward().shape)
    
    # mdoel_res = ResBlockUp(input_channels=128, n_filters=128)
    # print(mdoel_res)
    # model_res_down = ResBlockDown(input_channels=3, n_filters=128)
    # print(model_res_down)
    rand_upsample = torch.randn(1, 128, 4, 4)
    # print(UpSampleConv2D(128, 3, 128)(rand_upsample).shape)
    
    upsample = torch.jit.script(model_gen, example_inputs=rand_input)
    # print(upsample(rand_upsample))
    