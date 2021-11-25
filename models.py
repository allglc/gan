from torch import nn



class MLP(nn.Module):
    
    def __init__(self, input_dim=28*28, output_dim=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        y = self.net(x)
        return y


class Discriminator_MNIST(nn.Module):

    def __init__(self, hidden_dim=16, image_channels=1):
        super().__init__()
        kernel_size = 4
        stride = 2
        self.dicriminate = nn.Sequential(
            # input size: input_channels x 28 x 28
            nn.Conv2d(image_channels, hidden_dim, kernel_size, stride, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: HIDDEN_DIM x 13 x 13
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size, stride, bias=False),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: HIDDEN_DIM*2 x 5 x 5
            nn.Conv2d(hidden_dim*2, 1, kernel_size, stride, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dicriminate(x)


class Generator_MNIST(nn.Module):

    def __init__(self, z_dim=100, hidden_dim=64, image_channels=1):
        super().__init__()
        self.z_dim = z_dim
        self.generate = nn.Sequential(
            # input size: Z_DIM
            nn.ConvTranspose2d(self.z_dim, hidden_dim*4, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(inplace=True),
            # state dim: HIDDEN_DIM*4 x 3 x 3
            nn.ConvTranspose2d(hidden_dim*4, hidden_dim*2, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(inplace=True),
            # state dim: HIDDEN_DIM*2 x 6 x 6
            nn.ConvTranspose2d(hidden_dim*2, hidden_dim, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # state dim: HIDDEN_DIM x 13 x 13
            nn.ConvTranspose2d(hidden_dim, image_channels, kernel_size=4, stride=2, bias=False),
            # state dim: 28 x 28
            nn.Tanh()
        )
            
    def forward(self, x):
        return self.generate(x)


# disc = Discriminator()
# x0 = train_data[0][0].view(1, 1, 28, 28)
# disc(x0).shape

# gen = Generator()
# x0 = torch.randn(1,10,1,1)
# gen(x0).shape

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.z_dim = nz
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)