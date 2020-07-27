from torch import  nn


class Generator(nn.Module):
    def __init__(self, dim, zdim, nc):
        super(Generator, self).__init__()
        self.nc = nc
        self.dim = dim
        preprocess = nn.Sequential(
            nn.Linear(zdim, 4 * 4 * 4 * dim),
            nn.BatchNorm1d(4 * 4 * 4 * dim),
            nn.ReLU(True),
        )

        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * dim, 2 * dim, 2, stride=2),
            nn.BatchNorm2d(2 * dim),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * dim, dim, 2, stride=2),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(dim, nc, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * self.dim, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        return output.view(-1, self.nc, 32, 32)


class Discriminator(nn.Module):
    def __init__(self, dim, zdim, nc, out_feat=False):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.dim = dim
        main = nn.Sequential(
            nn.Conv2d(nc, dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
        )
        self.out_feat=out_feat
        self.main = main
        self.linear = nn.Linear(4*4*4*dim, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.dim)
        if self.out_feat:
            return output
        output = self.linear(output)
        return output


class Encoder(nn.Module):
     def __init__(self,dim, zdim, nc):
         super(Encoder, self).__init__()
         self.dim = dim
         main = nn.Sequential(
            nn.Conv2d(nc, dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, 2 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * dim, 4 * dim, 3, 2, padding=1),
            nn.LeakyReLU(),
            )
         self.main = main
         self.linear = nn.Linear(4*4*4*dim, zdim)

     def forward(self, input):
         output = self.main(input)
         output = output.view(-1, 4*4*4*self.dim)
         output = self.linear(output)
         return output
