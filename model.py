"""""
Define the DCGAN neural network structure in this model.py file, 
you can modify the neural network structure from it to build your own neural network model
"""""
#loading head file
import torch.nn as nn

# Defining a generator network
class NetG(nn.Module):
    def __init__(self, ngf, nz):
        super(NetG, self).__init__()
        # The input of layer1 is a 100x1x1 random noise, and the output size (ngf * 8) x4x4
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True)
        )
        # layer2 output size (ngf * 4) x8x8
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        # layer3 output size (ngf * 2) x16x16
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        # layer4 output size (ngf) x32x32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        # layer5 output size 3x96x96
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, 5, 3, 1, bias=False),
            nn.Tanh()
        )

    # Define forward propagation
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


# Define discriminator network
class NetD(nn.Module):
    def __init__(self, ndf):
        super(NetD, self).__init__()
        # Here,the original image data format is 3 x 96 x 96, and the output size is (ndf) x 90 x 90
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,ndf,kernel_size=5,stride=1,padding=0),#92,92
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=0),#90,90
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf, kernel_size=1, stride=1, padding=0),  # 90,90
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)

        )
        # layer2 output is (ndf*2) x 11 x 11
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=5, stride=2, padding=2),  # 45,45
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*2, kernel_size=3, stride=2, padding=0),  # 22,22
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*2, kernel_size=1, stride=2, padding=0),  # 11,11
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # layer3 output is (ndf*4) x 5 x 5
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=0),  # 5,5
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # layer4 output is (ndf*8) x 4 x 4
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf*4, ndf*8, kernel_size=2, stride=1, padding=0),  # 4,4
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer5 output is 1x1x1
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    # Defining forward propagation of NetD
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

