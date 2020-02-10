"""""
In order to be able to verify the trained generator model, 
here I will verify the robustness of the model 
through an example-by generating an image data and saving it in a local folder
"""""

import argparse
from model import NetG
import torch
from torch.autograd import Variable
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64)
opt = parser.parse_args()
if __name__ == '__main__':
    print(opt)
    net=NetG(64,100)
    net.load_state_dict(torch.load("./model_save/xxxx.pth"))
    data=torch.randn(100,1,1)
    data= Variable(data.unsqueeze(0))
    output=net(data)
    vutils.save_image(output.data,'./test.png' ,normalize=True)

