import os
import torch
import numpy as np

import sys
from torchvggish.hubconf import vggish
vggish = vggish(pretrained=True)
vggish.eval()

def extract_vgg(dirname):
    os.system('mkdir {}/afeat'.format(dirname))
    vnames = os.listdir(os.path.join(dirname, 'audio'))
    for vname in vnames:
        sname = vname[:-4] + '.npy'
        feat = vggish.forward(os.path.join(dirname, 'audio', vname))
        print(feat.shape, sname)
        np.save(os.path.join(dirname, 'afeat', sname), feat.detach().cpu().numpy())

# extract_vgg('Train')
extract_vgg('Test/Test_Noise')