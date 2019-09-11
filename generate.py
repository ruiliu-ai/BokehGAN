"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import numpy as np
import scipy
from collections import OrderedDict
import time
from imageio import imsave

import data
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
from util.visualizer import Visualizer
from util import util

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.cuda().eval()

# test
for i, data_i in enumerate(dataloader):
    time1 = time.time()
    mask = data_i['oriinstance']
    orilabel = data_i['orilabel'].cuda()
    gen0, fmask = model(torch.cat((data_i['label'], data_i['instance']), 1).cuda(), mode='inference')
    gen = F.interpolate(gen0, data_i['shape'], mode='bilinear')
    mask = F.interpolate(fmask, data_i['shape'], mode='bilinear')

    gen_comb = gen * (1-mask.cuda()) + (2*orilabel-1) * mask.cuda()
    print(time.time() - time1)

    name = data_i['path_lb'][0].strip().split('/')[-1]

    im_np = util.tensor2im(gen_comb)
    if len(im_np.shape) >= 4:
        im_np = im_np[0]
    scipy.misc.toimage(im_np).save('results/{}'.format(name), format='png')

