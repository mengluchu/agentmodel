#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:43:55 2021

@author: menglu
"""

''' Weighted aggregation, gaussian kernel'''
from matplotlib import pyplot as plt
from scipy import signal 


def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d

ag = gkern (3, 1)

signal.convolve(ag,ag, mode = 'valid') # valid does not pad. 
plt.imshow(gkern(23))

''' 
for understanding, longer code without importing signal. 
def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_kernel(size, sigma=1, verbose=False):
 
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
 
    kernel_2D *= 1.0 / kernel_2D.max()
 
    if verbose:
        plt.imshow(kernel_2D, interpolation='none',cmap='gray')
        plt.title("Image")
        plt.show()
 
    return kernel_2D
ag= gaussian_kernel(3)
plt.imshow(ag)
'''
''' 
#torch way 
#import torch
#import torch.nn.functional as F

#a= torch.Tensor.float(torch.from_numpy(ag))
#b= torch.Tensor.float(torch.from_numpy(ag))
#agg= F.conv2d(Variable(a.view(1,1,3,3)),Variable(b.view(1,1,3,3)))
#agg.item()
'''
