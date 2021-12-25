import numpy as np
import matplotlib.pyplot as plt

# Torch dependencies
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as VF

import torchvision
from torchvision import datasets, transforms

# Utility function to visualize image batches in grids
def show(imgs, title=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
        
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = VF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        
    fig.suptitle(title)
    plt.show()

def get_device():
    '''
        Returns:
            - gpu : Whether GPU is available or not.
            - device : The device available on this machine.
    '''
    # Initialize device.
    device = torch.device('cpu')
    gpu = False
    
    if(torch.cuda.is_available()):
        device = torch.device('cuda')
        gpu = True
        print('[INFO] GPU is available ...')
    else:
        print('[INFO] GPU is not available ... ')
        
    return gpu, device


