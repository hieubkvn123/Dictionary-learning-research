import os
import tqdm
import time
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

from utils import get_device
from models import LISTA, DenoiseNetwork

def train_lista(train_loader, input_dim=784, head_layers=1, L=3, norm=1, lambda_=1e-4, bn=True, 
        initializer='eye', lr=0.0001, epochs=10, out_plot_file=None, out_plot_dir='media'):
    '''
        Returns :
            - lista : Trained LISTA network for extracting sparse representation.
            - network : Head network used to perform pre-text tasks.

        Parameters :
            - train_loader : Train data loader.
            - input_dim : Input dimension.
            - head_layers : Number of linear - relu - batchnorm modules in head network.
            - L : Number of ISTA modules in lista network.
            - norm : Order of the regularization term. 1 for L1, 2 for L2.
            - lambda_ : Regularizer coefficient.
            - lr : Learning rate.
            - epochs : Number of training iterations.
            - out_plot_file : Output file for loss plot.
    '''
    # Check if output dir for loss plot exists
    if(not os.path.exists(out_plot_dir)):
        os.mkdir(out_plot_dir)

    if(out_plot_file is not None):
        out_plot_file = os.path.join(out_plot_dir, out_plot_file)

    # Print out the configuration
    print(f"[CONFIG] Input dimension : {input_dim}")
    print(f"[CONFIG] Number of head layers : {head_layers}")
    print(f"[CONFIG] Number of ISTA modules : {L}")
    print(f"[CONFIG] Regularization method : L{norm}")
    print(f"[CONFIG] Regularization coefficient : {lambda_}")
    print(f"[CONFIG] Initialization method : {initializer}")
    print("=======================================================")

    # Losses storage
    norm_losses = []
    mse_losses = []

    # Initialize networks
    gpu, device = get_device()
    lista = LISTA(input_dim, L=L, batch_norm=bn).to(device)
    network = DenoiseNetwork(input_dim, num_layers=head_layers).to(device)

    # Optimizers and loss function
    optimizer1 = optim.Adam(lista.parameters(), lr=lr)
    optimizer2 = optim.Adam(network.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Initialize networks weights
    def init_weights(m):
        if(isinstance(m, nn.Linear)):
            if(initializer == 'eye'):
                nn.init.eye_(m.weight)
            elif(initializer == 'zeros'):
                nn.init.constant_(m.weight, 0.1)
            elif(initializer == 'xavier'):
                nn.init.xavier_uniform_(m.weight)
            elif(initializer == 'normal'):
                nn.init.normal_(m.weight, std=0.1)
            else:
                raise Exception(f'Invalid initializer {initializer} ... ')

            m.bias.data.fill_(0.01)

    lista.apply(init_weights)
    network.apply(init_weights)

    try:
        for i in range(epochs):
            print(f'Epoch #[{i+1}/{epochs}]\n====================')
            time.sleep(1.0)

            with tqdm.tqdm(total=len(train_loader)) as pbar:
                for j, ((noisy, normal), labels) in enumerate(train_loader):
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()

                    noisy = torch.flatten(noisy, 1).to(device)
                    normal = torch.flatten(normal, 1).to(device)

                    x = lista(noisy)
                    reg = 0
                    if(norm is not None):
                        reg = lambda_ * torch.mean(torch.norm(x, p=norm, dim=1))
                    outputs = network(x)
                    mse = criterion(outputs, normal)
                    loss = mse + reg

                    norm_losses.append(reg)
                    mse_losses.append(mse)
                    loss.backward()

                    # Report epoch loss
                    pbar.set_postfix({
                        'MSE loss' : f'{mse.item():.4f}',
                        'L1 reg' : f'{reg:.4f}'
                    })

                    optimizer1.step()
                    optimizer2.step()

                    # Update progress bar
                    pbar.update(1)

    except KeyboardInterrupt:
        print('[INFO] Training halted ... ')

    # Visualize loss
    fig, ax1 = plt.subplots(figsize=(15, 4))
    ax2 = ax1.twinx()

    ax1.plot(list(range(1, len(mse_losses) + 1)), mse_losses, '-', label='MSE losses', color='blue')
    ax2.plot(list(range(1, len(norm_losses) + 1)), norm_losses, '-r', label=f'L{norm} losses', color='orange')

    ax1.set_ylabel('MSE Losses', color='blue')
    ax2.set_ylabel('L1 regularization', color='orange')
    ax1.set_xlabel('Epochs')

    fig.legend(loc='upper right')
    plt.title("Training Losses")
    
    if(out_plot_file is None):
        plt.show() 
    else:
        plt.savefig(out_plot_file)

    return lista, network

def visualize_result(loader, lista, network, out_file=None, out_dir='media'):
    '''
        Parrameters :
            - loader : Sample data loader.
            - lista : The sparse representation extractor network.
            - network : The head network.
    '''
    
    # Check if media folder exists
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    if(out_file is not None):
        out_file = os.path.join(out_dir, out_file)

    gpu, device = get_device()
    fig, ax = plt.subplots(4, 4, figsize=(16, 16))

    for i in range(4):
        (noisy, normal), labels = next(iter(loader))
        noisy = torch.flatten(noisy, 1).to(device)
        normal = torch.flatten(normal, 1).to(device)

        sparse_rep = lista(noisy)
        output = network(sparse_rep)
        output = output.reshape(8, 28, 28)
        if(not gpu):
            ax[i][0].imshow(noisy.detach().numpy().reshape(-1, 28, 28)[i], cmap='gray')
            ax[i][1].imshow(sparse_rep.detach().numpy().reshape(-1, 28, 28)[i], cmap='gray')
            ax[i][2].imshow(output.detach().numpy().reshape(-1, 28, 28)[i], cmap='gray')
            ax[i][3].imshow(normal.detach().numpy().reshape(-1, 28, 28)[i], cmap='gray')
        else:
            ax[i][0].imshow(noisy.cpu().detach().numpy().reshape(-1, 28, 28)[i], cmap='gray')
            ax[i][1].imshow(sparse_rep.cpu().detach().numpy().reshape(-1, 28, 28)[i], cmap='gray')
            ax[i][2].imshow(output.cpu().detach().numpy().reshape(-1, 28, 28)[i], cmap='gray')
            ax[i][3].imshow(normal.cpu().detach().numpy().reshape(-1, 28, 28)[i], cmap='gray')

        ax[i][0].set_title('Noisy image')
        ax[i][1].set_title('Sparse representation')
        ax[i][2].set_title('Predicted denoised image')
        ax[i][3].set_title('Ground truth')

    if(out_file is None):
        plt.show()
    else:
        plt.savefig(out_file)
