import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import torch
import torchvision
from torchvision import datasets, transforms

from models import AddGaussianNoise, LISTA, DenoiseNetwork
from train import train_lista, visualize_result

# Arguments definition
parser = ArgumentParser() 

parser.add_argument('--head-layers', required=False, type=int, default=1, help='Number of layers in the head denoising network')
parser.add_argument('--num-ista', required=False, type=int, default=3, help='Number of ISTA modules in the LISTA network')
parser.add_argument('--regularization', required=False, type=int, default=1, help='Regularization term - 1 for L1 and 2 for L2')
parser.add_argument('--lambda', required=False, type=float, default=1e-4, help='Regularization coefficient lambda')
parser.add_argument('--initializer', required=False, type=str, default='eye', help='Initialization methods : ["xavier", "eye", "zeros", "normal"]')
parser.add_argument('--bn', required=False, type=bool, default=True, help='Whether to apply BN in ISTA modules')
args = vars(parser.parse_args())

# Create output file format
out_dir = './media'
if(args['bn']):
    out_lost_file = f'LOSS_head-{args["head_layers"]}_ista-{args["num_ista"]}_reg-L{args["regularization"]}_init-{args["initializer"]}_bnorm.png'
    out_sample_pred_file = f'PRED_head-{args["head_layers"]}_ista-{args["num_ista"]}_reg-L{args["regularization"]}_init-{args["initializer"]}_bnorm.png'
else:
    out_lost_file = f'LOSS_head-{args["head_layers"]}_ista-{args["num_ista"]}_reg-L{args["regularization"]}_init-{args["initializer"]}.png'
    out_sample_pred_file = f'PRED_head-{args["head_layers"]}_ista-{args["num_ista"]}_reg-L{args["regularization"]}_init-{args["initializer"]}.png'

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0.0, 0.8)
])

mnist_data = torchvision.datasets.MNIST(
    '/tmp/mnist',
    download=True,
    train=True,
    transform=transform
)

loader = torch.utils.data.DataLoader(
    mnist_data,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

# Get a sample batch
(noisy, normal), labels = next(iter(loader))
noisy = torch.flatten(noisy, 1)
normal = torch.flatten(normal, 1)
input_dim = noisy.shape[1]

# Start training
lista, network = train_lista(loader, input_dim=input_dim, initializer=args['initializer'],
        head_layers=args['head_layers'], L=args['num_ista'], norm=args['regularization'],
        lambda_=args['lambda'], bn=True, out_plot_dir=out_dir, out_plot_file=out_lost_file)
visualize_result(loader, lista, network, out_dir=out_dir, out_file=out_sample_pred_file)

