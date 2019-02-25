# here is the training part
import torch
import numpy as np
from utils import *
from dataset import *
from noise2noise import Noise2Noise
from torch.utils.data import DataLoader

device = torch.cuda.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

# TODO
# validation batch_size, 
# true_image dataset extraction loop
# image_summary [index] : noise2noise.py : summary.py
config = {
    'device':device,
    'dataset':'gaussian',
    'batch_size_train':4,
    'batch_size_valid':1,
    'num_workers':4,
    'learning_rate':0.001,
    'checkpoint_dir': './models_trained/.',
    'checkpoint_filename': 'latest.pt',
    'root_dir':'/home/turing/Documents/BE/',
    'log_dir': './log/',
    'data_folder':'./data/',
    'num_epochs': 300,
}

trainloader = datasetLoader(config, 'train')
validloader = datasetLoader(config, 'valid')

NET = Noise2Noise(config, trainloader, validloader,trainable=True)

NET.load_model(config)

NET.train(config)