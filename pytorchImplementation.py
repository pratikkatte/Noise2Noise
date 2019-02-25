
# coding: utf-8

# In[1]:


import torch
import numpy as np
from utils import *
from dataset import *
from noise2noise import Noise2Noise
from torch.utils.data import DataLoader


# In[2]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[3]:


# TODO
# validation batch_size, 
# true_image dataset extraction loop
# image_summary [index] : noise2noise.py : summary.py
config = {
    'device':device,
    'dataset':'gaussian',
    'batch_size_train':8,
    'batch_size_valid':1,
    'num_workers':4,
    'learning_rate':0.001,
    'checkpoint_dir': './models_trained/.',
    'checkpoint_filename': 'latest.pt',
    'root_dir':'.',
    'log_dir': './log/',
    'data_folder':'./data/',
    'num_epochs': 100,
}


# In[4]:


trainloader = datasetLoader(config, 'train')
validloader = datasetLoader(config, 'valid')


# In[5]:


NET = Noise2Noise(config, trainloader, validloader,trainable=True)


# In[6]:


NET.load_model(config)


# In[ ]:


NET.loss_size


# In[7]:


NET.train(config)

