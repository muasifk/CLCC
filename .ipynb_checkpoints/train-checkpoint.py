
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import scipy
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import torch
import torchvision.transforms as transforms
# import albumentations as A
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as T
# import albumentations as A
from torchvision.transforms import ToTensor, Lambda, Compose
from tqdm.notebook import tqdm
from torch import nn






def train(train_dl, model, criterion, optimizer, lr_scheduler, device):
    model.train()
    train_loss_epoch = 0.0
    
    for i, data in enumerate(train_dl):
        img, gt, score  = data[0].to(device), data[1].to(device), data[2].to(device) # Read a single batch
        optimizer.zero_grad()  # sets gradients to zeros
        et = model(img) # predict the outputs (inputs is batch of images)
        batch_loss  = criterion(gt, et) # calculate loss (scalar value: mean or sum of losses for all images in the batch)
        train_loss_epoch += batch_loss.item() # add batch_loss to find cumulative epoch_loss which will be averaged later
        batch_loss.backward()  # Backpropagation
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
    train_loss_epoch = train_loss_epoch/len(train_dl.dataset) # average over the number of images to get mean error for thw whole epoch
    return train_loss_epoch
        