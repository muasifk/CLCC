
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





    

def validate(val_dl, model, criterion, device):
    model.eval()
    val_loss_epoch   = 0.0
    val_metric_epoch = 0.0


    for i, data in enumerate(val_dl):
        img, gt, score = data[0].to(device), data[1].to(device), data[2].to(device) # Read a single batch
        with torch.no_grad():
            et    = model(img)
            
        batch_loss        = criterion(gt, et) 
        val_loss_epoch   += batch_loss.item() 
        val_metric_epoch += abs(gt.sum() - et.sum())         
        
    val_loss_epoch     = val_loss_epoch/len(val_dl.dataset) # find average over all batches
    val_metric_epoch   = val_metric_epoch/len(val_dl.dataset) # find average over all batches
    return val_loss_epoch, val_metric_epoch
