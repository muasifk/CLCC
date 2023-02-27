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
from train import train
from validate import validate


def trainer(parameters):
    '''
    My Custom method to train and validate a model
    '''
    ## Parameters
    model         = parameters['model']
    train_dl      = parameters['train_dl']
    val_dl        = parameters['val_dl']
    optimizer     = parameters['optimizer']
    criterion     = parameters['criterion']
    # logdir        = parameters['logdir']
    lr_scheduler  = parameters['lr_scheduler']
    device        = parameters['device']
    n_epochs      = parameters['n_iterations']
    checkpoint    = parameters['checkpoint']
    
    val_metric_prev  = parameters['val_metric']
    val_loss_prev    = np.Inf  # min val loss
    
    ## Returns
    epochs     = []
    train_loss = []
    val_loss   = []
    val_metric = []
    
    for epoch in range(1, n_epochs+1):
        print(f"Epoch {epoch} of {n_epochs}")
        
        train_loss_epoch = train(train_dl, model, criterion, optimizer, lr_scheduler, device)
        val_loss_epoch, val_metric_epoch = validate(val_dl, model, criterion, device)
        
        ## Record stats
        train_loss.append(train_loss_epoch)
        val_loss.append(val_loss_epoch)
        val_metric.append(val_metric_epoch)
        
        print(f'Epoch:{epoch}  ==> \
        Train/Valid Loss: {train_loss_epoch:.4f} / {val_loss_epoch:.4f} ... MAE={val_metric_epoch:.2f}')
        
        if checkpoint is not None:
            if (val_metric_epoch < val_metric_prev):
                print(f'Validation MAE decreased ({val_metric_prev:.2f} --> {val_metric_epoch:.2f}):  Saving model ...')
                checkpoints = {'epoch'     : epoch, 'train_loss' : train_loss_epoch, 'val_loss'   : val_loss_epoch, 'val_metric' : val_metric_epoch, 
                               'state_dict' : model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                if isinstance(model, nn.DataParallel):
                    'Converting to single GPU before saving...'
                    checkpoints['state_dict'] : model.module.state_dict() 
                torch.save(checkpoints, checkpoint)
                
                # Update loss and MAE to compate in next epoch
                val_loss_prev   = val_loss_epoch
                val_metric_prev = val_metric_epoch
        
        epochs.append(epoch)
    
    ## Return a "history" dictionary of lists of "epochs, losses, metric values"
    history = {'epochs':epochs, 'train_loss':train_loss, 'val_loss':val_loss, 'val_metric':val_metric}
    
    if epoch == n_epochs:
        print('Training completed .. \n')      
    return history