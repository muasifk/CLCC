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
from torch.utils.data import Subset
from torchvision import datasets
import torchvision.transforms as T
# import albumentations as A
from torchvision.transforms import ToTensor, Lambda, Compose
from tqdm.notebook import tqdm
from torch import nn
from train import train
from validate import validate
from get_pacing_function import get_pacing_function

def trainer(parameters):
    '''
    My Custom method to train and validate a model
    '''
    ## Parameters
    model         = parameters['model']
    train_ds      = parameters['train_ds']
    val_ds        = parameters['val_ds']
    optimizer     = parameters['optimizer']
    criterion     = parameters['criterion']
    batch_size    = parameters['batch_size']
    lr_scheduler  = parameters['lr_scheduler']
    device        = parameters['device']
    n_epochs      = parameters['n_epochs']
    checkpoint    = parameters['checkpoint']
    
    val_metric_prev  = parameters['val_metric']
    val_loss_prev    = np.Inf  # min val loss
    
    ## Returns
    iterations = []
    train_loss = []
    val_loss   = []
    val_metric = []
    
    
    ###########  CL Set up
    N               = len(train_ds)
    iter_per_epoch  = N//batch_size  
    M               = iter_per_epoch # 32  # No. of samples to feed to the model in each iterations (256)
    n_iterations    = (N// batch_size+1)*n_epochs
    all_sum         = N/(n_iterations*(n_iterations+1)/2)
    pre_iterations  = 0
    startIter       = 0
    pacing_f        = 'linear'
    ##  b is fraction data at start of the training
    ##  a is the fraction of training needed for the pacing function to reach full training set
    ## standard training when a=0, b=1
    a               = 0.01 # 0.01,0.1,0.2, 0.4, 0.8, 1.6
    b               = 0.3 # 0.025,0.1,0.2, 0.4, 0.8
    pacing_function = get_pacing_function(n_iterations, N, a, b, pacing_f)
    startIter_next  = pacing_function(0) # <=======================================
    
    
    order     = np.arange(0, len(train_ds))
    print(f' M {M}, startIter {startIter}, startIter_next {startIter_next}')
    
    train_dss = Subset(train_ds, list(order[startIter:max(startIter_next, M)])) #  max => min
    train_dls = torch.utils.data.DataLoader(train_dss, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True) 
    dataiter  = iter(train_dls)
    step      = 0
    
    
    #=======================================
    #   Start Training
    #=======================================
    print (f'1: iter data between {startIter} and {startIter_next} with Pacing => {pacing_f}')
    for step in range(1, n_iterations+1):
        print(f"Step {step} of {n_iterations}")
        step += 1
        
        print('Loading samples', len(train_dss))
        #################  Training Loop
        model.train()
        train_loss_epoch = 0.0
        for data in train_dls:
            img, gt, score  = data[0].to(device), data[1].to(device), data[2].to(device) # Read a single batch
            optimizer.zero_grad() 
            et = model(img) 
            batch_loss  = criterion(gt, et) 
            train_loss_epoch += batch_loss.item() 
            batch_loss.backward()  
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
        # Record loss
        train_loss_epoch = train_loss_epoch/len(train_dls.dataset) # average over the number of images to get mean error for thw whole epoch
        
        # If we hit the end of the dynamic epoch build a new data loader
        pre_iterations = step 
        # print('startIter_next', startIter_next)
        # print('N', N)
        if startIter_next <= N:            
            startIter_next = pacing_function(step)# <=======================================
            # print ("%s iter data between %s and %s w/ Pacing %s and LEARNING RATE %s "%(step, startIter, startIter_next, pacing_f, optimizer))
            print (f'{step}: iter data between {startIter} and {startIter_next} with Pacing => {pacing_f}')
            train_dss  = Subset(train_ds, list(order[startIter:max(startIter_next, M)])) # max => min
            train_dls  = torch.utils.data.DataLoader(train_dss, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        
        #################  Validation Loop
        val_dl    = torch.utils.data.DataLoader(val_ds, batch_size=1, num_workers=2, pin_memory=True, shuffle=False)
        val_loss_epoch, val_metric_epoch = validate(val_dl, model, criterion, device)
        
        ## Record stats
        train_loss.append(train_loss_epoch)
        val_loss.append(val_loss_epoch)
        val_metric.append(val_metric_epoch)
        
        print(f'Step:{step}  ==> \
        Train/Valid Loss: {train_loss_epoch:.4f} / {val_loss_epoch:.4f} ... MAE={val_metric_epoch:.2f}')
        
        if checkpoint is not None:
            if (val_metric_epoch < val_metric_prev):
                print(f'Validation MAE decreased ({val_metric_prev:.2f} --> {val_metric_epoch:.2f}):  Saving model ...')
                checkpoints = {'iteration'     : step, 'train_loss' : train_loss_epoch, 'val_loss'   : val_loss_epoch, 'val_metric' : val_metric_epoch, 
                               'state_dict' : model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                if isinstance(model, nn.DataParallel):
                    'Converting to single GPU before saving...'
                    checkpoints['state_dict'] : model.module.state_dict() 
                torch.save(checkpoints, checkpoint)
                
                # Update loss and MAE to compate in next iteration
                val_loss_prev   = val_loss_epoch
                val_metric_prev = val_metric_epoch
        
        iterations.append(step)
    
    ## Return a "history" dictionary of lists of "iterations, losses, metric values"
    history = {'iterations':iterations, 'train_loss':train_loss, 'val_loss':val_loss, 'val_metric':val_metric}
    
    if step == n_iterations:
        print('Training completed .. \n')      
    return history