# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:52:54 2021

@author: Utility functions
"""

import os, random, time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import xml.etree.ElementTree as ET
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

'''
Help
https://www.kaggle.com/tthien/shanghaitech-a-test-density-gen
https://github.com/CommissarMa/MCNN-pytorch/blob/master/my_dataloader.py

MAE = sqrt(2/pi)*sqrt(MSE)  ## Holds true when bias is zero or when errors follow normal distribution with zero mean, and constant variance
'''


def collate_fn():
    '''
    Collate function for batching
    '''
    
    return None



    

def seed_everything(seed):
    ''' SEED Everything '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True 
seed=42
seed_everything(seed=seed)


##### Show sample test image
def display_sample(img, gt):
    '''Display a single sample from dataset '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,3))
    img = img.permute(1,2,0)
    ax1.imshow(img) # image
    ax2.imshow(gt.squeeze(0), cmap='jet') # GT Local
    ax1.set_title('Original image', fontweight='bold', fontstretch='ultra-expanded')
    ax2.set_title(f'Actual count: {gt.sum():.0f}', fontweight='bold', fontstretch='ultra-expanded')  
    # plt.savefig(figures+ f'/IMG_{i-1}.jpg', dpi=300)
    print('Image displayed .. \N{smiling face with sunglasses}')
    return fig
    
##### Show sample test image
def display_prediction(img, gt, et):
    ''' Display a single prediction on dataset'''
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
    ax1.imshow(img.permute(1,2,0)) # image
    # ax2.imshow(gt.permute(1,2,0), cmap='jet') # GT
    ax2.imshow(gt.squeeze(0), cmap='jet') # Colab
    # ax3.imshow(gt_h.squeeze(0).detach().permute(1,2,0), cmap='jet') # Pred
    ax3.imshow(et.squeeze(0).detach().cpu().squeeze(0), cmap='jet') # Colab
    ax1.set_title('Original image', fontsize=22, fontweight='medium', fontstretch='ultra-expanded')
    ax2.set_title(f'Count: {gt.sum():.0f}', fontsize=22, fontweight='medium', fontstretch='ultra-expanded')  
    ax3.set_title(f'Count: {et.sum():.0f}', fontsize=22, color='red', fontweight='medium', fontstretch='ultra-expanded')
    plt.tight_layout()
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    return fig
    
def plot_loss(history):
    ''' Plot loss from history '''
    plt.figure(figsize=(5,4))
    plt.plot(history['train_loss'], '-', lw=2, c='k', ms=6, mfc='k', mec='w', alpha=0.8, label='Train')
    plt.plot(history['val_loss'], ':', lw=2, c='b', ms=6, mfc='b', mec='w', alpha=0.8, label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend(loc=0, frameon=False)
    # plt.savefig(figures + '/loss_CCCnet2.pdf', dpi=300)
    plt.show()
    
    
def plot_loss_tb(net, images, labels):
    '''
    Plot Loss curve in tensorboard
    '''
    fig = plt.figure(figsize=(12, 48))
    
    return fig


##########################################################
##   PyTorch Visualization
##########################################################

def plot_kernels(tensor, ncols):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // ncols
    fig = plt.figure(figsize=(ncols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,ncols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
##############  To run
# from torch_utils import plot_kernels
# mm = model.double()
# filters = mm.modules
# body_model = [i for i in mm.children()][0]
# layer1 = body_model[0]
# tensor = layer1.weight.data.cpu().numpy()
# plot_kernels(tensor, ncols=10)
    
    

def calc_game(output, target, L=0):
    ''' Grid Mean Absolute Error (GAME) '''
    output = output[0][0]
    target = target[0]
    H, W = target.shape
    ratio = H / output.shape[0]
    output = cv2.resize(output, (W, H), interpolation=cv2.INTER_CUBIC) / (ratio*ratio)

    assert output.shape == target.shape

    # eg: L=3, p=8 p^2=64
    p = pow(2, L)
    abs_error = 0
    square_error = 0
    for i in range(p):
        for j in range(p):
            output_block = output[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]
            target_block = target[i*H//p:(i+1)*H//p, j*W//p:(j+1)*W//p]

            abs_error += abs(output_block.sum()-target_block.sum().float())
            square_error += (output_block.sum()-target_block.sum().float()).pow(2)

    return abs_error # , square_error

##################
##  Model Info
# https://github.com/sovrasov/flops-counter.pytorch
# https://github.com/Lyken17/pytorch-OpCounter
###################


def benchmark(model, input_shape, precision='fp32', nwarmup=50, nruns=1000):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if precision=='fp16':
        input_data = input_data.half()
        
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        print('Wait : running model over all inputs ...')
        for i in range(1, nruns+1):
            start_time = time.time()
            pred_loc  = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            # if i%10==0:
            #     print('Iteration %d/%d, avg batch time %.2f ms'%(i, nruns, np.mean(timings)*1000))
    
    throughput = (input_shape[0]/np.mean(timings)) # images/second
    inference_time = np.mean(timings)*1000  # in ms
    
    # print("Input shape:", input_data.size())
    print('tests', len(timings))
    print('Average inference time: %.2f milisecond'%inference_time)
    print('Average throughput: %.2f images/second'%throughput)
    return throughput, inference_time


#########################################

def model_parameters(model):
    '''
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
    '''
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return parameters


def model_size(model):
    '''
    Alternate methods: http://jck.bio/pytorch_estimating_model_size/
    '''
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_Mb = (param_size + buffer_size) / 1024**2
    # print('model size: {:.3f}MB'.format(size_Mb))
    return size_Mb


##############  Custom Metric Monitor
## https://albumentations.ai/docs/examples/pytorch_classification/

def save_checkpoint(epoch, model, optimizer, train_loss, val_loss, val_metric, filename):
    """
    Function to save the trained model to disk.
    """
    torch.save({
        'epoch'     :  epoch + 1,
        'train_loss':  train_loss,
        'val_loss'  :  val_loss,
        'val_metric':  val_metric,
        'state_dict':  model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        filename)
    
    
def load_checkpoint(model, optimizer, device, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print(f'Checkpoint exists (Loading..) {filename}')
        checkpoint    = torch.load(filename, map_location=device)
        start_epoch   = checkpoint['iteration']
        train_loss    = checkpoint['train_loss']
        val_loss      = checkpoint['val_loss']
        val_metric    = checkpoint['val_metric']
        # print('check', type(checkpoint['state_dict']))
        if isinstance(model, nn.DataParallel):
            # model = nn.DataParallel(model)
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
            
            
        # model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'===> loaded checkpoint {filename} epoch {start_epoch} ... MAE: {val_metric:.2f}')
    else:
        print("===> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, val_metric
    

#####################################################################################
def img_to_patches(img, patch_size):
    '''
    Convert image to non-overlapping patches
    
    Argument
        img          :  A 3D tensor (RGB image, greyscale image, or density map)
        patch_size   :  Desired patch size
    
    Return
        patches      : Non-overlapping oatches of size=patch_size
        
    '''
    assert img.shape[0] <= 3, f'Received image of shape: {img.shape}, accept image of shape: [channel, height, width]'
    p = patch_size
    c = img.shape[0]  # Channels (3 for RGB image, 1 for greyscale image or density map
    
    kc, kh, kw = c, p, p  # kernel size
    dc, dh, dw = c, p, p  # stride should be equal to kernel size for non-overlapping
    
    img = nn.functional.pad (img,
             (img.size(2)%kw // 2, img.size(2)%kw // 2,  
              img.size(1)%kh // 2, img.size(1)%kh // 2, 
              img.size(0)%kc // 2, img.size(0)%kc // 2))
    
    # print('Padded', img.shape)
    # patches = img.unfold(0, kc, dc).unfold(1, kh, dh).unfold(2, kw, dw)
    patches = img.unfold(0, kc, dc).unfold(1, kh, dh).unfold(2, kw, dw)
    unfold_shape = patches.shape
    # print('unfold shape', unfold_shape)
    patches = patches.contiguous().view(-1, kc, kh, kw)
    return unfold_shape, patches