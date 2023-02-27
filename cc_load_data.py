
import numpy as np
import os
# import sys
# from pathlib import Path
# import re
from glob import glob
# import shutil
# import random
from sklearn.model_selection import train_test_split

def load_data(root_dir, ds_name):
    '''
    Inputs   : root_dir, ds_name
    Outputs  : train_img_paths, train_gt_paths, test_img_paths, test_gt_paths
    '''
    ####################################################################################
    if ds_name == 'DroneRGBT':
        '''
        Note: We do not have GTs for test set, Hence, we are loading only the train_data and splitting into train and test data.
        Update as required.
        '''
        train_img_dir = root_dir + '/Train/RGB/'
        train_gt_dir  = root_dir + '/Train/GT/'
        train_img_paths  = sorted(glob(train_img_dir + '*.jpg'))
        train_gt_paths   = sorted(glob(train_gt_dir + '*.xml'))
        train_img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        train_gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        
        # >>>>>  Deleting some corrupted labels
        del train_img_paths[1317]
        del train_gt_paths[1317]
        del train_img_paths[1228]
        del train_gt_paths[1228]
        del train_img_paths[1159]
        del train_gt_paths[1159]
        del train_img_paths[696]
        del train_gt_paths[696]
        
        #>>>>> Selecting first 1800 samples
        n = 1800 #1800  # take n images
        train_img_paths = train_img_paths[0:n]
        train_gt_paths  = train_gt_paths[0:n]
        
        train_img_paths, test_img_paths, train_gt_paths, test_gt_paths = train_test_split(train_img_paths, train_gt_paths, test_size=0.3, random_state=200)  # 42
        # val set
        val_img_paths, val_gt_paths = test_img_paths, test_gt_paths
        
        
    ####################################################################################    
    if ds_name == 'CARPK':
        '''
        CARPK dataset
        '''
        root_dir   = root_dir + '/datasets/CARPK_devkit/data'
        img_dir    = root_dir + '/Images'
        gt_dir     = root_dir + '/Annotations-mat'   ## pre-generated .mat files
        img_paths  = sorted(glob(img_dir + '/*.png'))
        gt_paths   = sorted(glob(gt_dir + '/*.mat'))  # pre-generated .mat files
        #############  Train/Test Split
        train_files   = open(root_dir + '/ImageSets/train.txt', 'r').read().splitlines()
        test_files    = open(root_dir + '/ImageSets/test.txt', 'r').read().splitlines()
        train_img_paths = [f'{img_dir}/{x}.png' for x in train_files]
        train_gt_paths  = [f'{gt_dir}/{x}.mat' for x in train_files]
        test_img_paths  = [f'{img_dir}/{x}.png' for x in test_files]
        test_gt_paths   = [f'{gt_dir}/{x}.mat' for x in test_files]
        # val set
        val_img_paths, val_gt_paths = test_img_paths, test_gt_paths
    
    
    ####################################################################################
    if ds_name == 'ShanghaiTechPartA' or ds_name == 'ShanghaiTechPartB':
        '''
        ShanghaiTech Part B
        '''
        train_img_dir   = root_dir + '/train_data/images'
        train_gt_dir    = root_dir + '/train_data/ground-truth'
        test_img_dir    = root_dir + '/test_data/images'
        test_gt_dir     = root_dir + '/test_data/ground-truth'
        train_img_paths = glob(train_img_dir + '/*.jpg')
        test_img_paths  = glob(test_img_dir + '/*.jpg')
        train_gt_paths  = glob(train_gt_dir + '/*.mat')
        test_gt_paths   = glob(test_gt_dir + '/*.mat')
        train_img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        test_img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        train_gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        test_gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        # val set
        val_img_paths, val_gt_paths = test_img_paths, test_gt_paths
    
    
    ####################################################################################
    if ds_name == 'UCF-QNRF':
        '''
        UCF-QNRF
        '''
        train_dir   = root_dir + '/Train'
        test_dir    = root_dir + '/Test'
        train_img_paths = glob(train_dir + '/*.jpg')
        test_img_paths  = glob(test_dir + '/*.jpg')
        train_gt_paths  = glob(train_dir + '/*.mat')
        test_gt_paths   = glob(test_dir + '/*.mat')
        train_img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        test_img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        train_gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        test_gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        
        # >>>>>  Deleting some corrupted samples
        del train_img_paths[1]
        del train_gt_paths[1]
        # del train_img_paths[1601]
        # del train_gt_paths[1601]
        # val set
        val_img_paths, val_gt_paths = test_img_paths, test_gt_paths
        
    ####################################################################################    
    if ds_name == 'JHU-Crowd':
        train_dir   = root_dir + '/train'
        val_dir    = root_dir + '/val'
        test_dir    = root_dir + '/test'
        train_img_paths = glob(train_dir + '/images/*.jpg')
        val_img_paths   = glob(val_dir + '/images/*.jpg')
        test_img_paths  = glob(test_dir + '/images/*.jpg')
        train_gt_paths  = glob(train_dir + '/gt/*.txt')
        val_gt_paths    = glob(val_dir + '/gt/*.txt')
        test_gt_paths   = glob(test_dir + '/gt/*.txt')
        train_img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        val_img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        test_img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        train_gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        val_gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        test_gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        
        ## Delete corrupt samples
        train = [465, 511, 646, 777, 812, 1110, 1810, 2203]
        test  = [27, 44, 598, 800, 855, 1116, 1448, 1524]
        train = train[::-1] # reverse order
        test  = test[::-1]  # reverse order
        for i in train:
            del train_img_paths[i]
            del train_gt_paths[i]
        for i in test:
            del test_img_paths[i]
            del test_gt_paths[i]
            
    
    ####################################################################################
    if ds_name == 'Mall':
        '''
        Mall Dataset
        '''
        img_paths   = sorted(glob(root_dir + '/frames/*.jpg'))    # List of paths, See sample print(len(filepaths))
        gt_paths    = sorted(glob(root_dir + '/ground-truth/*.mat'))    # List of paths, See sample print(len(filepaths))
        img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))

        # Train/Test split
        train_img_paths, test_img_paths, train_gt_paths, test_gt_paths = train_test_split(img_paths, gt_paths, test_size=0.3, random_state=42)
        # val set
        val_img_paths, val_gt_paths = test_img_paths, test_gt_paths
        
    if ds_name == 'Lusail':
        '''
        Mall Dataset
        '''
        img_paths   = sorted(glob(root_dir + '/frames/*.jpg'))    # List of paths, See sample print(len(filepaths))
        gt_paths    = sorted(glob(root_dir + '/gt/*.csv'))    # List of paths, See sample print(len(filepaths))
        img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))

        # Train/Test split
        train_img_paths, test_img_paths, train_gt_paths, test_gt_paths = train_test_split(img_paths, gt_paths, test_size=0.3, random_state=42)
        # val set
        val_img_paths, val_gt_paths = test_img_paths, test_gt_paths
        
    ####################################################################################
    if ds_name == 'UCF_CC_50':
        '''
        UCF-CC-50
        '''
        img_paths   = sorted(glob(root_dir + '/*.jpg'))    # List of paths, See sample print(len(filepaths))
        gt_paths    = sorted(glob(root_dir + '/*.mat'))    # List of paths, See sample print(len(filepaths))
        img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))

        # Train/Test split
        train_img_paths, test_img_paths, train_gt_paths, test_gt_paths = train_test_split(img_paths, gt_paths, test_size=0.3, random_state=42)
        train_img_paths, test_img_paths  = img_paths[1:], img_paths[1:]
        train_gt_paths, test_gt_paths    = gt_paths[1:], gt_paths[1:]
        print('=================== Important =============================')
        print('>>>>>>>>>>  Use K-fold cross-validation:')
        print('train_img_paths = test_img_paths   AND train_gt_paths, test_gt_paths')
        print('==========================================================')
        

    ####################################################################################    
    if ds_name == 'UCSD':
        '''
        UCSD
        '''
        assert 2==2, '>>>>> Still not implemented ...'
        # img_paths = glob(root_dir + '/*.jpg')
        # gt_paths  = glob(root_dir + '/*.mat')
        # img_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        # gt_paths.sort(key=lambda f: int('_'.join(filter(str.isdigit, f))))
        # # Train/Test split
        # train_img_paths, test_img_paths, train_gt_paths, test_gt_paths = train_test_split(img_paths, gt_paths, test_size=0.3, random_state=42)
        
        
        
    
    ### Print datasets stats
    if ds_name == 'JHU-Crowd':
        assert len(train_img_paths) + len(val_img_paths) + len(test_img_paths) == len(train_gt_paths) + len(val_gt_paths) + len(test_gt_paths), f'\33[91m \N{crying face} Error: No. of input images is not same as no. of labels'
        # if (len(train_img_paths)!= 0 and len(train_gt_paths)!= 0 and len(train_img_paths) == len(train_gt_paths)):
        print('\33[32m')
        print(f'>>>>>>>> {ds_name} Dataset is successfuly loaded .. \N{grinning face}')
        print('\33[36m')
        print('=========================================')
        print(f'Train data (img/gt)   :  {len(train_img_paths)} = {len(train_gt_paths)}')
        print(f'Val data (img/gt)     :  {len(val_img_paths)}  = {len(val_gt_paths)}')
        print(f'Test data (img/gt)    :  {len(test_img_paths)} = {len(test_gt_paths)}')
        print(f'Total data (img/gt)   :  {len(train_img_paths) + len(val_img_paths) + len(test_img_paths)} = {len(train_gt_paths) + len(val_gt_paths) + len(test_gt_paths)}')
        print('==========================================')                                                                             
        # else:
            # print(f'  >>>>>>>> {ds_name} dataset loading failed .. ') # 
        return train_img_paths, train_gt_paths, val_img_paths, val_gt_paths, test_img_paths, test_gt_paths
            
    else:
        # assert len(train_img_paths) + len(test_img_paths) == len(train_gt_paths) + len(test_gt_paths),
        # f'Error: No. of input images is not same as no. of labels'
        # if (len(train_img_paths)!= 0 and len(train_gt_paths)!= 0 and len(train_img_paths) == len(train_gt_paths)):
        print('\33[32m')
        print(f'>>>>>>>> {ds_name} Dataset is successfuly loaded .. \N{grinning face}')
        print('\33[36m')
        print('=========================================')
        print(f'Train data (img/gt)   :  {len(train_img_paths)} = {len(train_gt_paths)}')
        print(f'Test data (img/gt)    :  {len(test_img_paths)} = {len(test_gt_paths)}')
        print(f'Total data (img/gt)   :  {len(train_img_paths) + len(test_img_paths)} = {len(train_gt_paths) + len(test_gt_paths)}')
        print('==========================================')                                                                             
        # else:
        #     print(f' \33[91m \N{crying face} >>>>>>>> {ds_name} dataset loading failed .. ') # 
        return train_img_paths, train_gt_paths, val_img_paths, val_gt_paths, test_img_paths, test_gt_paths
        
        