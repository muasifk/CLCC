
import numpy as np
from scipy.io import loadmat
import xml.etree.ElementTree as ET

def sort_data_bycount(train_img_paths, train_gt_paths, ds_name):
    '''
    Sort data paths
    '''
    ## Step 1: Find count for each image
    count = []
    for gt_path in train_gt_paths:
        if ds_name == 'ShanghaiTechPartA' or ds_name == 'ShanghaiTechPartB':
            mat    = loadmat(gt_path)   # when gt_path is provided
            pos    = mat.get("image_info")[0][0][0][0][0]
        if ds_name == 'Mall':
            mat  = loadmat(gt_path)
            pos  = mat['annot']
        if ds_name == 'CARPK':
            mat      = loadmat(gt_path)   # when gt_path is provided
            pos      = mat.get("annot")
        if ds_name == 'DroneRGBT':
            tree = ET.parse(gt_path)
            root = tree.getroot()
            pos  = [ [int(x.text), int(y.text)] for x,y in zip(root.iter('x'), root.iter('y'))] ## x is width, y is height
        
        count.append(len(pos))
    
    ### Step 2: Sort by count
    count, train_img_paths, train_gt_paths = zip(*sorted(zip(count, train_img_paths, train_gt_paths)))
    count = list(count)
    train_img_paths = list(train_img_paths)
    train_gt_paths  = list(train_gt_paths)
    return count, train_img_paths, train_gt_paths