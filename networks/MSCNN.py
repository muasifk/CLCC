# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 12:52:54 2021

@author: Utility functions
"""
import numpy as np
import torch
import torch.nn as nn

# class MSCNN(nn.Module):
#     '''
#     MSCNN: Multi-scale Convolution Neural Networks for Crowd Counting, ICIP 2017
#     All the paddings in the model are added by me.
#     '''
#     def __init__(self, load_weights=False):
#         super(MSCNN, self).__init__()
        
#         self.msb9 = nn.Sequential(
#             nn.Conv2d(3,16,9, padding=4),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16,32,9),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32,64,9),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2))
        
#         self.msb7 = nn.Sequential(
#             nn.Conv2d(3,16,7),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16,32,7),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32,64,7),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2))
        
#         self.msb5 = nn.Sequential(
#             nn.Conv2d(3,16,5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16,32,5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32,64,5),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2))
        
#         self.msb3 = nn.Sequential(
#             nn.Conv2d(3,16,3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),    
#             nn.Conv2d(16,32,3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),        
#             nn.Conv2d(32,64,3),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2))
        
#         self.fuse=nn.Sequential(
#             nn.Conv2d(30,1,1,padding=0),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(30,1,1,padding=0),
#             nn.ReLU(inplace=True))
        
#         if not load_weights:
#             self._initialize_weights()
            
#     def forward(self,x):
#         x1=self.msb9(x)
#         x2=self.msb7(x)
#         x3=self.msb5(x)
#         x4=self.msb3(x)
#         y=torch.cat((x1,x2,x3, x4),1)
#         y=self.fuse(y)
#         return y
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.01)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
            

##############   Model defined by me =============
class MSCNN(nn.Module):
    '''
    MSCNN: Multi-scale Convolution Neural Networks for Crowd Counting, ICIP 2017
    All the paddings in the model are added by me.
    '''
    def __init__(self, load_weights=False):
        super(MSCNN, self).__init__()
    
        self.fmap = nn.Sequential(
            nn.Conv2d(3, 64, 9, padding=4))
        
        self.msb1 = nn.Sequential(
            nn.Conv2d(64,16,9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,32,7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3, padding=1),
            nn.ReLU(inplace=True))
        
        self.ds1 = nn.Sequential(
            nn.MaxPool2d(2))
        
        self.mlp = nn.Sequential(
            nn.Conv2d(1,1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1,1,1),
            nn.ReLU(inplace=True))  
        
        if not load_weights:
            self._initialize_weights()
            
    def forward(self,x):
        x=self.fmap(x)
        x=self.msb1(x)
        x=self.ds1(x)
        x=self.mlp(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    