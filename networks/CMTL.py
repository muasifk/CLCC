

'''
CMTL 
Source: https://github.com/svishwa/crowdcount-cascaded-mtl/blob/master/src/models.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.nn import Conv2d, FC


class CMTL(nn.Module):
    '''
    CMTL: Implementation of CNN-based Cascaded Multi-task Learning of High-level Prior and Density, AVSS 2017
    Estimation for Crowd Counting (Sindagi et al.)
    '''
    def __init__(self, bn=False, num_classes=10):
        super(CMTL, self).__init__()
        
        self.num_classes = num_classes        
        self.base_layer = nn.Sequential(nn.Conv2d(3, 16, 9, padding='same'),  ## Asif changed channels: 1->3                                    
                                        nn.Conv2d(16, 32, 7, padding='same'))
        
        self.hl_prior_1 = nn.Sequential(nn.Conv2d( 32, 16, 9, padding='same'),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(16, 32, 7, padding='same'),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(32, 16, 7, padding='same'),
                                     nn.Conv2d(16, 8,  7, padding='same')) 
                
        self.hl_prior_2 = nn.Sequential(nn.AdaptiveMaxPool2d((32,32)),
                                        nn.Conv2d( 8, 4, 1, padding='same'))
        
        self.hl_prior_fc1 = nn.Linear(4*1024,512)
        self.hl_prior_fc2 = nn.Linear(512,256)
        self.hl_prior_fc3 = nn.Linear(256, self.num_classes)
        
        
        self.de_stage_1 = nn.Sequential(nn.Conv2d( 32, 20, 7, padding='same'),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(20, 40, 5, padding='same'),
                                     nn.MaxPool2d(2),
                                     nn.Conv2d(40, 20, 5, padding='same'),
                                     nn.Conv2d(20, 10, 5, padding='same'))
        
        self.de_stage_2 = nn.Sequential(nn.Conv2d( 18, 24, 3, padding='same'),
                                        nn.Conv2d( 24, 32, 3, padding='same'),                                        
                                        nn.ConvTranspose2d(32,16,4, stride=2, padding=1, output_padding=0, bias=True),
                                        nn.PReLU(),
                                        nn.ConvTranspose2d(16,8,4, stride=2, padding=1, output_padding=0, bias=True),
                                        nn.PReLU(),
                                        nn.Conv2d(8, 1, 1, padding='same'))
        
    def forward(self, img_tensor):
        x_base = self.base_layer(img_tensor)
        x_hlp1 = self.hl_prior_1(x_base)
        x_hlp2 = self.hl_prior_2(x_hlp1)
        x_hlp2 = x_hlp2.view(x_hlp2.size()[0], -1) 
        x_hlp = self.hl_prior_fc1(x_hlp2)
        x_hlp = F.dropout(x_hlp, training=self.training)
        x_hlp = self.hl_prior_fc2(x_hlp)
        x_hlp = F.dropout(x_hlp, training=self.training)
        x_cls = self.hl_prior_fc3(x_hlp)        
        x_den = self.de_stage_1(x_base)        
        x_den = torch.cat((x_hlp1,x_den),1)
        x_den = self.de_stage_2(x_den)
        return x_den, x_cls

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)