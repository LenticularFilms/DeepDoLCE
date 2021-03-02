import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import scipy.signal as signal

class Space2Features(nn.Module):

    def __init__(self,w_ceil:int):
        super(Space2Features, self).__init__()

        self.w_ceil = w_ceil

        self.eye_filter = nn.Parameter(torch.flip(torch.eye(self.w_ceil),(1,)).unsqueeze(1).unsqueeze(1),requires_grad=False)
        
        delay = torch.zeros(self.w_ceil,self.w_ceil,1,self.w_ceil)
        for i in range(0,self.w_ceil):
            delay[i,i,0,i] = 1.

        self.delay = nn.Parameter(delay,requires_grad=False)

    def forward(self,x,z):

        z_exp = F.conv2d(z, self.eye_filter, padding=(0, self.w_ceil-1))[...,0:z.shape[3]]

        x_exp = x * z_exp

        x_exp = torch.conv2d(x_exp, self.delay, padding=(0,self.w_ceil-1))[...,self.w_ceil-1:] 

        return x_exp


class RGBExtractor(nn.Module):

    def __init__(self,w_ceil:int,w:float):
        super(RGBExtractor, self).__init__()

        self.w_ceil = w_ceil
        self.w = w 

        self.window_size = 4

        enable_grad = True
        self.R = nn.Parameter(torch.zeros(1,1,int(w//self.window_size)),requires_grad=enable_grad)
        self.G = nn.Parameter(torch.zeros(1,1,int(w//self.window_size)),requires_grad=enable_grad)
        self.B = nn.Parameter(torch.zeros(1,1,int(w//self.window_size)),requires_grad=enable_grad)

        RGB_matrix = torch.zeros(3,1, w_ceil)
        RGB_matrix[0,0,int(np.round(w/4))] = 1
        RGB_matrix[1,0,int(np.round(w/2))] = 1
        RGB_matrix[2,0,int(np.round(3*w/4))] = 1

        self.RGB_matrix = nn.Parameter(RGB_matrix,requires_grad=False)

    def get_RGB_filter(self):

        R_sx = F.softmax(self.R,dim=2)
        G_sx = F.softmax(self.G,dim=2)
        B_sx = F.softmax(self.B,dim=2)

        R = torch.conv1d(self.RGB_matrix[0:1],R_sx,padding=int(self.w//self.window_size)//2)
        G = torch.conv1d(self.RGB_matrix[1:2],G_sx,padding=int(self.w//self.window_size)//2)
        B = torch.conv1d(self.RGB_matrix[2:3],B_sx,padding=int(self.w//self.window_size)//2)

        if int(self.w//self.window_size) % 2 == 0:
            R = R[:,:,0:-1]
            G = G[:,:,0:-1]
            B = B[:,:,0:-1]

        RGB_sx = torch.cat((R,G,B),dim = 0).squeeze(1)

        RGB_sx = RGB_sx.unsqueeze(2).unsqueeze(2)

        return RGB_sx

    def forward(self,x_exp,z):

        RGB_sx = self.get_RGB_filter()

        y =  torch.conv2d(x_exp, RGB_sx)

        RGB_reshape = (RGB_sx.squeeze(2).squeeze(2)).unsqueeze(1).unsqueeze(1)
        RGB_reshape = torch.flip(RGB_reshape,(3,))
        rgb_filter_image = torch.conv2d(z,RGB_reshape,padding=(0, self.w_ceil-1))[...,0:z.shape[3]]

        return y,rgb_filter_image
        

class RGBExtractor_fixed(nn.Module):

    def __init__(self,w_ceil:int,w:float):
        super(RGBExtractor_fixed, self).__init__()

        self.w_ceil = w_ceil
        self.w = w 

        self.space2Feature = Space2Features(self.w_ceil)

        RGB = torch.zeros(3, w_ceil, 1, 1)
        RGB[0,int(np.round(w/4)),0,0] = 1
        RGB[1,int(np.round(w/2)),0,0] = 1
        RGB[2,int(np.round(3*w/4)),0,0] = 1

        self.RGB = nn.Parameter(RGB,requires_grad=False)

    def get_RGB_filter(self):
        return self.RGB

    def autotune(self,x,z):

        diff_filter = torch.zeros((1,1,1,3))
        diff_filter[...,0] = -1
        diff_filter[...,2] = 1
        diff_filter = diff_filter.to(x.device)

        diff_x = F.conv2d(x,diff_filter,padding=(0,1))
        diff_x_3d = self.space2Feature(diff_x,z)
        profile = torch.abs(torch.sum((diff_x_3d).cpu().detach(),dim=(2,3)).squeeze())

        self.profile = profile 

        idx_first = torch.argmax(profile[0:int(self.w/3)])
        idx_last = torch.argmax(profile[int(3*self.w/4):]) + int(3*self.w/4)

        idx_green = (idx_first + idx_last) / 2 #self.w/2  #
        idx_red = (idx_first + idx_green) / 2 #self.w/3  #
        idx_blue = (idx_last + idx_green) / 2 #2*self.w/3 #

        with torch.no_grad():
            self.RGB.fill_(0.) # = 0 * self.RGB

            if np.ceil(idx_red) == idx_red:
                self.RGB[0,int(np.round(idx_red)),0,0] = 1.
            else:
                self.RGB[0,int(np.floor(idx_red)),0,0] = np.ceil(idx_red) - idx_red 
                self.RGB[0,int(np.floor(idx_red))+1,0,0] =  idx_red - np.floor(idx_red)


            if np.ceil(idx_green) == idx_green:
                self.RGB[1,int(np.round(idx_green)),0,0] = 1.
            else:
                self.RGB[1,int(np.floor(idx_green)),0,0] = np.ceil(idx_green) - idx_green
                self.RGB[1,int(np.floor(idx_green))+1,0,0] =  idx_green - np.floor(idx_green) 

            if np.ceil(idx_blue) == idx_blue:
                self.RGB[2,int(np.round(idx_blue)),0,0] = 1.
            else:
                self.RGB[2,int(np.floor(idx_blue)),0,0] = np.ceil(idx_blue) - idx_blue 
                self.RGB[2,int(np.floor(idx_blue))+1,0,0] =  idx_blue - np.floor(idx_blue)

        return

    def forward(self,x_exp,z):

        y =  torch.conv2d(x_exp, self.RGB)

        RGB_reshape = (self.RGB.squeeze(2).squeeze(2)).unsqueeze(1).unsqueeze(1)
        RGB_reshape = torch.flip(RGB_reshape,(3,))
        rgb_filter_image = torch.conv2d(z,RGB_reshape,padding=(0, self.w_ceil-1))[...,0:z.shape[3]]

        return y,rgb_filter_image


class LenticuleSpreader(nn.Module):

    def __init__(self,w_ceil:int):
        super(LenticuleSpreader, self).__init__()

        self.w_ceil = w_ceil

        spreader_filter = torch.zeros(3,3,1,self.w_ceil)
        spreader_filter[0,0,:,:] = 1
        spreader_filter[1,1,:,:] = 1
        spreader_filter[2,2,:,:] = 1
        self.spreader_filter = nn.Parameter(spreader_filter,requires_grad=False)

    def forward(self,x):

        if x.shape[1] == 3:
            y = torch.conv2d(x,self.spreader_filter,padding=(0,self.w_ceil-1))[:,:,:,0:x.shape[3]]
        elif x.shape[1] == 1:
            y = torch.conv2d(x,self.spreader_filter[0:1,0:1,:,:],padding=(0,self.w_ceil-1))[:,:,:,0:x.shape[3]]
        else:
            raise NotImplementedError

        return y


class ColorRestoration(nn.Module):

    def __init__(self,w,learnable=False,autotune=True):
        super(ColorRestoration, self).__init__()

        self.w = w 
        self.w_ceil = int(np.ceil(w))
        self.autotune = autotune
        self.learnable =learnable

        self.space2Feature = Space2Features(self.w_ceil)

        if learnable:
            self.RGB_extractor = RGBExtractor(self.w_ceil,self.w)
        else:            
            self.RGB_extractor = RGBExtractor_fixed(self.w_ceil,self.w)
        
        self.spreader = LenticuleSpreader(self.w_ceil)

    
    def forward(self,x,z):

        x_exp = self.space2Feature(x,z)

        if self.autotune and not self.learnable:
            self.RGB_extractor.autotune(x,z)

        x_RGB,rgb_filter_image = self.RGB_extractor(x_exp,z)

        y_unorm = self.spreader(x_RGB)

        z_norm = self.spreader(z)
        
        y = y_unorm / z_norm

        return {"y": y, "rgb_filter_image": rgb_filter_image} 




