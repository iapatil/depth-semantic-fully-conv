import torch
import torch.utils.data as data
import os
import os.path
from scipy.ndimage import imread
import numpy as np
from torchvision import utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ListDataset(data.Dataset):
    def __init__(self,data_dir,listing,input_transform=None,target_depth_transform=None,
                target_labels_transform=None,co_transform=None,random_scale = None):

        self.data_dir = data_dir
        self.listing = listing
        #self.depth_imgs = depth_imgs
        self.input_transform = input_transform
        self.target_depth_transform = target_depth_transform
        self.target_labels_transform = target_labels_transform
        self.co_transform = co_transform

    def __getitem__(self, index):
        img_name = self.listing[index]

        input_dir,target_depth_dir,target_label_dir = self.data_dir

        input_im, target_depth_im,target_label_im = imread(os.path.join(input_dir,img_name)),\
                                                    imread(os.path.join(target_depth_dir,img_name[:-3]+'png')),\
                                                    imread(os.path.join(target_label_dir,img_name[:-3]+'png'))


        if self.co_transform is not None:
            input_im, target_depth_im,target_label_im = self.co_transform(input_im,target_depth_im,target_label_im)

        if self.input_transform is not None:
            input_im = self.input_transform(input_im)

        if self.target_depth_transform is not None :
            target_depth_im = self.target_depth_transform(target_depth_im)

        if self.target_labels_transform is not None :
            target_label_im = self.target_labels_transform(target_label_im)

        input_rgb_im = input_im
        input_depth_im  = torch.cat((target_depth_im,target_depth_im,target_depth_im),dim = 0)
        target_im = target_label_im

        return input_rgb_im,input_depth_im,target_im

    def __len__(self):
        return len(self.listing)
