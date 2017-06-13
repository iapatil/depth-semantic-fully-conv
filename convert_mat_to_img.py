#encoding: utf-8
import os
import numpy as np
import h5py
from PIL import Image
import random
import collections


def convert_nyu(path):
    print("load dataset: %s" % (path))
    f = h5py.File(path)

    labels_selected = [21,11,3,157,5,83,19,28,59,88,64,7,80,36,42,89,169,119,122,\
        143,141,4,85,17,172,15,135,123,26,45,331,158,124,24,144,136,55]
    labels_count = np.zeros(894)
    #assert(len(labels_selected) == 37)

    if not os.path.exists(os.path.join("data", "nyu_datasets_changed","input")):
        os.makedirs(os.path.join("data", "nyu_datasets_changed","input"))

    if not os.path.exists(os.path.join("data", "nyu_datasets_changed","target_depths")):
        os.makedirs(os.path.join("data", "nyu_datasets_changed","target_depths"))

    if not os.path.exists(os.path.join("data", "nyu_datasets_changed","labels_38")):
        os.makedirs(os.path.join("data", "nyu_datasets_changed","labels_38"))

    #trains = []
    #max_ = 0
    for i, (image, depth, label) in enumerate(zip(f['images'], f['depths'], f['labels'])):
    #for i,label in enumerate(f['labels']):

        ra_image = image.transpose(2, 1, 0)
        ra_depth = depth.transpose(1, 0)
        ra_label = label.transpose(1,0)

        h,w = ra_label.shape

        #ra_colored_label = np.zeros((h,w,3))
        ra_colored_label = np.zeros((h,w))

        # flag = 0
        # color = [(0,0,0),(0,0,255),(255,0,0),(0,255,0),(255,255,0),(255,0,255), #magenta
        # (192,192,192), #silver
        # (128,128,128), #gray
        # (128,0,0) ,#maroon
        # (128,128,0) ,#olive
        # (0,128,0) ,#green
        # (128,0,128), # purple
        # (0,128,128) , # teal
        # (65,105,225) , #royal blue
        # (255,250,205) , #lemon chiffon
        # (255,20,147) , #deep pink
        # (218,112,214) , #orchid]
        # (135,206,250) , #light sky blue
        # (127,255,212),  #aqua marine
        # (0,255,127) , #spring green
        # (255,215,0) , #gold
        # (165,42,42) , #brown
        # (148,0,211) , #violet
        # (210,105,30) , # chocolate
        # (244,164,96),  # sandy brown
        # (240,255,240),  #honeydew
        # (112,128,144), (64,224,208) ,(100,149,237) ,(30,144,255),(221,160,221) ,(205,133,63)]



        values = np.unique(ra_label)
        for flag in values:
            selected_label = np.where(labels_selected == flag)[0]
            if len(selected_label) > 0:
                ra_colored_label[np.where(ra_label==flag)] = selected_label[0]+1

            else:
                ra_colored_label[np.where(ra_label==flag)] = 0
        # j = 1
        # ra_colored_label =  np.zeros((h,w))
        # for flag in labels_selected:
        #     if(len(np.where(ra_label==flag)[0]) > 0):
        #         ra_colored_label[np.where(ra_label==flag)] = j
        #     j = j+1
        #assert(False)

        #assert(j==values.size)

        #re_depth = (ra_depth/np.max(ra_depth))*255.0
        re_depth = (ra_depth-np.min(ra_depth))/(np.max(ra_depth)-np.min(ra_depth))*255.0

        #ra_colored_label = (ra_colored_label-np.min(ra_colored_label))/(np.max(ra_colored_label)-np.min(ra_colored_label))*255.0
        #
        image_pil = Image.fromarray(np.uint8(ra_image))
        depth_pil = Image.fromarray(np.uint8(re_depth))
        labels_pil = Image.fromarray(np.uint8(ra_colored_label))

        image_name = os.path.join("data", "nyu_datasets_changed","input", "%05d.jpg" % (i))
        image_pil.save(image_name)
        #
        depth_name = os.path.join("data", "nyu_datasets_changed","target_depths" ,"%05d.png" % (i))
        depth_pil.save(depth_name)
        #
        label_name = os.path.join("data","nyu_datasets_changed", "labels_38","%05d.png" % (i))
        labels_pil.save(label_name)

    #     trains.append((image_name, depth_name, label_name))
    #
    # random.shuffle(trains)
    #
    # with open('train.csv', 'w') as output:
    #     for (image_name, depth_name, label_name) in trains:
    #         output.write("%s,%s,%s" % (image_name, depth_name, label_name))
    #         output.write(" n")

if __name__ == '__main__':
    current_directory = os.getcwd()
    nyu_path = 'nyu_depth_v2_labeled.mat'
    convert_nyu(nyu_path)
