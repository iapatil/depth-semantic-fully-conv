# Preprocess the NYU Depth v2 (labeled) dataset into RGB, ground-truth depth and semantic label images.
# Pre-requisite:  The file 'nyu_depth_v2_labeled.mat' should be in the same directory as this file.
# Output: The RGB, depth and semantic label images will be stored in 'data/input', 'data/target_depths' and 'data/labels_38' folders respectively.

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

    # Select the most dominant 37 labels
    labels_selected = [21,11,3,157,5,83,19,28,59,88,64,7,80,36,42,89,169,119,122,\
        143,141,4,85,17,172,15,135,123,26,45,331,158,124,24,144,136,55]
    labels_count = np.zeros(894)
    assert(len(labels_selected) == 37)

    # Create all folders, if they don't exist.
    if not os.path.exists(os.path.join("data", "nyu_datasets_changed","input")):
        os.makedirs(os.path.join("data", "nyu_datasets_changed","input"))

    if not os.path.exists(os.path.join("data", "nyu_datasets_changed","target_depths")):
        os.makedirs(os.path.join("data", "nyu_datasets_changed","target_depths"))

    if not os.path.exists(os.path.join("data", "nyu_datasets_changed","labels_38")):
        os.makedirs(os.path.join("data", "nyu_datasets_changed","labels_38"))


    for i, (image, depth, label) in enumerate(zip(f['images'], f['depths'], f['labels'])):
        # Transpose images to correct shape
        ra_image = image.transpose(2, 1, 0)
        ra_depth = depth.transpose(1, 0)
        ra_label = label.transpose(1,0)

        # Preprocess label images to get values in the range[0,37]
        # All labels except the dominant 37-classes are labeled as '0'
        h,w = ra_label.shape

        ra_colored_label = np.zeros((h,w))

        values = np.unique(ra_label)
        for flag in values:
            selected_label = np.where(labels_selected == flag)[0]
            if len(selected_label) > 0:
                ra_colored_label[np.where(ra_label==flag)] = selected_label[0]+1

            else:
                ra_colored_label[np.where(ra_label==flag)] = 0

        # Normalize depth images to the range [0,255]
        re_depth = (ra_depth-np.min(ra_depth))/(np.max(ra_depth)-np.min(ra_depth))*255.0

        # Save the images in different folders
        image_pil = Image.fromarray(np.uint8(ra_image))
        depth_pil = Image.fromarray(np.uint8(re_depth))
        labels_pil = Image.fromarray(np.uint8(ra_colored_label))

        image_name = os.path.join("data", "nyu_datasets_changed","input", "%05d.jpg" % (i))
        image_pil.save(image_name)

        depth_name = os.path.join("data", "nyu_datasets_changed","target_depths" ,"%05d.png" % (i))
        depth_pil.save(depth_name)

        label_name = os.path.join("data","nyu_datasets_changed", "labels_38","%05d.png" % (i))
        labels_pil.save(label_name)


if __name__ == '__main__':
    current_directory = os.getcwd()
    nyu_path = 'nyu_depth_v2_labeled.mat'
    convert_nyu(nyu_path)
