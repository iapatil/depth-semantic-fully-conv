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

    labels_count = np.zeros(8941)


    for i, label in enumerate(f['labels']):

        ra_label = label.transpose(1,0)

        h,w = ra_label.shape

        unique_labels,unique_counts = np.unique(ra_label, return_counts= True)
        labels_count[unique_labels] += unique_counts

    sorted_labels = np.argsort(labels_count)
    sorted_values = np.sort(labels_count)
    print('Labels: ',sorted_labels[-10:])
    print('Counts: ',sorted_values[-10:])


if __name__ == '__main__':
    current_directory = os.getcwd()
    nyu_path = 'nyu_depth_v2_labeled.mat'
    convert_nyu(nyu_path)
