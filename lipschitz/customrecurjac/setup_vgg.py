## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import numpy as np
import os
import pickle
import gzip
import urllib.request
import imageio
import cv2
import Config


def pre_proc(img):
    img_resize = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
    # img_BGR = img_resize[...,::-1]
    # img_CHW = np.around(img_BGR / 255.0, decimals=12)
    return img_resize


def read_face_from_aligned(file_list):
    result = []
    useless = file_list[0]
    for file_name in file_list:
        face = imageio.imread(file_name)
        face = pre_proc(face)
        result.append(face)
    # result = np.array(result)
    return result


def extract_data(demographic='black', num_samples=10, target=None, label_dir=None, exact_sample=None):
    data = []
    labels = []
    new_labels = []
    one_hot = []
    if label_dir is not None:
        with open(os.path.join(Config.DATA, '{}/finetune-{}.txt'.format(label_dir, label_dir)), 'r') as infile:
          for line in infile:
            labels.append(line.strip())
    ROOT = os.path.join(Config.DATA, 'newsmallvgg')
    if target is not None:
        target_name = labels[int(target)]
        path = os.path.join(ROOT, 'newsmallvgg-val-align-160', target_name)
        file_list = []
        if exact_sample is not None:
            print(path)
            samples = [os.listdir(path)[exact_sample]]
            num_samples = 1
        else:
            samples = os.listdir(path)[:num_samples]
        for file in samples:
            file_list.append(os.path.join(path, file))
        data.extend(read_face_from_aligned(file_list))
        new_labels.extend([target] * num_samples)
        data = np.array(data)
        for l in new_labels:
            temp = np.zeros(8631)
            temp[int(target)] = 1
            one_hot.append(temp)
        one_hot = np.array(one_hot)
    else:
        path = os.path.join(ROOT, demographic)
        for person in os.listdir(path):
            file_list = []
            if exact_sample is not None:
                samples = [os.listdir(path, person)[exact_sample]]
                num_samples = 1
            else:
                samples = os.listdir(path, person)[:num_samples]
            for file in samples:
                file_list.append(os.path.join(path, person, file))
            data.extend(read_face_from_aligned(file_list))
            new_labels.extend([person] * num_samples)
        data = np.array(data)
        for l in new_labels:
            temp = np.zeros(8631)
            temp[labels.index(l)] = 1
            one_hot.append(temp)
        one_hot = np.array(one_hot)
    return data, one_hot


class VGG_DATA:
    def __init__(self, demographic='black', num_samples=10, target=None, label_dir=None, exact_sample=None):
        # train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
        # train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        self.test_data, self.test_labels = extract_data(demographic, num_samples, target, label_dir, exact_sample)
        
        # VALIDATION_SIZE = 5000
        
        # self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        # self.validation_labels = train_labels[:VALIDATION_SIZE]
        # self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        # self.train_labels = train_labels[VALIDATION_SIZE:]
