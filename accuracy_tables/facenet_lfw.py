#!/usr/bin/env python
# coding: utf-8

# ### facenet-pytorch LFW evaluation
# This notebook demonstrates how to evaluate performance against the LFW dataset.

# In[1]:

from tensorflow import keras
import tensorflow as tf
from keras.models import load_model, model_from_json
from facenet_pytorch import MTCNN, fixed_image_standardization, training, extract_face
# InceptionResnetV1
from torch_inception_resnet_v1 import *
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms
import numpy as np
import os
import Config
import argparse
import csv
from tqdm import tqdm
from skimage.transform import resize
from inception_resnet_v1 import InceptionResNetV1
from PIL import Image
import pandas as pd
from sklearn import metrics

# -*- coding: utf-8 -*-
"""Inception-ResNet V1 model for Keras.
# Reference
http://arxiv.org/abs/1602.07261
https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py
https://github.com/myutwo150/keras-inception-resnet-v2/blob/master/inception_resnet_v2.py
"""


def model_wrapper(model,device):
    model.eval()
    model = model.to(device)
    def wrapped_model(x):
        #x_rs = np.transpose(x,[0,3,1,2])
        tensor_x = torch.tensor(x)
        tensor_x = tensor_x.type(torch.FloatTensor)
        tensor_x_gpu = tensor_x.to(device)
        tmp = model(tensor_x_gpu)
        return tmp.cpu().detach()
    return wrapped_model


def get_resnet18_pretrained(key='balanced'):

    # model =  keras.Sequential()
    # model.add(keras.layers.Permute((3,1,2)))
    if key == 'default_3':
        model = InceptionResNetV1(weights_path='fpfair/weights/facenet_vggface2_default_3_keras_weights.h5')
        return model
    elif key == 'default_4':
        model = InceptionResNetV1(weights_path='fpfair/weights/facenet_vggface2_default_4_keras_weights.h5')
        return model
    elif key == 'balance_sex_3':
        model = InceptionResNetV1(weights_path='fpfair/weights/facenet_vggface2_sex_balance_3_keras_weights.h5')
        return model
    elif key == 'balance_sex_4':
        model = InceptionResNetV1(weights_path='fpfair/weights/facenet_vggface2_sex_balance_4_keras_weights.h5')
        return model
    elif key == 'balance_race_3':
        model = InceptionResNetV1(weights_path='fpfair/weights/facenet_vggface2_race_balance_3_keras_weights.h5')
        return model
    elif key == 'balance_race_4':
        model = InceptionResNetV1(weights_path='fpfair/weights/facenet_vggface2_race_balance_4_keras_weights.h5')
        return model
    elif key == 'torch_Xu_facenet':
        model = InceptionResnetV1(classify=False, pretrained='vggface2')
        model.load_state_dict(torch.load(os.path.join('fpfair/weights/torch_trained_model_0'), map_location=torch.device('cpu')))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model_wrapper(model, device)
        return model
    elif key == 'vggface2_race_balanced_facenet':
        model = InceptionResnetV1(classify=False, num_classes=1842)
        model.load_state_dict(torch.load(os.path.join('fpfair/weights/torch_race_balanced_trained_model'), map_location=torch.device('cpu')))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model_wrapper(model, device)
        return model
    elif key == 'vggface2_sex_balanced_facenet':
        model = InceptionResnetV1(classify=False, num_classes=4866)
        model.load_state_dict(torch.load(os.path.join('fpfair/weights/torch_sex_balanced_trained_model'), map_location=torch.device('cpu')))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model_wrapper(model, device)
        return model
    elif key == 'vggface2_default_facenet':
        model = InceptionResnetV1(classify=False, pretrained='vggface2')
        model.load_state_dict(torch.load(os.path.join('fpfair/weights/torch_trained_model'), map_location=torch.device('cpu')))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model_wrapper(model, device)
        return model

args = Config.parse_arguments('embedding')
params = Config.set_parameters(attribute=args.attribute, all_flag=args.all_flag, uniform_flag=args.uniform_flag, folder=args.folder)


pairs_path = 'fpfair/pairs/{}/{}_pairs.txt'.format(params['folder'], params['folder'])
crop_dir = 'fpfair/{}/{}_cropped'.format(params['folder'], params['folder'])
crop_dir = params['align_dir']

if params['all_flag'] and params['uniform_flag']:
    pairs_path = 'fpfair/pairs/{}/{}_pairs_all_uniform_{}.txt'.format(params['folder'], params['folder'], params['attribute'])
    attr_dir = os.path.join('fpfair', params['folder'], params['folder'])
    string = '-all_uniform'
elif params['all_flag']:
    pairs_path = 'fpfair/pairs/{}/{}_pairs_all_{}.txt'.format(params['folder'], params['folder'], params['attribute'])
    attr_dir = os.path.join('fpfair', params['folder'], params['folder'])
    string = '-all'
elif params['uniform_flag']:
    pairs_path = 'fpfair/pairs/{}/{}_pairs_uniform_{}.txt'.format(params['folder'], params['folder'], params['attribute'])
    attr_dir = os.path.join('fpfair', params['folder'], 'attributes', params['attribute'])
    string = '-uniform'
else:
    pairs_path = 'fpfair/pairs/{}/{}_pairs_{}.txt'.format(params['folder'], params['folder'], params['attribute'])
    attr_dir = os.path.join('fpfair', params['folder'], 'attributes', params['attribute'])
    string = ''

fieldnames = ['demographic', 'ACC', 'TPR', 'FPR', 'FAR', 'FP', 'FN', 'VAL', 'VAL_STD']
outfile = os.path.join('fpfair/fileio/{}-{}{}.csv'.format(args.model.strip(), params['attribute'], string))
#writer = csv.DictWriter(outfile, fieldnames=fieldnames)
#writer.writeheader()

batch_size = 16
epochs = 15
workers = 0 if os.name == 'nt' else 8



# In[3]:


device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# print('Running on device: {}'.format(device))
print(params['attribute'], params['all_flag'])

# In[4]:












# mtcnn = MTCNN(
#     image_size=160,
#     margin=14,
#     device=device,
#     selection_method='center_weighted_size'
# )


# In[5]:


# Define the data loader for the input set of images
dataset = datasets.ImageFolder(crop_dir, transform=None)


# In[6]:



# overwrites class labels in dataset with path so path can be used for saving output in mtcnn batches
dataset.samples = [
    (p, p)
    for p, _ in dataset.samples
]

loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)


# In[ ]:


crop_paths = []
box_probs = []

for i, (x, b_paths) in enumerate(loader):
    # crops = [p.replace(attr_dir, crop_dir) for p in b_paths]
    crops = [p for p in b_paths]
    # try:
    #     mtcnn(x, save_path=crops)
    # except TypeError as e:
    #     continue
    crop_paths.extend(crops)
    print('Batch {} of {}'.format(i + 1, len(loader)), end='\r')



# In[8]:


# Remove mtcnn to reduce GPU memory usage
# del mtcnn
# torch.cuda.empty_cache()










# In[9]:


# create dataset and data loaders from cropped images output from MTCNN

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

# dataset = datasets.ImageFolder(crop_dir, transform=trans)
dataset = datasets.ImageFolder(crop_dir, transform=trans)

embed_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SequentialSampler(dataset)
)


# In[10]:


#Load pretrained resnet model
#resnet = InceptionResnetV1(
#   classify=False,
#   pretrained='vggface2'
#) #.to(device)
#resnet.eval()

resnet = get_resnet18_pretrained(args.model.strip())


# In[11]:

#skip = os.path.exists('fpfair/embeddings/{}_{}_embeddings.npz'.format(args.model.strip(), args.attribute))
skip = False

if not skip:
    classes = []
    embeddings = []
    count = 0
    with torch.no_grad():
#        for xb, yb in tqdm(embed_loader):
#            # print(xb)
#            xb = np.transpose(xb.numpy(), (0, 2, 3, 1))
#            xb = resize(xb, (xb.shape[0], 160, 160, 3))
#            if count < 1:
#                img = Image.fromarray(((xb[0] * 128) + 127.5).astype('uint8'), 'RGB')
#                img.save('test_pngs/out{}.png'.format(count))
#            b_embeddings = resnet(tf.convert_to_tensor(xb))
#            classes.extend(yb.numpy())
#            embeddings.extend(b_embeddings)
#            count += 1
#
#    np.savez('fpfair/embeddings/{}_{}_embeddings.npz'.format(args.model.strip(), args.attribute), embeddings=embeddings, classes=classes)

        for xb, yb in tqdm(embed_loader):
            xb = xb #.to(device)
            b_embeddings = resnet(xb)
            #embeddings.extend(b_embeddings.numpy())
            embeddings.extend(b_embeddings.numpy())
            classes.extend(yb.numpy())
    
else:
    infile = np.load('fpfair/embeddings/{}_{}_embeddings.npz'.format(args.model.strip(), args.attribute), allow_pickle=True)
    embeddings = infile['embeddings']
    classes = infile['classes']



embeddings_dict = dict(zip(crop_paths,embeddings))


# #### Evaluate embeddings by using distance metrics to perform verification on the official LFW test set.
# 
# The functions in the next block are copy pasted from `facenet.src.lfw`. Unfortunately that module has an absolute import from `facenet`, so can't be imported from the submodule
# 
# added functionality to return false positive and false negatives

# In[13]:


from sklearn.model_selection import KFold
from scipy import interpolate

# LFW functions taken from David Sandberg's FaceNet implementation
def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    is_false_positive = []
    is_false_negative = []

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        # print(embeddings1.shape, embeddings2.shape, mean)
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx], _ ,_ = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _, _, _ = calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx], is_fp, is_fn = calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        tpr = np.mean(tprs,0)
        fpr = np.mean(fprs,0)
        is_false_positive.extend(is_fp)
        is_false_negative.extend(is_fn)

    return tpr, fpr, accuracy, is_false_positive, is_false_negative

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    # print(predict_issame.shape, actual_issame.shape, actual_issame[0])

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    is_fp = np.logical_and(predict_issame, np.logical_not(actual_issame))
    is_fn = np.logical_and(np.logical_not(predict_issame), actual_issame)

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc, is_fp, is_fn

def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate([embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
          mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
        val[fold_idx], far[fold_idx] = calculate_val_far(threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    #thresholds = np.arange(0, 20, 0.05)
    thresholds = np.arange(0, 4, 0.01)
    #thresholds = np.arange(16, 24, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, fp, fn  = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    #thresholds = np.arange(0, 20, 0.005)
    thresholds = np.arange(0, 4, 0.001)
    #thresholds = np.arange(16, 24, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 0.1, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far, fp, fn

def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        #dir_list = os.listdir(os.path.join(lfw_dir, pair[0]))
        if len(pair) == 3:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list += (path0,path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)

    return path_list, issame_list

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs, dtype=object)


# In[14]:


pairs = read_pairs(pairs_path)
print(pairs.shape)
print(crop_dir)
path_list, issame_list = get_paths(crop_dir, pairs)
# print(path_list, issame_list)
embeddings = np.array([embeddings_dict[path] for path in path_list])
print(embeddings.shape)
print(len(issame_list))
print(len(path_list))
# print(embeddings[0])

tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(embeddings, issame_list)
auc = metrics.auc(fpr, tpr)

# In[15]:

print(embeddings.shape)
print(len(issame_list))

demographic = [params['attribute']] * len(accuracy)
print(len(accuracy))
#print(len(tpr))
#print(len(fpr))
##print(len(far))
print(len(fp))
print(len(fn))
##print(len(val))
##print(len(val_std))
new_acc = []
new_val = []
new_val_std = []
new_far = []
new_auc = []
for i in range(len(fp)):
    if i >= len(accuracy):
        new_acc.append('na')
    else:
        new_acc.append(accuracy[i])
    if i > 0:
        new_val.append('na')
        new_val_std.append('na')
        new_far.append('na')
        new_auc.append('na')
    else:
        new_val.append(val)
        new_val_std.append(val_std)
        new_far.append(far)
        new_auc.append(auc)
df = pd.DataFrame({'ACC': new_acc, 'FAR': new_far, 'FP': fp, 'FN': fn, 'VAL': new_val, 'VAL_STD': new_val_std, 'AUC': auc})
df.to_csv(outfile)

#writer.writerow({'demographic': params['attribute'],
#                 'ACC': np.mean(accuracy) * 100,
#                 'TPR': np.mean(tpr) * 100,
#                 'FPR': np.mean(fpr) * 100,
#                 'FAR': np.mean(far) * 100,
#                 'FP': np.mean(fp) * 100,
#                 'FN': np.mean(fn) * 100,
#                 'VAL': np.mean(val)* 100,
#                 'VAL_STD': np.mean(val_std)})


print(accuracy)
print('ACC: {}'.format(np.mean(accuracy) * 100))
print('TPR: {}'.format(np.mean(tpr) * 100))
print('FPR: {}'.format(np.mean(fpr) * 100))
print('FAR: {}'.format(np.mean(far) * 100))
print('FP: {}'.format(np.mean(fp) * 100))
print('FN: {}'.format(np.mean(fn) * 100))
print('VAL: {}'.format(np.mean(val)* 100))
print('VAL_STD: {}'.format(np.mean(val_std)))

