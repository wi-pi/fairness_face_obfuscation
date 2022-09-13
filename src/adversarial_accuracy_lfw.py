import os
import Config
import tensorflow as tf
import numpy as np
from models.face_models import get_model
from utils.crop import *
from utils.eval_utils import *
import argparse
from tqdm import tqdm
import random
import csv
import pandas as pd

def write_csv(writer, accuracies, thresholds):
    """
    Description

    Keyword arguments:
    """
    out_dict = {}
    for i in range(len(thresholds)):
        for key, val in accuracies.items():
            out_dict[key] = val[i]
        writer.writerow(out_dict)


def get_targets(people, file_names):
    embedding_targets = {}
    for i, file in enumerate(file_names):
        for f in file:
            # print(f.split(people[i])[2].split('_marg_')[0][6:])
            target = f.split(people[i])[2].split('_marg_')[0][6:]
            if people[i] not in embedding_targets:
                embedding_targets[people[i]] = []
            embedding_targets[people[i]].append(target)
    return embedding_targets


# def restrict_pairs(people, file_names, attributes):
#     indices = {}
#     attributes = []
#     for i, file in enumerate(file_names):
#         if people[i] not in indices:
#             indices[people[i]] = []
#             count = 0
#         for f in file:
#             indices[people[i]].append(count)
#             count += 1
#     return indices


def load_prestored_pairs(params, attr_key):
    targ_pairs = {}
    untarg_pairs = {}
    with open(os.path.join(Config.DATA, 'fileio', 'targ_pairs_{}.txt'.format(params['adv_dir_folder'])), 'r', newline='') as infile:
        for line in infile:
            if '=====' in line:
                attr = line.replace('=====', '').strip()
                targ_pairs[attr] = []
                count = -1
            elif '-----' in line:
                targ_pairs[attr].append([])
                count += 1
            else:
                targ_pairs[attr][count].append(line.strip())
    with open(os.path.join(Config.DATA, 'fileio', 'untarg_pairs_{}.txt'.format(params['adv_dir_folder'])), 'r', newline='') as infile:
        for line in infile:
            if '=====' in line:
                attr = line.replace('=====', '').strip()
                untarg_pairs[attr] = []
                count = -1
            elif '-----' in line:
                untarg_pairs[attr].append([])
                count += 1
            else:
                untarg_pairs[attr][count].append(line.strip())
    return targ_pairs[attr_key], untarg_pairs[attr_key]


def matching(people, embeddings, normal, thresholds, embedding_targets, targeted=None, untargeted=None):
    """
    Evaluates matching accuracy (same or different face)

    Keyword arguments:
    faces -- 
    people -- 
    embeddings -- 
    """
    
    # mean_dist, mean_cos = compute_threshold(embeddings, people)
    # print(mean_dist, mean_cos)
    targ = []
    untarg = []
    targ_pairs = []
    untarg_pairs = []
    for i, thresh in tqdm(enumerate(thresholds)):
        if targeted is None or untargeted is None:
            b, t = evaluate_targeted(people, embeddings, normal, thresh, embedding_targets)
            a, u = evaluate_untargeted(people, embeddings, normal, thresh)
        else:
            if len(targeted[i]) > 0:
                a, t = evaluate_targeted(people, embeddings, normal, thresh, embedding_targets, targeted[i])
            else:
                a = 0
                t = []
            if len(untargeted[i]) > 0:
                b, u = evaluate_untargeted(people, embeddings, normal, thresh, untargeted[i])
            else:
                b = 0
                u = []
        targ.append(a)
        untarg.append(b)
        targ_pairs.append(t)
        untarg_pairs.append(u)
    return targ, untarg, targ_pairs, untarg_pairs


if __name__ == '__main__':
    params = Config.parse_and_configure_arguments()
    target_csv = open(os.path.join(Config.DATA, 'fileio', 'performance_targeted_{}-{}.csv'.format(params['adv_dir_folder'], params['target_model'])), 'w', newline='')
    untarget_csv = open(os.path.join(Config.DATA, 'fileio', 'performance_untargeted_{}-{}.csv'.format(params['adv_dir_folder'], params['target_model'])), 'w', newline='')

    _, people, _ = load_images(params, only_one=True)
    _, people_targets, file_names = load_images(params, only_files=True)
    embedding_targets = get_targets(people_targets, file_names)
    means, normal = load_target_embeddings(params)
    _, embeddings, _ = load_embeddings(params)
    attributes, ppl = load_attributes(params)

    fieldnames = []
    targ_accuracies = {}
    untarg_accuracies = {}

    targ_pairs = {}
    untarg_pairs = {}
    all_attributes = params['attributes']
    people_attr = params['sources']
    # indices = restrict_pairs(people_targets, file_names, people_attr)
    for i in all_attributes:
        fieldnames.append(i)
        targ_accuracies[i] = []
        untarg_accuracies[i] = []

        targ_pairs[i] = []
        untarg_pairs[i] = []

    targ_writer = csv.DictWriter(target_csv, fieldnames=fieldnames)
    untarg_writer = csv.DictWriter(untarget_csv, fieldnames=fieldnames)

    targ_writer.writeheader()
    untarg_writer.writeheader()

    # targ_accuracies['all'], untarg_accuracies['all'] = matching(people, embeddings, normal, params['thresholds'], embedding_targets)

    # people_attr = {}
    # for attr in all_attributes:
    #     people_attr[attr] = []
    #     for i in people:
    #         if i in attributes[attr]:
    #             people_attr[attr].append(i)
    
    for i, attr in enumerate(all_attributes):
        print(attr, len(people_attr[i]))
        # if params['model'] != params['target_model']:
        #     targeted, untargeted = load_prestored_pairs(params, attr)
        # else:
        targeted = None
        untargeted = None
        targ_accuracies[attr], untarg_accuracies[attr], targ_pairs[attr], untarg_pairs[attr] = matching(people_attr[i], embeddings, normal, params['thresholds'], embedding_targets, targeted, untargeted)
        # print(targ_accuracies[attr], untarg_accuracies[attr])
    write_csv(targ_writer, targ_accuracies, params['thresholds'])
    write_csv(untarg_writer, untarg_accuracies, params['thresholds'])

    # if params['model'] == params['target_model']:
    #     with open(os.path.join(Config.DATA, 'fileio', 'targ_pairs_{}.txt'.format(params['adv_dir_folder'])), 'w', newline='') as outfile:
    #         for key, val in targ_pairs.items():
    #             outfile.write('{}=====\n'.format(key))
    #             for t, pairs in enumerate(val):
    #                 outfile.write('{}-----\n'.format(t))
    #                 for j in pairs:
    #                     outfile.write('{}\n'.format(j))
    #     with open(os.path.join(Config.DATA, 'fileio', 'untarg_pairs_{}.txt'.format(params['adv_dir_folder'])), 'w', newline='') as outfile:
    #         for key, val in untarg_pairs.items():
    #             outfile.write('{}=====\n'.format(key))
    #             for t, pairs in enumerate(val):
    #                 outfile.write('{}-----\n'.format(t))
    #                 for j in pairs:
    #                     outfile.write('{}\n'.format(j))
