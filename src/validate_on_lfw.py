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


TOPN = False


def write_csv(writer, accuracies, adv_acc):
    """
    Description

    Keyword arguments:
    """
    out_dict = {}
    for i, _ in enumerate(accuracies['all']):
        for key, val in accuracies.items():
            out_dict[key] = val[i]
            out_dict['{}_adv'.format(key)] = adv_acc[key][i]
        writer.writerow(out_dict)


def write_csv_topn(writer, accuracies, totals):
    """
    Description

    Keyword arguments:
    """
    out_dict = {}
    for key, val in accuracies.items():
        out_dict[key] = val / totals[key]
    writer.writerow(out_dict)


def load_prestored_pairs(params, attr_key):
    pos_pairs = {}
    neg_pairs = {}
    with open(os.path.join(Config.DATA, 'fileio', 'pos_pairs_{}.txt'.format(params['adv_dir_folder'])), 'r', newline='') as infile:
        for line in infile:
            if '=====' in line:
                attr = line.replace('=====', '').strip()
                pos_pairs[attr] = []
            else:
                pos_pairs[attr].append(line.strip())
    with open(os.path.join(Config.DATA, 'fileio', 'neg_pairs_{}.txt'.format(params['adv_dir_folder'])), 'r', newline='') as infile:
        for line in infile:
            if '=====' in line:
                attr = line.replace('=====', '').strip()
                neg_pairs[attr] = []
            else:
                neg_pairs[attr].append(line.strip())
    return neg_pairs[attr_key], pos_pairs[attr_key]


def matching(people, embeddings, means, normal, same, diff, thresholds):
    """
    Evaluates matching accuracy (same or different face)

    Keyword arguments:
    faces -- 
    people -- 
    embeddings -- 
    """
    
    # mean_dist, mean_cos = compute_threshold(embeddings, people)
    # print(mean_dist, mean_cos)
    acc = []
    adv = []
    max_adv = 0
    pos_pairs = []
    neg_pairs = []
    for i in tqdm(thresholds):
        a, b, false_pos, false_neg = matching_accuracy(embeddings, means, i, i, normal, same, diff)
        acc.append(a)
        adv.append(b)
        if b > max_adv:
            pos_pairs = false_pos
            neg_pairs = false_neg
            max_adv = b
    return acc, adv, pos_pairs, neg_pairs
    # increment the threshold and write results to csv
    # compute the threshold all the way earlier


def classifying(people, embedding_means, embeddings, params):
    """
    Evaluates top-n accuracy (with a bucket of labels)

    Keyword arguments:
    faces -- 
    people -- 
    embeddings -- 
    embedding_means -- 
    params -- 
    """
    
    correct = 0
    total = 0
    correct_cos = 0
    total_cos = 0
    for person in tqdm(people):
        labels, distances, labels_cos, cosines = whitebox_eval(embedding_means=embedding_means,
            embeddings=embeddings[person],
            params=params,
            topn=3)
        for key, val in labels.items():
            total += 1
            if labels[key][0] == person:
                correct += 1
        for key, val in labels_cos.items():
            total_cos += 1
            if labels_cos[key][0] == person:
                correct_cos += 1
    print(correct/total)
    print(correct_cos/total_cos)


def topnfriends(all_attributes, people, attributes, embeddings, means, size, epochs):
    totals = {}
    accuracies = {}
    datas = {}
    peoples = {}
    for attr in all_attributes:
        peoples[attr] = []
        for i in people:
            if i in attributes[attr]:
                peoples[attr].append(i)
    for i in all_attributes:
        totals['skewpos-{}'.format(i)] = 0
        totals['skewneg-{}'.format(i)] = 0
        totals['balance-{}'.format(i)] = 0
        accuracies['skewpos-{}'.format(i)] = 0
        accuracies['skewneg-{}'.format(i)] = 0
        accuracies['balance-{}'.format(i)] = 0
        datas['skewpos-{}'.format(i)] = []
        datas['skewneg-{}'.format(i)] = []
        datas['balance-{}'.format(i)] = []
    for it in tqdm(range(0, epochs)):
        friends = {}
        sampling = {}
        for i in all_attributes:
            friends['skewpos-{}'.format(i)] = []
            friends['skewneg-{}'.format(i)] = []
            friends['balance-{}'.format(i)] = []
        for i in all_attributes:
            for j in all_attributes:
                temp = peoples[j]
                np.random.shuffle(temp)
                if i == j:
                    friends['skewpos-{}'.format(i)].extend(temp[:50])
                    friends['skewneg-{}'.format(i)].extend(temp[:5])
                    sampling['skewpos-{}'.format(i)] = temp[:50]
                    sampling['skewneg-{}'.format(i)] = temp[:5]
                    sampling['balance-{}'.format(i)] = temp[:17]
                else:
                    friends['skewpos-{}'.format(i)].extend(temp[:5])
                    friends['skewneg-{}'.format(i)].extend(temp[:20])
                friends['balance-{}'.format(i)].extend(temp[:17])
        for key, val in friends.items():
            samples = random_topn_samples(sampling[key], embeddings, size)
            data, a, t = topn_buckets(embeddings, means, samples, val)
            accuracies[key] += a
            totals[key] += t
            datas[key].append(data)
    for key, val in accuracies.items():
        print(key, val / totals[key])
    return accuracies, totals


if __name__ == '__main__':
    params = Config.parse_and_configure_arguments()
    tf_config = Config.set_gpu(params['gpu'])
    if params['source'] == None:
        if params['adversarial_flag']:
            if TOPN:
                csvfile = open(os.path.join(Config.DATA, 'fileio', 'performance_topn_{}-{}.csv'.format(params['adv_dir_folder'], params['target_model'])), 'w', newline='')
            else:
                csvfile = open(os.path.join(Config.DATA, 'fileio', 'performance_{}-{}.csv'.format(params['adv_dir_folder'], params['target_model'])), 'w', newline='')
        else:
            if TOPN:
                csvfile = open(os.path.join(Config.DATA, 'fileio', 'performance_topn_{}.csv'.format(params['all_name'])), 'w', newline='')
            else:
                csvfile = open(os.path.join(Config.DATA, 'fileio', 'performance_{}.csv'.format(params['all_name'])), 'w', newline='')
        
        _, people, _ = load_images(params, only_one=True)
        _, _, file_names = load_images(params, only_files=True)

        if params['adversarial_flag']:
            means, normal = load_target_embeddings(params)
            _, embeddings = load_embeddings(params)
            sizes = {'Black': 10000, 'White': 10000, 'Asian': 10000, 'Indian': 10000, 'Male': 10000, 'NOTMale': 10000, 'all': 40000}
            size = 200
            epochs = 100
            # topn_sizes = {'Black': 3000, 'White': 3000, 'Asian': 3000, 'Indian': 3000, 'Male': 3000, 'NOTMale': 3000, 'all': 10000}
        else:
            normal = None
            means, embeddings = load_embeddings(params)
            sizes = {'Black': 4490, 'White': 45760, 'Asian': 7746, 'Indian': 86, 'Male': 46769, 'NOTMale': 16818, 'all': 40000}
            size = 5
            epochs = 500
            # topn_sizes = {'Black': 300, 'White': 5000, 'Asian': 400, 'Indian': 70, 'Male': 5000, 'NOTMale': 2000, 'all': 5000}

        attributes, ppl = load_attributes(params)

        if TOPN:
            all_attributes = []
            if params['attribute'] == 'race':
                all_attributes.extend(Config.LFWRACE_ATTRIBUTES)
            else:
                all_attributes.extend(Config.LFWSEX_ATTRIBUTES)
            fieldnames = []
            accuracies, totals = topnfriends(all_attributes, people, attributes, embeddings, means, size, epochs)
            for key, val in accuracies.items():
                fieldnames.append(key)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            write_csv_topn(writer, accuracies, totals)
        else:
            all_attributes = []
            if params['attribute'] == 'race':
                all_attributes.extend(Config.LFWRACE_ATTRIBUTES)
            else:
                all_attributes.extend(Config.LFWSEX_ATTRIBUTES)
            fieldnames = ['all', 'all_adv']
            accuracies = {'all': []}
            adv_acc = {'all': []}
            pos_pairs = {'all': []}
            neg_pairs = {'all': []}
            for i in all_attributes:
                fieldnames.append(i)
                fieldnames.append('{}_adv'.format(i))
                accuracies[i] = []
                adv_acc[i] = []
                pos_pairs[i] = []
                neg_pairs[i] = []
                
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            if params['model'] != params['target_model']:
                same, diff = load_prestored_pairs(params, 'all')
            else:
                same, diff = random_matching_pairs(people, embeddings, means, normal, sizes['all'])
            accuracies['all'], adv_acc['all'], pos_pairs['all'], neg_pairs['all'] = matching(people, embeddings, means, normal, same, diff, params['thresholds'])

            people_attr = {}
            for attr in all_attributes:
                people_attr[attr] = []
                for i in people:
                    if i in attributes[attr]:
                        people_attr[attr].append(i)

            for attr in all_attributes:
                print(attr)
                if params['model'] != params['target_model']:
                    same, diff = load_prestored_pairs(params, attr)
                else:
                    same, diff = random_matching_pairs(people_attr[attr], embeddings, means, normal, sizes[attr])
                accuracies[attr], adv_acc[attr], pos_pairs[attr], neg_pairs[attr] = matching(people_attr[attr], embeddings, means, normal, same, diff, params['thresholds'])
            write_csv(writer, accuracies, adv_acc)
            if params['model'] == params['target_model']:
                with open(os.path.join(Config.DATA, 'fileio', 'pos_pairs_{}.txt'.format(params['adv_dir_folder'])), 'w', newline='') as outfile:
                    for key, val in pos_pairs.items():
                        outfile.write('{}=====\n'.format(key))
                        for j in val:
                            outfile.write('{}\n'.format(j))
                with open(os.path.join(Config.DATA, 'fileio', 'neg_pairs_{}.txt'.format(params['adv_dir_folder'])), 'w', newline='') as outfile:
                    for key, val in neg_pairs.items():
                        outfile.write('{}=====\n'.format(key))
                        for j in val:
                            outfile.write('{}\n'.format(j))
    else:
        if TOPN:
            csvfile = open(os.path.join(Config.DATA, 'fileio', 'performance_topn_{}-{}-{}.csv'.format(params['adv_dir_folder'], params['target_model'], )), 'w', newline='')
        else:
            csvfile = open(os.path.join(Config.DATA, 'fileio', 'performance_{}-{}-{}.csv'.format(params['adv_dir_folder'], params['target_model'])), 'w', newline='')
        _, people, _ = load_images(params, only_one=True)

        if params['adversarial_flag']:
            means, normal = load_target_embeddings(params)
            _, embeddings = load_embeddings(params)
            sizes = {'Black': 10000, 'White': 10000, 'Asian': 10000, 'Indian': 10000, 'Male': 10000, 'NOTMale': 10000, 'all': 40000}
            size = 200
            epochs = 100
            # topn_sizes = {'Black': 3000, 'White': 3000, 'Asian': 3000, 'Indian': 3000, 'Male': 3000, 'NOTMale': 3000, 'all': 10000}
        else:
            normal = None
            means, embeddings = load_embeddings(params)
            sizes = {'Black': 4490, 'White': 45760, 'Asian': 7746, 'Indian': 86, 'Male': 46769, 'NOTMale': 16818, 'all': 40000}
            size = 5
            epochs = 500
            # topn_sizes = {'Black': 300, 'White': 5000, 'Asian': 400, 'Indian': 70, 'Male': 5000, 'NOTMale': 2000, 'all': 5000}

        attributes, ppl = load_attributes(params)

        if TOPN:
            all_attributes = []
            if params['attribute'] == 'race':
                all_attributes.extend(Config.LFWRACE_ATTRIBUTES)
            else:
                all_attributes.extend(Config.LFWSEX_ATTRIBUTES)
            fieldnames = []
            accuracies, totals = topnfriends(all_attributes, people, attributes, embeddings, means, size, epochs)
            for key, val in accuracies.items():
                fieldnames.append(key)
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            write_csv_topn(writer, accuracies, totals)
        else:
            all_attributes = []
            if params['attribute'] == 'race':
                all_attributes.extend(Config.LFWRACE_ATTRIBUTES)
            else:
                all_attributes.extend(Config.LFWSEX_ATTRIBUTES)
            fieldnames = ['all', 'all_adv']
            accuracies = {'all': []}
            adv_acc = {'all': []}
            for i in all_attributes:
                fieldnames.append(i)
                fieldnames.append('{}_adv'.format(i))
                accuracies[i] = []
                adv_acc[i] = []
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            same, diff = random_matching_pairs(people, embeddings, means, normal, sizes['all'])
            accuracies['all'], adv_acc['all'] = matching(people, embeddings, means, normal, same, diff, params['thresholds'])

            people_attr = {}
            for attr in all_attributes:
                people_attr[attr] = []
                for i in people:
                    if i in attributes[attr]:
                        people_attr[attr].append(i)

            for attr in all_attributes:
                print(attr)
                same, diff = random_matching_pairs(people_attr[attr], embeddings, means, normal, sizes[attr])
                accuracies[attr], adv_acc[attr] = matching(people_attr[attr], embeddings, means, normal, same, diff, params['thresholds'])
            write_csv(writer, accuracies, adv_acc)