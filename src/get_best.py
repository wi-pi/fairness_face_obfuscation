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


def write_csv(writer, top, top_attributes):
    """
    Description

    Keyword arguments:
    """
    out_dict = {}
    for key, val in top.items():
        if val > 7 or (top_attributes[key] == 'white' and val > 3):
            out_dict['person'] = key
            out_dict['count'] = val
            out_dict['attribute'] = top_attributes[key]
            writer.writerow(out_dict)


def matching(people, embeddings, means, normal, size, epochs, attribute):
    """
    Evaluates matching accuracy (same or different face)

    Keyword arguments:
    faces -- 
    people -- 
    embeddings -- 
    """
    top = {}
    top_attributes = {}
    for i in range(0, epochs):
        same, diff = random_matching_pairs(people, embeddings, means, normal, size)
        accuracy, total = get_best_people(embeddings, means, 10.0, 10.0, normal, same, diff)
        people_list = []
        acc_list = []
        for person, val in accuracy.items():
            people_list.append(person)
            acc_list.append(accuracy[person] / total[person])
            indices = np.argsort(np.array(acc_list))
        for j in indices[:70]:
            p = people_list[j]
            if p not in top:
                top[p] = 0
                top_attributes[p] = attribute
            top[p] += 1
    return top, top_attributes

if __name__ == '__main__':
    params = Config.parse_and_configure_arguments()
    tf_config = Config.set_gpu(params['gpu'])
    csvfile = open(os.path.join(Config.DATA, 'fileio', 'best_{}_{}.csv'.format(params['all_name'], params['attribute'])), 'w', newline='')
    
    _, people, _ = load_images(params, only_one=True)

    normal = None
    means, embeddings = load_embeddings(params)
    # sizes = {'Black': 4490, 'White': 45760, 'Asian': 7746, 'Indian': 86, 'Male': 46769, 'NOTMale': 16818, 'all': 40000}
    sizes = {'black': 15680, 'white': 126680, 'asian': 9980, 'indian': 6420, 'middle': 7720, 'latino': 6140, 'male': 103900, 'female': 68720, 'all': 80000}
    size = 5
    epochs = 30

    attributes, ppl = load_attributes(params)

    all_attributes = []
    if params['attribute'] == 'race':
        all_attributes.extend(Config.VGGRACE_ATTRIBUTES)
        all_attributes.extend(Config.VGGRACE_ATTRIBUTES)
    else:
        all_attributes.extend(Config.VGGSEX_ATTRIBUTES)
        all_attributes.extend(Config.VGGSEX_ATTRIBUTES)
    fieldnames = ['person', 'count', 'attribute']
    for i in all_attributes:
        fieldnames.append('person-{}'.format(i))
        fieldnames.append('count-{}'.format(i))
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    people_attr = {}
    for attr in all_attributes:
        people_attr[attr] = []
        for i in people:
            if i in attributes[attr]:
                people_attr[attr].append(i)

    for attr in all_attributes:
        top, top_attributes = matching(people_attr[attr], embeddings, means, normal, sizes[attr], epochs, attr)
        print(top)
        write_csv(writer, top, top_attributes)
