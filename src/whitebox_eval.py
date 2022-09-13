import imageio
import numpy as np 
import os
import csv
import time
import argparse
from utils.crop import *
from utils.eval_utils import *
import tensorflow as tf
from models.face_models import *
import Config
from tqdm import tqdm


def write_csv(writer,
              params,
              source,
              target,
              image_names,
              margins,
              amplifications,
              labels,
              distances,
              labels_cos,
              cosines):
    """
    Description

    Keyword arguments:
    """
    for key, val in labels.items():
        # print('Image: {}'.format(key))
        out_dict = {}
        out_dict['model_name'] = params['model']
        out_dict['target_model_name'] = params['target_model']
        out_dict['attack_name'] = params['attack_name']
        out_dict['attack_loss'] = params['attack_loss']
        out_dict['source'] = source
        out_dict['target'] = target
        out_dict['match_source'] = source == labels[key][0]
        out_dict['match_target'] = target == labels[key][0]
        out_dict['cos_source'] = source == labels_cos[key][0]
        out_dict['cos_target'] = target == labels_cos[key][0]
        out_dict['image_name'] = image_names[key]
        out_dict['margin'] = margins[key]
        out_dict['amplification'] = amplifications[key]
        for i, v in enumerate(labels[key]):
            # print('Top {}: {} = {}'.format(i + 1, labels[key][i], distances[key][i]))
            out_dict['top{}'.format(i + 1)] = labels[key][i]
            out_dict['distance{}'.format(i + 1)] = distances[key][i]
        for i, v in enumerate(labels_cos[key]):
            out_dict['topcos{}'.format(i + 1)] = labels_cos[key][i]
            out_dict['cosine{}'.format(i + 1)] = cosines[key][i]

        writer.writerow(out_dict)


if __name__ == "__main__":
    params = Config.parse_and_configure_arguments()
    means, _ = load_base_embeddings(params)

    csvfile = open(os.path.join(Config.DATA, 'fileio', 'whitebox_eval_{}-{}.csv'.format(params['all_name'],
        params['target_model'])), 'w', newline='')
    fieldnames = ['model_name',  'target_model_name', 'attack_name', 'attack_loss', 'source', 'match_source',
                  'match_target', 'target', 'cos_source', 'cos_target', 'image_name', 'margin', 'amplification']
    for i in range(1, params['topn'] + 1):
        fieldnames.append('top{}'.format(i))
        fieldnames.append('distance{}'.format(i))
        fieldnames.append('topcos{}'.format(i))
        fieldnames.append('cosine{}'.format(i))
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    _, people_db = load_images(params=params, only_one=True)

    _, embeddings = load_embeddings(params)
    for person in tqdm(people_db):
        labels, distances, labels_cos, cosines = whitebox_eval(embedding_means=means,
                                                               embeddings=embeddings[person],
                                                               params=params,
                                                               topn=params['topn'])
        split = person.split(':')
        source = split[0]
        target = split[1]
        write_csv(writer=writer,
                  params=params,
                  source=source,
                  target=target,
                  image_names=image_names,
                  margins=margins,
                  amplifications=amplifications,
                  labels=labels,
                  distances=distances,
                  labels_cos=labels_cos,
                  cosines=cosines)
