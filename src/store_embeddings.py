import os
import Config
import tensorflow as tf
import numpy as np
from models.face_models import get_model
from utils.crop import *
from utils.eval_utils import *
import argparse
from tqdm import tqdm


def storing(embeddings, embedding_means, embedding_targets, params):
    if params['adversarial_flag']:
        np.savez(os.path.join(Config.DATA, 'embeddings', 'attributes', 'embeddings-{}-{}.npz'.format(params['adv_dir_folder'], params['target_model'])), embeddings=embeddings, embedding_means=embedding_means, embedding_targets=embedding_targets)
    else:
        np.savez(os.path.join(Config.DATA, 'embeddings', 'attributes', 'embeddings-{}.npz'.format(params['all_name'])), embeddings=embeddings, embedding_means=embedding_means, embedding_targets=embedding_targets)


if __name__ == '__main__':
    params = Config.parse_and_configure_arguments()
    correct = 0
    total = 0
    correct_cos = 0
    total_cos = 0
    if params['folder'] == 'vggface2':
        faces, people, files = load_images(params, only_thirty=True)
    else:
        faces, people, files = load_images(params)
    targets = []
    if params['adversarial_flag']:
        for i, identity in enumerate(files):
            targets.append([])
            for file in identity:
                targets[i].append(file.split(people[i])[2].split('_marg_')[0][6:])
    embedding_means, embeddings, embedding_targets = compute_embeddings(faces=faces,
                                                     people=people,
                                                     params=params,
                                                     targets=targets)
    print('load_embeddings')
    storing(embeddings, embedding_means, embedding_targets, params)
        
        
        
#     embedding_means, embeddings = compute_embeddings(faces=faces,
#                                                      people=people,
#                                                      params=params)
#     if params['adversarial_flag'] and params['target_model'] == params['model']:
#         means, _ = load_base_embeddings(params)
#     else:
#         means = None
#     distances, orig_dist = compute_distances(embedding_means, embeddings, people, params, means)
#     storing(embeddings, embedding_means, distances, params, orig_dist)
