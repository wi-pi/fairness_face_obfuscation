import numpy as np
import tensorflow as tf
import os
import Config
from utils.crop import *
from tensorflow.keras import backend
from models.face_models import get_model
import math
from tqdm import tqdm
from shutil import copy2
from random import randrange
import random
import pandas as pd


def load_adv_images(params):
    """
    

    Keyword arguments:
    """
    
    model = params['model']
    attack_name = params['attack_name']
    attack_loss = params['attack_loss']
    margin_list = params['margin_list']
    amp_list = params['amp_list']
    face_db = []
    name_db = []
    margin_db = []
    amp_db = []
    people_db = []
    if params['mean_loss'] == 'embedding':
        mean_str = '_mean'
    else:
        mean_str = ''
    for s in tqdm(params['flats']):
        for t in params['targets']:
            if t != s:
                faces = []
                image_names = []
                margins = []
                amplifications = []
                people_db.append('{}:{}'.format(s, t))
                for file in os.listdir(os.path.join(params['folder_dir'], s)):
                    source_file = file.replace('.jpg', '')
                    for ii, margin in enumerate(margin_list):
                        marg_str = '%0.2f' % margin
                        for jj, amp in enumerate(amp_list):
                            amp_str = '%0.3f' % amp
                            adv_img_file = '{}_{}_{}_loss_{}_{}_marg_{}_amp_{}.png'.format(attack_name,
                                                                                           model,
                                                                                           attack_loss[0],
                                                                                           source_file,
                                                                                           t,
                                                                                           marg_str,
                                                                                           amp_str)
                            adv_img_path = '{}/data/new_adv_imgs/{}/{}/{}_loss/crop{}/{}/'.format(Config.DATA,
                                                                                                 attack_name,
                                                                                                 model,
                                                                                                 attack_loss,
                                                                                                 mean_str,
                                                                                                 adv_img_file)
                            try:
                                face = imageio.imread(adv_img_path)
                                face = pre_proc(face,
                                                params=params)
                                faces.append(face)
                                image_names.append(source_file)
                                margins.append(marg_str)
                                amplifications.append(amp_str)
                            except FileNotFoundError as e:
                                print(e)
                face_db.append(np.array(faces))
                name_db.extend(image_names)
                margin_db.extend(margins)
                amp_db.extend(amplifications)
    return face_db, name_db, margin_db, amp_db, people_db


def move_adv_images(params):
    """
    Keyword arguments:
    """
    
    model_name = params['model']
    attack_name = params['attack_name']
    attack_loss = params['attack_loss']
    margin_list = params['margin_list']
    amp_list = params['amp_list']

    if params['mean_loss'] == 'embedding':
        mean_str = '_mean'
    else:
        mean_str = ''

    if params['attribute'] == 'none':
        targets = params['targets']
    else:
        targets = params['flats']

    all_path = os.path.join(params['adversarial_dir'], params['adv_dir_folder'])
    if not os.path.exists(all_path):
        os.mkdir(all_path)

    for s in tqdm(params['flats']):
        id_path = os.path.join(all_path, s)
        if not os.path.exists(id_path):
            os.mkdir(id_path)
        for t in targets:
            if t != s:
                for file in os.listdir(os.path.join(params['folder_dir'], s)):
                    source_file = file.replace('.jpg', '').replace('.png', '')
                    for ii, margin in enumerate(margin_list):
                        marg_str = '%0.2f' % margin
                        for jj, amp in enumerate(amp_list):
                            amp_str = '%0.3f' % amp
                            adv_img_file = '{}_{}_{}_loss_{}_{}_marg_{}_amp_{}.png'.format(attack_name,
                                                                                           model_name,
                                                                                           attack_loss[0],
                                                                                           source_file,
                                                                                           t,
                                                                                           marg_str,
                                                                                           amp_str)
                            adv_img_path = '{}/new_adv_imgs/{}/{}/{}_loss/crop{}/{}'.format(Config.DATA,
                                                                                                 attack_name,
                                                                                                 model_name,
                                                                                                 attack_loss,
                                                                                                 mean_str,
                                                                                                 adv_img_file)
                            try:
                                os.rename(adv_img_path, os.path.join(id_path, adv_img_file))
                            except FileNotFoundError as e:
                                pass


def load_images(params, only_one=False, only_files=False,folder_is_img_directory=False, only_thirty=False):
    """
    Description

    Keyword arguments:
    """
    
    face_db = []
    people = []
    files = []
    if params['adversarial_flag']:
        load_dir = os.path.join(params['adversarial_dir'], params['adv_dir_folder'])
    elif folder_is_img_directory:
        load_dir = params['folder_dir']
    else:
        load_dir = params['align_dir']
    for person in tqdm(os.listdir(load_dir)):
        person_path = os.path.join(load_dir, person)
        file_list = os.listdir(person_path)
        if len(file_list) > 0:
            if only_one:
                x = randrange(0, len(file_list))
                file_list = [os.path.join(person_path, file_list[x])]
            elif only_thirty:
                for i in range(len(file_list))[:30]:
                    file_list[i] = os.path.join(person_path, file_list[i])
                file_list = file_list[:30]
            else:
                for i in range(len(file_list)):
                    file_list[i] = os.path.join(person_path, file_list[i])
            if not only_files:
                faces = read_face_from_aligned(file_list=file_list,
                                               params=params)
                face_db.append(faces)
            people.append(person)
            files.append(file_list)
    return face_db, people, files


def load_attributes(params):
    infile = np.load(os.path.join(Config.DATA, 'embeddings', 'attributes', 'attributes-{}.npz'.format(params['folder'])), allow_pickle=True)
    return infile['attributes'].item(), infile['people'].item()


def load_embeddings(params):
    if params['adversarial_flag']:
        infile = np.load(os.path.join(Config.DATA, 'embeddings', 'attributes', 'embeddings-{}-{}.npz'.format(params['adv_dir_folder'], params['target_model'])), allow_pickle=True)
        return infile['embedding_means'].item(), infile['embeddings'].item(), infile['embedding_targets'].item()
    else:
        infile = np.load(os.path.join(Config.DATA, 'embeddings', 'attributes', 'embeddings-{}.npz'.format(params['all_name'])), allow_pickle=True)
        return infile['embedding_means'].item(), infile['embeddings'].item()


def load_base_embeddings(params):
    infile = np.load(os.path.join(Config.DATA, 'embeddings', 'attributes', 'embeddings-{}.npz'.format(params['all_name'].replace('{}_'.format(params['attack_name']), ''))), allow_pickle=True)
    return infile['embedding_means'].item(), infile['embeddings'].item()


def load_target_embeddings(params):
    infile = np.load(os.path.join(Config.DATA, 'embeddings', 'attributes', 'embeddings-{}.npz'.format(params['all_name'].replace('{}_'.format(params['attack_name']), '').replace(params['model'], params['target_model']))), allow_pickle=True)
    print(infile['embedding_means'].item()['Julian_Battle'].shape)
    return infile['embedding_means'].item(), infile['embeddings'].item()


def load_distances(params):
    if params['adversarial_flag']:
        infile = np.load(os.path.join(Config.DATA, 'embeddings', 'attributes', 'distances-{}-{}.npz'.format(params['adv_dir_folder'], params['target_model'])), allow_pickle=True)
    else:
        infile = np.load(os.path.join(Config.DATA, 'embeddings', 'attributes', 'distances-{}.npz'.format(params['all_name'])), allow_pickle=True)
    distances = infile['distances'].item()
    if 'orig_dist' in infile:
        orig_dist = infile['orig_dist'].item()
    else:
        orig_dist = None
    return distances, orig_dist


def compute_embeddings(faces,
                       people,
                       params,
                       targets):
    """
    Description

    Keyword arguments:
    """
    
    embeddings = {}
    embedding_means = {}
    embedding_targets = {}
    fr_model = get_model(params=params)
    batch_size = params['batch_size']
    for p, person in tqdm(enumerate(faces)):
        cur_embedding = []
        sub_batch = -len(person)
        for i in range(0, len(person), batch_size):
            cur_batch = len(person) - i
            cur_imgs = person[i:i+batch_size]
            if batch_size > cur_batch:
                sub_batch = batch_size - cur_batch
                cur_imgs = np.pad(cur_imgs, ((0,sub_batch),(0,0),(0,0),(0,0)))
            cur_embedding.extend(fr_model(cur_imgs))
        embedding_mean = np.mean(cur_embedding[:-sub_batch], axis=0)
        embedding_means[people[p]] = embedding_mean
        embeddings[people[p]] = np.array(cur_embedding[:-sub_batch])
        if params['adversarial_flag']:
            embedding_targets[people[p]] = targets[p]
    return embedding_means, embeddings, embedding_targets


def compute_threshold(embeddings, people):
    """
    Description

    Keyword arguments:
    """
    
    distances = []
    cosines = []
    for person in tqdm(people):
        embed = embeddings[person]
        for i in range(embed.shape[0]):
            for j in range(embed.shape[0]):
                if i != j:
                    cos_sim = np.dot(embed[i], embed[j]) / (np.linalg.norm(embed[i]) * np.linalg.norm(embed[j]))
                    distance = np.linalg.norm(embed[i] - embed[j])
                    cos_sim = np.arccos(cos_sim) / math.pi
                    distances.append(distance)
                    cosines.append(cos_sim)
    return np.mean(distances), np.mean(cosines)


def compute_distances(embedding_means, embeddings, people, params, means):
    """
    Description

    Keyword arguments:
    """
    
    distances = {}
    orig_dist = {}
    for p1, person1 in tqdm(enumerate(people)):
        for p2, person2 in enumerate(people):
            if p1 != p2:
                embed1 = embedding_means[person1]
                embed2 = embedding_means[person2]
                distance = np.linalg.norm(embed1 - embed2)
                distances['{}-{}'.format(person1, person2)] = distance
    if params['adversarial_flag'] and params['target_model'] == params['model']:
        for p, person in tqdm(enumerate(people)):
            embed1 = embedding_means[person]
            embed2 = means[person]
            distance = np.linalg.norm(embed1 - embed2)
            orig_dist[person] = distance

    return distances, orig_dist


def whitebox_eval(embedding_means,
                  embeddings,
                  params,
                  topn=1):
    """
    Description

    Keyword arguments:
    """
    
    final_l2 = {}
    final_cos = {}

    people = embedding_means.keys()
    
    for person in people:
        embedding_mean_person = embedding_means[person]
        length = embeddings.shape[0]
        output = [0]*length
        output_cos = [0]*length

        for i in range(length):
            # dot = np.dot(embeddings, embedding_mean_person)
            # norm = np.linalg.norm(embeddings) * np.linalg.norm(embedding_mean_person)
            # cos_sim = dot / norm
            cos_sim = np.dot(embeddings[i], embedding_mean_person) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embedding_mean_person))
            distance = np.linalg.norm(embeddings[i] - embedding_mean_person)
            cos_sim = np.arccos(cos_sim) / math.pi
            output[i] = distance
            output_cos[i] = cos_sim
        final_l2[person] = output
        final_cos[person] = output_cos
    # {matt: [3,2,1], leo: [4,5,4], bill: [6,7,7]}
    def return_top(dictionary,
                   final_cos,
                   n):
        """
        Description

        Keyword arguments:
        """
    
        distances = {}
        cosines = {}
        label_dict = {}
        final_labels = {}
        final_labels_cos = {}
        final_distances = {}
        final_cosines = {}
        keys = dictionary.keys()
        for i, key in enumerate(keys):
            label_dict[i] = key
            for j, dist in enumerate(dictionary[key]):
                if j not in distances:
                    distances[j] = []
                distances[j].append(dist)
            for j, cos in enumerate(final_cos[key]):
                if j not in cosines:
                    cosines[j] = []
                cosines[j].append(cos)

        for key, val in distances.items():
            indices = np.argsort(np.array(val))
            # print(indices)
            final_labels[key] = []
            final_distances[key] = []
            for i in range(n):
                final_labels[key].append(label_dict[indices[i]])
                final_distances[key].append(val[indices[i]])
        for key, val in cosines.items():
            indices = np.argsort(np.array(val))
            # print(indices)
            final_labels_cos[key] = []
            final_cosines[key] = []
            for i in range(n):
                final_labels_cos[key].append(label_dict[indices[i]])
                final_cosines[key].append(val[indices[i]])

        return final_labels, final_distances, final_labels_cos, final_cosines
    final_labels, final_distances, final_labels_cos, final_cosines = return_top(final_l2, final_cos, topn)
    return final_labels, final_distances, final_labels_cos, final_cosines


def topn_buckets(embeddings, means, samples, friends):
    total = 0
    acc = 0
    data = {'source': [], 'top1': [], 'top2': [], 'top3': [], 'top4': [], 'top5': [], 'dist1': [], 'dist2': [], 'dist3': [], 'dist4': [], 'dist5': []}
    for i in samples:
        split = i.split(':')
        distances = []
        data['source'].append(split[0])
        for j in friends:
            distance = np.linalg.norm(embeddings[split[0]][int(split[1])] - means[j])
            distances.append(distance)
        indices = np.argsort(np.array(distances))
        if split[0] == friends[indices[0]]:
            acc += 1
        total += 1
        for ind in range(0, 5):
            data['top{}'.format(ind+1)].append(friends[ind])
            data['dist{}'.format(ind+1)].append(distances[ind])
    return data, acc, total


def evaluate_targeted(people, embeddings, normal_embeddings, mean_distance, embedding_targets, targeted=None):
    total = 0
    acc = 0
    targeted_pairs = []

    if targeted != None:
        for pair in targeted:
            split = pair.split(':')
            person1, person2, index1, index2 = split[0], split[1], int(split[2]), int(split[3])
            distance = np.linalg.norm(embeddings[person1][index1] - normal_embeddings[person2])
            if distance <= mean_distance:
                acc += 1
            total += 1
    else:
        for person1 in people:
            embed1 = embeddings[person1]
            for i in range(embed1.shape[0]):
                # if i in indices[person1]:
                # embed2 = normal_embeddings[embedding_targets[person1][0]]
                for j, embed2 in enumerate(normal_embeddings[embedding_targets[person1][i]]):
                    distance = np.linalg.norm(embed1[i] - embed2)
                    if distance <= mean_distance:
                        acc += 1
                        targeted_pairs.append('{}:{}:{}:{}'.format(person1, embedding_targets[person1][i], i, 0))
                    total += 1
    return acc / total, targeted_pairs


def evaluate_untargeted(people, embeddings, normal_embeddings, mean_distance, untargeted=None):
    total = 0
    acc = 0
    untargeted_pairs = []

    if untargeted != None:
        for pair in untargeted:
            split = pair.split(':')
            person, index1, index2 = split[0], int(split[1]), int(split[2])
            distance = np.linalg.norm(embeddings[person][index1] - normal_embeddings[person])
            if distance <= mean_distance:
                acc += 1
            total += 1
    else:
        for person in people:
            embed1 = embeddings[person]
            for i in range(embed1.shape[0]):
                # if i in indices[person]:
                # embed2 = normal_embeddings[person]
                for j, embed2 in enumerate(normal_embeddings[person]):
                    distance = np.linalg.norm(embed1[i] - embed2)
                    if distance > mean_distance:
                        acc += 1
                        untargeted_pairs.append('{}:{}:{}'.format(person, i, 0))
                    total += 1
    return acc / total, untargeted_pairs

def evaluate_targeted_df_return(people, embeddings, normal_embeddings, mean_distance, embedding_targets, targeted=None):
    total = 0
    acc = 0
    targeted_pairs = []
    return_dict = {'person1':[], 'person2' : [], 'distance': [], 'thresh' : [], 'success' : []}
    if targeted != None:
        for pair in targeted:
            split = pair.split(':')
            person1, person2, index1, index2 = split[0], split[1], int(split[2]), int(split[3])
            distance = np.linalg.norm(embeddings[person1][index1] - normal_embeddings[person2])
            return_dict['person1'].append(person1)
            return_dict['person2'].append(person2)
            return_dict['distance'].append(distance)
            return_dict['thresh'].append(mean_distance)
            return_dict['success'].append(False)
            if distance <= mean_distance:
                return_dict['success'][-1] = True
                acc += 1
            total += 1
            
    else:
        for person1 in people:
            embed1 = embeddings[person1]
            for i in range(embed1.shape[0]):
                # if i in indices[person1]:
                # embed2 = normal_embeddings[embedding_targets[person1][0]]
                for j, embed2 in enumerate(normal_embeddings[embedding_targets[person1][i]]):
                    distance = np.linalg.norm(embed1[i] - embed2)
                    return_dict['person1'].append(person1)
                    return_dict['person2'].append(embedding_targets[person1][i])
                    return_dict['distance'].append(distance)
                    return_dict['thresh'].append(mean_distance)
                    return_dict['success'].append(False)
                    if distance <= mean_distance:
                        return_dict['success'][-1] = True
                        acc += 1
                        targeted_pairs.append('{}:{}:{}:{}'.format(person1, embedding_targets[person1][i], i, 0))
                    total += 1
    df_return = pd.DataFrame(return_dict)
    return acc / total, targeted_pairs, df_return


def evaluate_untargeted_df_return(people, embeddings, normal_embeddings, mean_distance, untargeted=None):
    total = 0
    acc = 0
    untargeted_pairs = []
    return_dict = {'person1':[], 'distance': [], 'thresh' : [], 'success' : []}
    if untargeted != None:
        for pair in untargeted:
            split = pair.split(':')
            person, index1, index2 = split[0], int(split[1]), int(split[2])
            distance = np.linalg.norm(embeddings[person][index1] - normal_embeddings[person])
            return_dict['person1'].append(person)
            return_dict['distance'].append(distance)
            return_dict['thresh'].append(mean_distance)
            return_dict['success'].append(False)
            if distance <= mean_distance:
                return_dict['success'][-1] = True
                acc += 1
            total += 1
    else:
        for person in people:
            embed1 = embeddings[person]
            for i in range(embed1.shape[0]):
                # if i in indices[person]:
                # embed2 = normal_embeddings[person]
                for j, embed2 in enumerate(normal_embeddings[person]):
                    distance = np.linalg.norm(embed1[i] - embed2)
                    return_dict['person1'].append(person)
                    return_dict['distance'].append(distance)
                    return_dict['thresh'].append(mean_distance)
                    return_dict['success'].append(False)
                    if distance > mean_distance:
                        acc += 1
                        return_dict['success'][-1] = True
                        untargeted_pairs.append('{}:{}:{}'.format(person, i, 0))
                    total += 1
    df_return = pd.DataFrame(return_dict)
    return acc / total, untargeted_pairs, df_return


def matching_accuracy(embeddings, means, mean_distance, mean_cosine, normal_embeddings=None, same_pairs=None, diff_pairs=None):
    """
    Description

    Keyword arguments:
    """
    
    total = 0
    dist_acc = 0
    cos_acc = 0
    adv_acc = 0
    adv_total = 0
    false_pos = []
    false_neg = []
    targeted_pairs = []
    untargeted_pairs = []

    def increment(acc, total, dist, mean, same, pair):
        if same and dist <= mean:
            acc += 1
        elif not same and dist > mean:
            acc += 1
        elif same:
            false_neg.append(pair)
        elif not same:
            false_pos.append(pair)
        total += 1
        return acc, total

    def increment_targeted(acc, total, dist, mean, pair):
        if dist <= mean:
            acc += 1
            targeted_pairs.append(pair)
        total += 1
        return acc, total

    def increment_untargeted(acc, total, dist, mean, pair):
        if dist > mean:
            acc += 1
            targeted_pairs.append(pair)
        total += 1
        return acc, total

    if same_pairs != None and diff_pairs != None:
        if normal_embeddings != None:
            new_embeddings = normal_embeddings
        else:
            new_embeddings = embeddings
        for i in same_pairs:
            split = i.split(':')
            person, index1, index2 = split[0], int(split[1]), int(split[2])
            distance = np.linalg.norm(embeddings[person][index1] - new_embeddings[person][index2])
            dist_acc, total = increment(dist_acc, total, distance, mean_distance, True, i)
            distance = np.linalg.norm(embeddings[person][index1] - means[person])
            adv_acc, adv_total = increment(adv_acc, adv_total, distance, mean_distance, True, i)
            distance = np.linalg.norm(embeddings[person][index2] - means[person])
            adv_acc, adv_total = increment(adv_acc, adv_total, distance, mean_distance, True, i)
        for i in diff_pairs:
            split = i.split(':')
            person1, person2, index1, index2 = split[0], split[1], int(split[2]), int(split[3])
            distance = np.linalg.norm(embeddings[person1][index1] - new_embeddings[person2][index2])
            dist_acc, total = increment(dist_acc, total, distance, mean_distance, False, i)
            distance = np.linalg.norm(embeddings[person1][index1] - means[person2])
            adv_acc, adv_total = increment(adv_acc, adv_total, distance, mean_distance, False, i)
            distance = np.linalg.norm(embeddings[person2][index2] - means[person1])
            adv_acc, adv_total = increment(adv_acc, adv_total, distance, mean_distance, False, i)
    else:
        for person1, embed1 in tqdm(embeddings.items()):
            for person2, embed2 in embeddings.items():
                for i in range(embed1.shape[0]):
                    for j in range(embed2.shape[0]):
                        if i != j or person1 != person2:
                            # cos_sim = np.dot(embed1[i], embed2[j]) / (np.linalg.norm(embed1[i]) * np.linalg.norm(embed2[j]))
                            distance = np.linalg.norm(embed1[i] - embed2[j])
                            # cos_sim = np.arccos(cos_sim) / math.pi
                            if (distance <= mean_distance and person1 is person2 or
                                distance > mean_distance and person1 != person2):
                                dist_acc += 1
                            # if (cos_sim <= mean_cosine and person1 is person2 or
                                # distance > mean_distance and person1 != person2):
                                # cos_acc += 1
                            total += 1
    return dist_acc / total, adv_acc / adv_total, false_pos, false_neg
    # print(cos_acc / total)


def get_best_people(embeddings, means, mean_distance, mean_cosine, normal_embeddings=None, same_pairs=None, diff_pairs=None):
    """
    Description

    Keyword arguments:
    """
    
    total = 0
    dist_acc = 0
    cos_acc = 0
    accuracies = {}
    totals = {}

    def increment(accuracies, totals, dist, mean, same, person):
        if person not in accuracies:
            accuracies[person] = 0
        if person not in totals:
            totals[person] = 0
        if same and dist <= mean:
            accuracies[person] += 1
        elif not same and dist > mean:
            accuracies[person] += 1
        totals[person] += 1
        return accuracies, totals

    if same_pairs != None and diff_pairs != None:
        if normal_embeddings != None:
            new_embeddings = normal_embeddings
        else:
            new_embeddings = embeddings
        for i in same_pairs:
            split = i.split(':')
            person, index1, index2 = split[0], int(split[1]), int(split[2])
            distance = np.linalg.norm(embeddings[person][index1] - new_embeddings[person][index2])
            accuracies, totals = increment(accuracies, totals, distance, mean_distance, True, person)
        for i in diff_pairs:
            split = i.split(':')
            person1, person2, index1, index2 = split[0], split[1], int(split[2]), int(split[3])
            distance = np.linalg.norm(embeddings[person1][index1] - new_embeddings[person2][index2])
            accuracies, totals = increment(accuracies, totals, distance, mean_distance, False, person1)
            accuracies, totals = increment(accuracies, totals, distance, mean_distance, False, person2)
    return accuracies, totals


def random_matching_pairs(people, embeddings, means, normal, size):
    """
    Description

    Keyword arguments:
    """
    same_pairs = []
    diff_pairs = []
    shapes = {}
    normal_shapes = {}
    for p in people:
        shapes[p] = embeddings[p].shape[0]
        if normal != None:
            normal_shapes[p] = normal[p].shape[0]
    total = 0
    for key, val in shapes.items():
        total += val
    # if total > 5000:
    done = {}
    for i in tqdm(range(size)):
        p1 = randrange(0, len(people))
        p2 = randrange(0, len(people))
        j = randrange(0, shapes[people[p1]])
        if normal != None:
            k = randrange(0, normal_shapes[people[p2]])
        else:
            k = randrange(0, shapes[people[p2]])
        string = '{}:{}:{}:{}'.format(people[p1], people[p2], j, k)
        while string in done:
            p1 = randrange(0, len(people))
            p2 = randrange(0, len(people))
            j = randrange(0, shapes[people[p1]])
            if normal != None:
                k = randrange(0, normal_shapes[people[p2]])
            else:
                k = randrange(0, shapes[people[p2]])
            string = '{}:{}:{}:{}'.format(people[p1], people[p2], j, k)
        done[string] = 0
        diff_pairs.append(string)
    for i in tqdm(range(size)):
        p = randrange(0, len(people))
        j = randrange(0, shapes[people[p]])
        if normal != None:
            k = randrange(0, normal_shapes[people[p]])
        else:
            k = randrange(0, shapes[people[p]])
        string = '{}:{}:{}'.format(people[p], j, k)
        while string in done or j == k:
            p = randrange(0, len(people))
            j = randrange(0, shapes[people[p]])
            if normal != None:
                k = randrange(0, normal_shapes[people[p]])
            else:
                k = randrange(0, shapes[people[p]])
            string = '{}:{}:{}'.format(people[p], j, k)
        done[string] = 0
        same_pairs.append(string)
    return same_pairs, diff_pairs
    # else:
    #     for p1 in tqdm(range(0, len(people))):
    #         for p2 in range(p1, len(people)):
    #             for i in range(0, shapes[people[p1]]):
    #                 for j in range(0, shapes[people[p2]]):
    #                     if p1 == p2 and i != j:
    #                         same_pairs.append('{}:{}:{}'.format(people[p1], i, j))
    #                     elif p1 != p2:
    #                         diff_pairs.append('{}:{}:{}:{}'.format(people[p1], people[p2], i, j))
    #     np.random.shuffle(same_pairs)
    #     np.random.shuffle(diff_pairs)
    #     return same_pairs[:size], diff_pairs[:size]


def random_topn_samples(people, embeddings, size):
    """
    Description

    Keyword arguments:
    """
    samples = []
    shapes = {}
    for p in people:
        shapes[p] = embeddings[p].shape[0]
    done = {}
    for i in range(size):
        p = randrange(0, len(people))
        j = randrange(0, shapes[people[p]])
        string = '{}:{}'.format(people[p], j)
        while string in done:
            p = randrange(0, len(people))
            j = randrange(0, shapes[people[p]])
            string = '{}:{}'.format(people[p], j)
        done[string] = 0
        samples.append(string)
    return samples
