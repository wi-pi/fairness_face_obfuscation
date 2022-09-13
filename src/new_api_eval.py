import numpy as np
import os
import time
import Config
import argparse
from FaceAPI.microsoft_face import verify_API
from FaceAPI.aws_rekognition import aws_compare
from FaceAPI.facepp import facepp
from FaceAPI import credentials
import csv
from utils.eval_utils import *
from tqdm import tqdm

BASE_DIR = credentials.s3_bucket_url


def write_dict(writer, sources, targets, scores, ths, successes, source_files, target_files):
    out_dict = {}
    for so, ta, sc, th, su, sf, tf in zip(sources, targets, scores, ths, successes, source_files, target_files):
        out_dict['source'] = so
        out_dict['target'] = ta
        out_dict['score'] = sc
        out_dict['th0'] = th[0]
        out_dict['th1'] = th[1]
        out_dict['th2'] = th[2]
        out_dict['success'] = su
        out_dict['source_file'] = sf
        out_dict['target_file'] = tf
        writer.writerow(out_dict)


def get_targets(people, file_names):
    targets = {}
    for i, file in enumerate(file_names):
        for f in file:
            target = f.split(people[i])[2].split('_marg_')[0][6:]
            if people[i] not in targets:
                targets[people[i]] = []
            targets[people[i]].append(target)
    return targets


def get_scores_targeted(api_name,
                        people_adv,
                        targets_adv,
                        files,
                        adv_files):
    sources = []
    targets = []
    scores = []
    ths = []
    successes = []
    source_files = []
    target_files = []
    total = 0
    success = 0
    for source, adv_imgs in tqdm(zip(people_adv, adv_files)):
        targs = targets_adv[source]
        for targ, adv_img in tqdm(zip(targs, adv_imgs)):
            if ('_marg_10.00_amp' in adv_img and '_0001_' in adv_img) or (params['attack'] == 'lowkey'):
                # for tar_img in files[targ]:
                tar_img = files[targ][0]
                new_adv_img = adv_img.replace(Config.DATA, Config.S3_DIR + '/fairnessfaces')
                new_tar_img = tar_img.replace(Config.DATA, Config.S3_DIR + '/fairnessfaces')
                # print(adv_img)
                # print(tar_img)
                if api_name == 'awsverify':
                    score = aws_compare(new_adv_img, new_tar_img, 0)
                elif api_name == 'azure':
                    score, isSame, has_face = verify_API(new_adv_img, new_tar_img, 1)
                elif api_name == 'facepp':
                    score, th = facepp(new_adv_img, new_tar_img)
                if score != None:
                    this_success = False
                    if api_name == 'awsverify':
                        ths.append([50, 50, 50])
                        if score >= 50:
                            this_success = True
                    elif api_name == 'azure':
                        ths.append([0.5, 0.5, 0.5])
                        if score >= 0.5:
                            this_success = True
                    elif api_name == 'facepp':
                        ths.append(th)
                        if score >= th[1]:
                            this_success = True

                    if this_success:
                        # print('{} attack succeed targeted {} {}\n'.format(score, adv_img, tar_img))
                        success += 1
                        successes.append('success')
                    else:
                        # print('{} attack failed targeted {} {}\n'.format(score, adv_img, tar_img))
                        successes.append('failure')

                    sources.append(source)
                    targets.append(targ)
                    scores.append(score)
                    source_files.append(adv_img)
                    target_files.append(tar_img)
                    total += 1
                    print('Success rate: {}'.format(success/total), end='\r')
    return sources, targets, scores, ths, successes, source_files, target_files


def get_scores_untargeted(api_name,
                          people_adv,
                          files,
                          adv_files):
    sources = []
    targets = []
    scores = []
    ths = []
    successes = []
    source_files = []
    target_files = []
    total = 0
    success = 0

    for source, adv_imgs in tqdm(zip(people_adv, adv_files)):
        for adv_img in tqdm(adv_imgs):
            if ('_marg_10.00_amp' in adv_img and '_0001_' in adv_img) or (params['attack'] == 'lowkey' and '_0001_' in adv_img):
                # for tar_img in files[source]:
                tar_img = files[source][0]
                new_adv_img = adv_img.replace(Config.DATA, Config.S3_DIR + '/fairnessfaces')
                new_tar_img = tar_img.replace(Config.DATA, Config.S3_DIR + '/fairnessfaces')
                # print(adv_img)
                # print(tar_img)
                # print(new_tar_img)
                if api_name == 'awsverify':
                    score = aws_compare(new_adv_img, new_tar_img, 0)
                elif api_name == 'azure':
                    score, isSame, has_face = verify_API(new_adv_img, new_tar_img, 1)
                elif api_name == 'facepp':
                    score, th = facepp(new_adv_img, new_tar_img)
                if score != None:
                    this_success = False
                    if api_name == 'awsverify':
                        ths.append([50, 50, 50])
                        if score <= 50:
                            this_success = True
                    elif api_name == 'azure':
                        ths.append([0.5, 0.5, 0.5])
                        if score <= 0.5:
                            this_success = True
                    elif api_name == 'facepp':
                        ths.append(th)
                        if score <= th[1]:
                            this_success = True

                    if this_success:
                        # print('{} attack succeed untargeted {} {}\n'.format(score, adv_img, tar_img))
                        success += 1
                        successes.append('success')
                    else:
                        # print('{} attack failed untargeted {} {}\n'.format(score, adv_img, tar_img))
                        successes.append('failure')

                    sources.append(source)
                    targets.append(source)
                    scores.append(score)
                    source_files.append(adv_img)
                    target_files.append(tar_img)
                    total += 1
                    print('Success rate: {}'.format(success/total), end='\r')
    return sources, targets, scores, ths, successes, source_files, target_files


if __name__ == "__main__":
    params = Config.parse_and_configure_arguments()
    tf_config = Config.set_gpu(params['gpu'])
    
    fieldnames = ['source', 'target', 'score', 'th0', 'th1', 'th2', 'success', 'source_file', 'target_file']
    _, people_adv, adv_file_names = load_images(params, only_files=True)
    targets_adv = get_targets(people_adv, adv_file_names)
    params['adversarial_flag'] = False
    _, people, file_names = load_images(params, only_files=True)
    all_attributes = params['attributes']
    people_attr = params['sources']
    files = {}
    for p, f in zip(people, file_names):
        files[p] = f
    params['adversarial_flag'] = True

    if params['attack'] != 'lowkey' and params['targeted_success'] == True:
        target_csv = open(os.path.join(Config.DATA, 'fileio', '{}_targeted_{}.csv'.format(params['api_name'], params['adv_dir_folder'])), 'w', newline='')
        targ_writer = csv.DictWriter(target_csv, fieldnames=fieldnames)
        targ_writer.writeheader()
        sources, targets, scores, ths, successes, source_files, target_files = get_scores_targeted(params['api_name'], people_adv, targets_adv, files, adv_file_names)
        write_dict(targ_writer, sources, targets, scores, ths, successes, source_files, target_files)
    if params['targeted_success'] == False:
        untarget_csv = open(os.path.join(Config.DATA, 'fileio', '{}_untargeted_{}.csv'.format(params['api_name'], params['adv_dir_folder'])), 'w', newline='')
        untarg_writer = csv.DictWriter(untarget_csv, fieldnames=fieldnames)
        untarg_writer.writeheader()
        sources, targets, scores, ths, successes, source_files, target_files = get_scores_untargeted(params['api_name'], people_adv, files, adv_file_names)
        write_dict(untarg_writer, sources, targets, scores, ths, successes, source_files, target_files)
