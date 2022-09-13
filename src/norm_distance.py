import Config
import argparse
import numpy as np
import os
import csv
from tqdm import tqdm
from utils.attack_utils import load_images, return_write_path, save_image, transpose_back
from utils.eval_utils import load_images as load_adv_images
from utils.crop import apply_delta

def write_csv(writer,
              params,
              source,
              target,
              margin,
              l2norm):
    """
    Description

    Keyword arguments:
    """
    out_dict = {}
    out_dict['model_name'] = params['model']
    out_dict['attack_name'] = params['attack_name']
    out_dict['attack_loss'] = params['attack_loss']
    out_dict['l2norm'] = l2norm
    out_dict['source'] = source
    out_dict['target'] = target
    out_dict['margin'] = margin
    writer.writerow(out_dict)


if __name__ == '__main__':
    count = 0
    params = Config.parse_and_configure_arguments()
    csvfile = open(os.path.join(Config.DATA, 'fileio', 'norm_distance_{}.csv'.format(params['adv_dir_folder'])), 'w', newline='')
    fieldnames = ['model_name', 'attack_name', 'attack_loss', 'l2norm', 'source', 'target', 'margin']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    faces, people, files = load_adv_images(params)
    for bucket, person, face in tqdm(zip(files, people, faces)):
        _, sourcefiles, _, _ = load_images(params=params, source=person, target=person, detector=None, only_paths=True)
        for f, adv_face in zip(bucket, face):
            index = 0
            try:
                for i, file in enumerate(sourcefiles):
                    stripfile = file.replace('.jpg', '').replace('.png', '')
                    if stripfile in f:
                        break
                bigsplit = f.split('_marg_')
                margin = bigsplit[1].split('_amp_')[0]
                target = bigsplit[0].split('{}_'.format(stripfile))[1]
                split = f.split('_amp_')[0]
                npz_file = split.replace(os.path.join(params['adversarial_dir'], params['adv_dir_folder']), '').replace('/{}/'.format(person), '') + '.npz'
                npz_path = os.path.join(Config.DATA, 'new_adv_imgs/{}/{}/{}_loss/npz'.format(params['attack_name'],
                                                                                                 params['model'],
                                                                                                 params['attack_loss']))
                inname = os.path.join(npz_path, npz_file)
                npzfile = np.load(inname, allow_pickle=True)
                write_csv(writer=writer,
                          params=params,
                          source=person,
                          target=target,
                          margin=margin,
                          l2norm=np.linalg.norm(npzfile['delta_clip_stack'][0]))
            except FileNotFoundError as e:
                count += 1
                print(e)
                continue
    print(count)
