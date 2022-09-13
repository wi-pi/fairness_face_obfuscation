import numpy as np
import tensorflow as tf
import Config
from attacks.cw_new import CW
from attacks.pgd_new import PGD
from models.face_models import get_model
from utils.attack_utils import (set_bounds, transpose_back, initialize_dict, populate_dict, save_image,
    save_np, load_images, return_write_path, amplify)
from tensorflow.keras import backend
from tqdm import tqdm
from mtcnn import MTCNN


def find_adv(params,
             fr_model,
             face,
             face_stack_source,
             face_stack_target,
             margin=0):
    """
    Description

    Keyword arguments:
    sess -- tensorflow session
    params -- parameter dict (Config)
    fr_model -- loaded facenet or centerface tensorflow model
    face -- single face or batch of faces to perturb
    face_stack_source -- single face or batch of source faces (used in hinge loss)
    face_stack_target -- single face or batch of target faces (used in all loss)
    """

    num_base = 15
    num_src = face_stack_source.shape[0]
    num_target = face_stack_target.shape[0]

    if params['attack'] == 'cw':
        best_lp, best_const, best_adv, best_delta = CW(model=fr_model,
                       params=params,
                       imgs=face,
                       src_imgs=face_stack_source,
                       target_imgs=face_stack_target,
                       num_base=num_base,
                       num_src=num_src,
                       num_target=num_target,
                       margin=margin)
    elif params['attack'] == 'pgd':
        best_lp = []
        best_const = []
        best_adv = []
        best_delta = []
        if params['batch_size'] <= 0:
            batch_size = num_base
        else:
            batch_size = min(params['batch_size'], num_base)
        for i in range(0,len(face),batch_size):
            adv = PGD(params=params,
                      model_fn=fr_model,
                      x=face[i:i+batch_size],
                      src=face_stack_source,
                      tar=face_stack_target,
                      margin=np.float32(margin))
            
            delta = adv - face[i:i+batch_size]
            const = [None] * face.shape[0]

            best_lp.extend(best_lp)
            best_const.extend(const)
            best_adv.extend(adv)
            best_delta.extend(delta)

    return best_adv, best_delta, best_lp, best_const


def outer_attack(params,
                 faces,
                 file_names,
                 source,
                 target,
                 imgs,
                 dets):
    """
    Outer attack loop of margin (kappa) values. Finds adversarial example, amplifies delta,
    saves perturbed image, creates .npz file with delta values.

    Keyword arguments:
    params -- parameter dict (Config)
    faces -- dict of base, source, and target faces
    file_names -- base image file names
    source -- source person labal
    target -- target person labal
    imgs -- original images containing faces
    dets -- bounding box coordinates of faces
    """
    for margin in params['margin_list']:
        Config.BM.mark('Adversarial Example Generation')
        adv, delta, lp, const = find_adv(params=params, 
                                         fr_model=fr_model,
                                         face=faces['base'], 
                                         face_stack_source=faces['source'],
                                         face_stack_target=faces['target'],
                                         margin=margin)
        Config.BM.mark('Adversarial Example Generation')

        Config.BM.mark('Dictionary Initialization')
        adv_crop_dict, delta_clip_dict, adv_img_dict = initialize_dict(file_names=file_names)
        Config.BM.mark('Dictionary Initialization')

        Config.BM.mark('Amplifying and Writing Images')
        for amplification in params['amp_list']:
            img_path, crop_path, npz_path = return_write_path(params=params,
                                                              file_names=file_names,
                                                              target=target,
                                                              margin=margin,
                                                              amplification=amplification,
                                                              source=source)
            adv_crop_stack, delta_clip_stack, adv_img_stack = amplify(params=params,
                                                                      face=faces['base'],
                                                                      delta=delta,
                                                                      amp=amplification,
                                                                      dets=dets,
                                                                      imgs=imgs)
            save_image(file_names = file_names,
                       out_img_names = img_path,
                       out_img_names_crop = crop_path,
                       adv_img_stack = adv_img_stack,
                       adv_crop_stack = adv_crop_stack)
            adv_crop_dict, delta_clip_dict, adv_img_dict = populate_dict(file_names = file_names,
                                                                         adv_crop_dict = adv_crop_dict,
                                                                         adv_crop_stack = adv_crop_stack,
                                                                         delta_clip_dict = delta_clip_dict,
                                                                         delta_clip_stack = delta_clip_stack,
                                                                         adv_img_dict = adv_img_dict,
                                                                         adv_img_stack = adv_img_stack)
        Config.BM.mark('Amplifying and Writing Images')

        Config.BM.mark('Saving Numpy Array')
        save_np(out_npz_names = npz_path,
                adv_crop_dict = adv_crop_dict,
                delta_clip_dict = delta_clip_dict,
                adv_img_dict = adv_img_dict)
        Config.BM.mark('Saving Numpy Array')


if __name__ == "__main__":
    params = Config.parse_and_configure_arguments()
    Config.BM.mark('Model Loaded')
    fr_model = get_model(params)
    detector = MTCNN()
    Config.BM.mark('Model Loaded')

    if params['source'] == None:
        sources = params['flats']
    else:
        sources = params['sources'][int(params['source'])]

    if params['attribute'] == None:
        targets = params['targets']
    else:
        if params['different_flag'] and params['target'] == None:
            targets = params['other'][int(params['source'])]
        elif params['different_flag'] and params['target'] != None:
            targets = params['sources'][int(params['target'])]
        else:
            targets = params['sources'][int(params['source'])]

    for source in tqdm(sources):
        for target in tqdm(targets,leave=False):
            if source != target:
                faces, file_names, imgs, dets = load_images(params=params, source=source, target=target, detector=detector)
                outer_attack(params=params,
                             faces=faces,
                             file_names=file_names,
                             source=source,
                             target=target,
                             imgs=imgs,
                             dets=dets)
