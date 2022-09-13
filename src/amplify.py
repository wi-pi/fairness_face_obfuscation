import Config
import argparse
import numpy as np
import os
from utils.attack_utils import load_images, return_write_path, save_image, transpose_back, amplify
from utils.crop import apply_delta
from utils.eval_utils import load_images as load_adv_images
from tqdm import tqdm
from mtcnn import MTCNN
import imageio


AMPLIFICATION = 3.0
AMP_STR = '%0.3f' % AMPLIFICATION


if __name__ == '__main__':
    params = Config.parse_and_configure_arguments()

    new_dir = os.path.join(params['adversarial_dir'], params['adv_dir_folder'].replace('1.0', str(AMPLIFICATION)))
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    detector = MTCNN()
    faces, people, files = load_adv_images(params)
    for bucket, person, face in tqdm(zip(files, people, faces)):
        _, sourcefiles, imgs, dets = load_images(params=params, source=person, target=person, detector=detector)
        if not os.path.exists(os.path.join(new_dir, person)):
            os.mkdir(os.path.join(new_dir, person))
        for f, adv_face in zip(bucket, face):
            index = 0
            for i, file in enumerate(sourcefiles):
                if file.replace('.jpg', '').replace('.png', '') in f:
                    index = i
            split = f.split('_amp_')[0]
            npz_file = split.replace(os.path.join(params['adversarial_dir'], params['adv_dir_folder']), '').replace('/{}/'.format(person), '') + '.npz'
            outfilename = split.replace(params['adv_dir_folder'], params['adv_dir_folder'].replace('1.0', str(AMPLIFICATION)))
            outfilename = outfilename + '_amp_{}.png'.format(AMP_STR)
            npz_path = os.path.join(Config.DATA, 'new_adv_imgs/{}/{}/{}_loss/npz'.format(params['attack_name'],
                                                                                         params['model'],
                                                                                         params['attack_loss']))
            inname = os.path.join(npz_path, npz_file)
            npzfile = np.load(inname, allow_pickle=True)
            delta = npzfile['delta_clip_stack'][0]
            adv_crop_stack, delta_clip_stack, adv_img_stack = amplify(params=params,
                                                                      face=np.array([adv_face]),
                                                                      delta=np.array([delta]),
                                                                      amp=AMPLIFICATION,
                                                                      dets=np.array([dets[index]]),
                                                                      imgs=np.array([imgs[index]]))
            imageio.imwrite(outfilename, (adv_crop_stack[0][:,:,::-1] * 255).astype(np.uint8))
