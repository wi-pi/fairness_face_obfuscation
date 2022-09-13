import os
import numpy as np
import imageio
import cv2
import Config
from utils.crop import *
from tqdm import tqdm
from mtcnn import MTCNN


def load_images(params, folder, detector):
    """
    Description

    Keyword arguments:
    """
    boxes = {'forehead': [(90, 25, 130, 65)],
             'cheek': [(25, 110, 65, 150)],
             'eye': [(30, 40, 110, 120)],
             'eyes': [(20, 30, 210, 130)],
             'mouth': [(60, 120, 160, 220)]}
    patches = {'forehead': [], 'cheek': [], 'eye': [], 'eyes': [], 'mouth': []}

    file_names = []
    imgs = []
    base_faces = []

    base_files = os.listdir(folder)
    for file in base_files[:25]:
        img = imageio.imread(os.path.join(folder, file))
        face, det, features = crop_face(img, params, detector)
        if face is not None:
            img = np.around(img / 255.0, decimals=12)
            file_names.append(file)
            base_faces.append(np.array([face]))
            imgs.append(img)
    faces = np.squeeze(np.array(base_faces))
    if len(imgs) <= 1:
        faces = np.expand_dims(faces, axis=0)

    for key, val in boxes.items():
        for f in faces:
            img_size = np.asarray(f.shape)[0:2]
            det = np.squeeze(val)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0], 0)
            bb[1] = np.maximum(det[1], 0)
            bb[2] = np.minimum(det[2], img_size[1])
            bb[3] = np.minimum(det[3], img_size[0])
            cropped = f[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled = cv2.resize(cropped, (229, 229), params['interpolation'])
            face = np.array(scaled)
            patches[key].append(face)

    return faces, file_names, imgs, patches


if __name__ == '__main__':
    args = Config.parse_arguments('embedding')
    Config.set_gpu(args.gpu)
    params = Config.set_parameters(model=args.model)
    detector = MTCNN()
    folder = os.path.join(Config.ROOT, 'tcav', 'faces')
    for attribute in tqdm(os.listdir(folder)):
        attribute_folder = os.path.join(folder, attribute)
        for person in tqdm(os.listdir(attribute_folder)):
            person_folder = os.path.join(attribute_folder, person)
            faces, file_names, imgs, patches = load_images(params, person_folder, detector)
            outdir = os.path.join(Config.ROOT, 'tcav', 'concepts')
            for key, val in patches.items():
                concept_dir = os.path.join(outdir, '{}-{}'.format(attribute, key))
                if not os.path.exists(concept_dir):
                    os.mkdir(concept_dir)
                for i, face in enumerate(val):
                    outfilename = os.path.join(concept_dir, '{}-{}'.format(person, file_names[i]))
                    imageio.imwrite(outfilename, (face * 255).astype(np.uint8))
