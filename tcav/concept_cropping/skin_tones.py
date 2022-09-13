import os
import numpy as np
import imageio
import cv2
import Config
from utils.crop import *
from tqdm import tqdm
from mtcnn import MTCNN


skin_tone_rgb = {'type1': [(244, 242, 245), (237, 236, 234), (250, 249, 247), (253, 251, 231), (253, 246, 231), (254, 247, 230)], 'type2': [(250, 240, 239), (243, 235, 230), (244, 241, 235), (251, 252, 244), (252, 248, 238), (254, 246, 226), (255, 249, 226)], 'type3': [(255, 249, 226), (241, 232, 196), (240, 227, 174), (225, 211, 148), (242, 227, 152), (236, 215, 160), (236, 218, 134)], 'type4': [(228, 197, 103), (226, 194, 106), (224, 194, 124), (223, 185, 120), (200, 166, 100), (189, 152, 98), (157, 107, 65)], 'type5': [(143, 87, 60), (122, 76, 44), (100, 45, 14), (101, 44, 26), (96, 45, 27), (86, 46, 36), (62, 26, 13)], 'type6': [(45, 32, 36), (20, 21, 42)]}


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
    for file in base_files[:20]:
        img = imageio.imread(os.path.join(folder, file))
        face, det, features = crop_face(img, params, detector)
        #print(features)
        if face is not None:
            #img = np.around(img / 255.0, decimals=12)
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


def cluster(img):
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    return dominant


def best_match(color):
    closest = 'ERROR'
    best_dist = 30 / 255
    for key, val in skin_tone_rgb.items():
        for item in val:
            dist = np.linalg.norm((np.array(item) / 255.0) - np.array(color))
            if dist < best_dist:
                best_dist = dist
                closest = key
    return closest


if __name__ == '__main__':
    params = Config.parse_and_configure_arguments()
    detector = MTCNN()
    #folder = os.path.join(Config.ROOT, 'tcav', 'faces')
    folder = params['align_dir']
    with open('vggskins.csv', 'a') as outfile:
        for person in tqdm(os.listdir(folder)[7314:]):
            best_matches_cheek = []
            best_matches_forehead = []
            best_matches_face = []
            person_folder = os.path.join(folder, person)
            faces, file_names, imgs, patches = load_images(params, person_folder, detector)
            for face, file, img, forehead, cheek in zip(faces, file_names, imgs, patches['forehead'], patches['cheek']):
                cheek_color = cluster(cheek)
                forehead_color = cluster(forehead)
                face_color = cluster(face)
                best_matches_cheek.append(best_match(cheek_color))
                best_matches_forehead.append(best_match(forehead_color))
                best_matches_face.append(best_match(face_color))
            new_matches_cheek = [i for i in best_matches_cheek if i != 'ERROR']
            new_matches_forehead = [i for i in best_matches_forehead if i != 'ERROR']
            new_matches_face = [i for i in best_matches_face if i != 'ERROR']
            print(new_matches_cheek)
            print(new_matches_forehead)
            print(new_matches_face)
            if ((len(new_matches_cheek) > 2 and len(new_matches_forehead) > 2 and len(new_matches_face) > 2) and
                (max(set(new_matches_face), key=new_matches_face.count) == max(set(new_matches_forehead), key=new_matches_forehead.count) or
                max(set(new_matches_face), key=new_matches_face.count) == max(set(new_matches_cheek), key=new_matches_cheek.count))):
                outdir = os.path.join(Config.ROOT, 'tcav', 'concepts')
                outfile.write('{},{}\n'.format(person, max(set(new_matches_face), key=new_matches_face.count)))
                print('{},{}\n'.format(person, max(set(new_matches_face), key=new_matches_face.count)))
                for i in range(len(faces)):
                    if best_matches_cheek[i] != 'ERROR':
                        outfilename = os.path.join(outdir, best_matches_cheek[i], 'cheek-{}-{}'.format(person, file_names[i]))
                        imageio.imwrite(outfilename, (patches['cheek'][i] * 255).astype(np.uint8))
                    if best_matches_forehead[i] != 'ERROR':
                        outfilename = os.path.join(outdir, best_matches_forehead[i], 'forehead-{}-{}'.format(person, file_names[i]))
                        imageio.imwrite(outfilename, (patches['forehead'][i] * 255).astype(np.uint8))
