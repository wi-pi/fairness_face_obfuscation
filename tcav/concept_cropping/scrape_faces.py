from deepface import DeepFace
import Config
import os
from utils.crop import *
from models.face_models import get_model
from tqdm import tqdm
from shutil import copy2


def findApparentAge(age_predictions):
    output_indexes = np.array([i for i in range(0, 101)])
    apparent_age = np.sum(age_predictions * output_indexes)
    return apparent_age


RACE = ['asian', 'indian', 'black', 'white', 'middle', 'latino']
SEX = ['female', 'male']

AGE_LOW = 10
AGE_HIGH = 30


def get_attribute(model, img_224, action):
    if action == 'age':
        age_predictions = model.predict(img_224)[0,:]
        result = findApparentAge(age_predictions)

    elif action == 'sex':
        DATA = {'female': 0, 'male': 0}
        gender_prediction = model.predict(img_224)
        for i in gender_prediction:
            if np.argmax(i) == 0:
                DATA['female'] += 1
            elif np.argmax(i) == 1:
                DATA['male'] += 1

    elif action == 'race':
        DATA = {'asian': 0, 'indian': 0, 'black': 0, 'white': 0, 'middle': 0, 'latino': 0}
        race_predictions = model.predict(img_224)
        for i in race_predictions:
            temp = RACE[np.argmax(i)]
            DATA[temp] += 1

    return DATA


if __name__ == '__main__':
    args = Config.parse_arguments('embedding')
    Config.set_gpu(args.gpu)
    params = Config.set_parameters(model=args.model,
                                   folder=args.folder,
                                   attribute=args.attribute)
    fr_model = get_model(params)
    if params['attribute'] == 'race':
        COUNTS = {'asian': 0, 'indian': 0, 'black': 0, 'white': 0, 'middle': 0, 'latino': 0}
    elif params['attribute'] == 'sex':
        COUNTS = {'male': 0, 'female': 0}
    for identity in tqdm(os.listdir(params['align_dir'])):
        person_path = os.path.join(params['align_dir'], identity)
        file_list = os.listdir(person_path)
        temp_list = []
        if len(file_list) >= 100:
            for i in file_list[:100]:
                temp_list.append(os.path.join(person_path, i))
            faces = read_face_from_aligned(temp_list, params)

            result = get_attribute(fr_model, faces, params['attribute'])
            attr = []
            keys = []
            for key, val in result.items():
                attr.append(val)
                keys.append(key)
            index = np.argmax(attr)
            if attr[index] > 50:
                print(attr[index], keys[index], identity)
                if COUNTS[keys[index]] < 10:
                    new_dir = os.path.join(Config.ROOT, 'tcav', 'faces', keys[index])
                    if not os.path.exists(new_dir):
                        os.mkdir(new_dir)
                    new_path = os.path.join(new_dir, identity)
                    if not os.path.exists(new_path):
                        os.mkdir(new_path)
                    for i, val in enumerate(temp_list):
                        copy2(val, os.path.join(new_path, file_list[i]))
                COUNTS[keys[index]] += 1
                stop = True
                for key, val in COUNTS.items():
                    if val < 10:
                        stop = False
                if stop:
                    exit()
