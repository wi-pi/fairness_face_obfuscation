from deepface import DeepFace
import Config
import os
from utils.crop import *
from models.face_models import get_model
from tqdm import tqdm


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
    ATTRIBUTE_DICT = {'asian': [], 'indian': [], 'black': [], 'white': [], 'middle': [], 'latino': [], 'female': [], 'male': []}
    PEOPLE_DICT = {}
    params = Config.parse_and_configure_arguments()
    Config.set_gpu(params['gpu'])
    
    params['model'] = 'Gender'
    sex_model = get_model(params)
    params['model'] = 'Race'
    race_model = get_model(params)
    for identity in tqdm(os.listdir(params['align_dir'])):
        person_path = os.path.join(params['align_dir'], identity)
        file_list = os.listdir(person_path)
        temp_list = []
        PEOPLE_DICT[identity] = []
        for i in file_list[:100]:
            temp_list.append(os.path.join(person_path, i))
        faces = read_face_from_aligned(temp_list, params)
        result = get_attribute(race_model, faces, 'race')
        attr = []
        keys = []
        for key, val in result.items():
            attr.append(val)
            keys.append(key)
        index = np.argmax(attr)
        ATTRIBUTE_DICT[keys[index]].append(identity)
        PEOPLE_DICT[identity].append(keys[index])
        result = get_attribute(sex_model, faces, 'sex')
        attr = []
        keys = []
        for key, val in result.items():
            attr.append(val)
            keys.append(key)
        index = np.argmax(attr)
        ATTRIBUTE_DICT[keys[index]].append(identity)
        PEOPLE_DICT[identity].append(keys[index])
    np.savez(os.path.join(Config.DATA, 'embeddings', 'attributes', 'attributes-{}.npz'.format(params['folder'])), attributes=ATTRIBUTE_DICT, people=PEOPLE_DICT)
