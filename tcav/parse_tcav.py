import os
import Config
import argparse
from models.face_models import get_model


SOURCE_DIR = 'tcav/concepts'
CAV_DIR = 'tcav/cavs'
ACTIVATION_DIR = 'tcav/activations'

def add(dictionary, key):
    if key not in dictionary:
        dictionary[key] = 1
    else:
        dictionary[key] += 1
    return dictionary

if __name__ == '__main__':
    fileio = os.path.join(Config.DATA, 'fileio')
    bottlenecks = {}
    concepts = {}
    targets = {}
    combo = {}
    for file in os.listdir(fileio):
        infile = os.path.join(fileio, file)
        if file.startswith('tcav_out_Facenet') and 'many' in file and 'gender' not in file:
            with open(infile, 'r') as text:
                data = {}
                for line in text:
                    line = line.strip()
                    if line == '===========================':
                        data = {}
                    elif line.startswith('cav_concept, '):
                        data['cav_concept'] = line.split('cav_concept, ')[1]
                    elif line.startswith('negative_concept, '):
                        data['negative_concept'] = line.split('negative_concept, ')[1]
                    elif line.startswith('target_class, '):
                        data['target_class'] = line.split('target_class, ')[1]
                    elif line.startswith('cav_accuracies, '):
                        data['cav_accuracies'] = float(line.split("'overall': ")[1].replace('}', ''))
                    elif line.startswith('bottleneck, '):
                        data['bottleneck'] = line.split('bottleneck, ')[1]
                    if len(list(data.keys())) == 5 and data['cav_accuracies'] >= 0.8:
                        bottlenecks = add(bottlenecks, data['bottleneck'])
                        concepts = add(concepts, data['cav_concept'])
                        concepts = add(concepts, data['negative_concept'])
                        combo = add(combo, '{}:{}'.format(data['cav_concept'], data['negative_concept']))
                        targets = add(targets, data['target_class'])
    for key, val in bottlenecks.items():
        print('{}: {}'.format(key, val))
    for key, val in concepts.items():
        print('{}: {}'.format(key, val))
    for key, val in targets.items():
        print('{}: {}'.format(key, val))
    # for key, val in combo.items():
    #     print('{}: {}'.format(key, val))
