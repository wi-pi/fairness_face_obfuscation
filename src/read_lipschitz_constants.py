import Config
import os
import argparse
import numpy as np
from utils.eval_utils import load_attributes


# attributes = Config.VGGRACE_ATTRIBUTES
# demographics = Config.VGGRACE_SOURCES

# attributes = Config.VGGSEX_ATTRIBUTES
# demographics = Config.VGGSEX_SOURCES

# params = Config.set_parameters(folder='vggface2')
# attributes, _ = load_attributes(params)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist')
parser.add_argument('--jacbound', default='recurjac')
parser.add_argument('--layerbound', default='fastlin-interval')

args = parser.parse_args()

ALL_LAYERS = 'all_layers-'

DATASET = args.dataset
LAYERBOUND = args.layerbound
JACBOUND = args.jacbound
LIPSTEPS = '80'
EPS = '0.1'


if 'vgg' in DATASET:
    if 'imbalanced' in args.dataset:
        BALANCED = 'imbal'
    elif 'balanced' in args.dataset:
        BALANCED = 'bal'
    else:
        BALANCED = ''

    if 'race' in args.dataset:
        attributes = Config.TCAVRACE_ATTRIBUTES
        DEMOGRAPHIC_CLASSES = [['n000875', 'n009084', 'n004551', 'n002127', 'n004806'],
                               ['n000223', 'n004185', 'n002483', 'n002560', 'n008647'],
                               ['n000359', 'n001198', 'n002150', 'n002837', 'n004976'],
                               ['n000157', 'n004843', 'n008004', 'n006265', 'n007239']]
    elif 'sex' in args.dataset:
        attributes = Config.TCAVSEX_ATTRIBUTES
        DEMOGRAPHIC_CLASSES = [['n000223', 'n004806', 'n001985', 'n002560', 'n008293', 'n004088', 'n009084', 'n002150', 'n006752', 'n008647'],
                               ['n000359', 'n001866', 'n002483', 'n002837', 'n003308', 'n004066', 'n004185', 'n004843', 'n005774', 'n006265']]
    elif 'imbalanced' in args.dataset:
        attributes = Config.TCAVRACE_ATTRIBUTES
        DEMOGRAPHIC_CLASSES = [['n000875'],
                               ['n000223', 'n000254', 'n002483', 'n002560', 'n004066', 'n004185', 'n006628', 'n006725', 'n008647'],
                               ['n000359'],
                               ['n000157']]
    elif 'balanced' in args.dataset:
        attributes = Config.TCAVRACE_ATTRIBUTES
        DEMOGRAPHIC_CLASSES = [['n000875', 'n002127', 'n004514'],
                               ['n000223', 'n000254', 'n002483'],
                               ['n000359', 'n001198', 'n002150'],
                               ['n000157', 'n004843', 'n006212']]

    labels = []
    with open(os.path.join(Config.DATA, '{}/finetune-{}.txt'.format(DATASET, DATASET)), 'r') as infile:
        for line in infile:
            labels.append(line.strip())
    with open(os.path.join(Config.DATA, 'fileio', 'lipschitz_output-{}{}-{}-{}-{}-{}-{}.txt'.format(ALL_LAYERS, DATASET, BALANCED, LAYERBOUND, JACBOUND, LIPSTEPS, EPS)), 'w') as outfile:
        for i, demographic in enumerate(DEMOGRAPHIC_CLASSES):
            constants = []
            minimums = []
            maximums = []
            globalconst = []
            for person in demographic:
                try:
                    with open(os.path.join(Config.DATA, 'fileio', 'lipschitz_constant-{}{}-{}-{}-{}-{}-{}-{}.txt'.format(ALL_LAYERS, DATASET, BALANCED, LAYERBOUND, JACBOUND, LIPSTEPS, EPS, labels.index(person))), 'r') as infile:
                        for line in infile:
                            if line.startswith('[L0]'):
                                split = line.split('numimage = 3,')[1].split('avg_lipschitz[')[1:-1]
                                for l in split:
                                    constants.append(float(l.split('] = ')[1].split(', ')[0]))
                                globalconst.append(float(line.split('opnorm_global_lipschitz = ')[1].strip()))
                            elif line.startswith('[L1]'):
                                split = line.split('lipschitz_min = ')[1].split(', lipschitz_max = ')
                                minimums.append(float(split[0]))
                                maximums.append(float(split[1].split(', margin = ')[0]))
                except Exception as e:
                    continue
            outfile.write('{}: avg={:.2f}, avgmin={:.2f}, avgmax={:.2f}, avgglobal={:.2f}, minmin={:.2f}, maxmax={:.2f}\n'.format(attributes[i], np.mean(constants), np.mean(minimums), np.mean(maximums), np.mean(globalconst), min(minimums), max(maximums)))
            print('{}: avg={}, avgmin={}, avgmax={}, avgglobal={}'.format(attributes[i], np.mean(constants), np.mean(minimums), np.mean(maximums), np.mean(globalconst)))
else:
    for BALANCED in ['bal', 'imbal']:
        with open(os.path.join(Config.DATA, 'fileio', 'lipschitz_output-{}{}-{}-{}-{}-{}-{}.txt'.format(ALL_LAYERS, DATASET, BALANCED, LAYERBOUND, JACBOUND, LIPSTEPS, EPS)), 'w') as outfile:
            for labels in [[1, 2, 4, 5, 8], [0, 3, 6, 7, 9]]:
                globalconst = []
                constants = []
                minimums = []
                maximums = []
                for label in labels:
                    try:
                        with open(os.path.join(Config.DATA, 'fileio', 'lipschitz_constant-{}{}-{}-{}-{}-{}-{}-{}.txt'.format(ALL_LAYERS, DATASET, BALANCED, LAYERBOUND, JACBOUND, LIPSTEPS, EPS, label)), 'r') as infile:
                            for line in infile:
                                if line.startswith('[L0]'):
                                    split = line.split('numimage = 3,')[1].split('avg_lipschitz[')[1:-1]
                                    for l in split:
                                        constants.append(float(l.split('] = ')[1].split(', ')[0]))
                                    globalconst.append(float(line.split('opnorm_global_lipschitz = ')[1].strip()))
                                elif line.startswith('[L1]'):
                                    split = line.split('lipschitz_min = ')[1].split(', lipschitz_max = ')
                                    minimums.append(float(split[0]))
                                    maximums.append(float(split[1].split(', margin = ')[0]))
                    except Exception as e:
                        continue
                outfile.write('{}: avg={:.2f}, avgmin={:.2f}, avgmax={:.2f}, avgglobal={:.2f}, minmin={:.2f}, maxmax={:.2f}\n'.format(labels, np.mean(constants), np.mean(minimums), np.mean(maximums), np.mean(globalconst), min(minimums), max(maximums)))
                print('{}: avg={}, avgmin={}, avgmax={}, avgglobal={}'.format(labels, np.mean(constants), np.mean(minimums), np.mean(maximums), np.mean(globalconst)))
