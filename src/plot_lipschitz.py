import Config
import os
import argparse
import numpy as np
from utils.eval_utils import load_attributes
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2", "#AED4C9", "#59599C"]
ALL_PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4], MUTED[8]])
RACE_PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[6], MUTED[9]])
SEX_PAL = sns.color_palette([MUTED[4], MUTED[8]])
OTHER_PAL = sns.color_palette([MUTED[10], MUTED[11]])

plt.style.use('seaborn-muted')
# sns.set(font_scale=1.5, style='ticks', palette='muted')

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
parser.add_argument('--flag-type', default='min')
args = parser.parse_args()
sns.set(font_scale=1.5, style='ticks', palette=OTHER_PAL, rc={'figure.figsize':(8,6)})
ALL_LAYERS = 'all_layers-'

DATASET = args.dataset
LAYERBOUND = args.layerbound
JACBOUND = args.jacbound
LIPSTEPS = '20'
EPS = '0.1'
FLAG_TYPE = args.flag_type
propernames = {'asian': 'Asian', 'Asian': 'Asian', 'indian': 'Indian', 'Indian': 'Indian', 'black': 'Black', 'Black': 'Black',
               'white': 'White', 'White': 'White', 'male': 'Male', 'Male': 'Male', 'female': 'Female', 'NOTMale': 'Female',
               'middle': 'Mid-East', 'latino': 'Latino', 'bal': 'Uniform', 'imbal': 'Non-Uniform'}

if 'vgg' in DATASET:
    constants = []
    distribution = []
    classes = []

    if 'race' in args.dataset:
        attributes = Config.LIPSCHITZ_ATTRIBUTES
        cur_attr = 'race'
    elif 'sex' in args.dataset:
        attributes = Config.TCAVSEX_ATTRIBUTES
        cur_attr = 'sex'

    for b in ['imbal', 'bal']:
        labels = []
        with open(os.path.join(Config.DATA, 'vgg-{}-{}/finetune-vgg-{}-{}.txt'.format(b, cur_attr, b, cur_attr)), 'r') as infile:
            for line in infile:
                labels.append(line.strip())
            attribute_people, _ = load_attributes({'folder': 'vggface2'})
            DEMOGRAPHIC_CLASSES = []
            for i in range(len(attributes)):
                DEMOGRAPHIC_CLASSES.append([])
            for label in labels:
                for attr in attributes:
                    if label in attribute_people[attr]:
                        DEMOGRAPHIC_CLASSES[attributes.index(attr)].append(label)

#                DEMOGRAPHIC_CLASSES = [['n000875', 'n009084', 'n004551', 'n002127', 'n004806'],
#                                       ['n000223', 'n004185', 'n002483', 'n002560', 'n008647'],
#                                       ['n000359', 'n001198', 'n002150', 'n002837', 'n004976'],
#                                       ['n000157', 'n004843', 'n008004', 'n006265', 'n007239']]
#            elif 'sex' in args.dataset:
#                attributes = Config.TCAVSEX_ATTRIBUTES
#                DEMOGRAPHIC_CLASSES = [['n000223', 'n004806', 'n001985', 'n002560', 'n008293', 'n004088', 'n009084', 'n002150', 'n006752', 'n008647'],
#                                       ['n000359', 'n001866', 'n002483', 'n002837', 'n003308', 'n004066', 'n004185', 'n004843', 'n005774', 'n006265']]
#            elif propernames[b] == 'imbalanced':
#                attributes = Config.TCAVRACE_ATTRIBUTES
#                DEMOGRAPHIC_CLASSES = [['n000875'],
#                                       ['n000223', 'n000254', 'n002483', 'n002560', 'n004066', 'n004185', 'n006628', 'n006725', 'n008647'],
#                                       ['n000359'],
#                                       ['n000157']]
#            elif propernames[b] == 'balanced':
#                attributes = Config.TCAVRACE_ATTRIBUTES
#                DEMOGRAPHIC_CLASSES = [['n000875', 'n002127', 'n004514'],
#                                       ['n000223', 'n000254', 'n002483'],
#                                       ['n000359', 'n001198', 'n002150'],
#                                       ['n000157', 'n004843', 'n006212']]

        for i, demographic in enumerate(DEMOGRAPHIC_CLASSES):
            max_max_local_lipschitz = 0
            for person in demographic:
                for sample in range(0, 3):
                    with open(os.path.join(Config.DATA, 'fileio', 'lipschitz_constant-{}vgg-{}-{}--{}-{}-{}-{}-{}-{}.txt'.format(ALL_LAYERS, b, cur_attr, LAYERBOUND, JACBOUND, LIPSTEPS, EPS, labels.index(person), sample)), 'r') as infile:
                        for line in infile:
                            if line.startswith('[L0]'):
                                split = line.split('numimage = 1,')[1].split('avg_lipschitz[')[1:-1]
                                for l in split:
                                    if FLAG_TYPE == 'avg':
                                        constants.append(float(l.split('] = ')[1].split(', ')[0]))
                                        distribution.append(propernames[b].capitalize())
                                        classes.append(propernames[attributes[i]])
                                if FLAG_TYPE == 'global':
                                    constants.append(float(line.split('opnorm_global_lipschitz = ')[1].strip()))
                                    distribution.append(propernames[b].capitalize())
                                    classes.append(propernames[attributes[i]])
                            elif line.startswith('[L1]'):
                                split = line.split('lipschitz_min = ')[1].split(', lipschitz_max = ')
                                if FLAG_TYPE == 'min':
                                    constants.append(float(split[0]))
                                    distribution.append(propernames[b].capitalize())
                                    classes.append(propernames[attributes[i]])
                                if FLAG_TYPE == 'max':
                                    constants.append(float(split[1].split(', margin = ')[0]))
                                    max_max_local_lipschitz = max(max_max_local_lipschitz, float(split[1].split(', margin = ')[0]))
                                    distribution.append(propernames[b].capitalize())
                                    classes.append(propernames[attributes[i]])
            print(b, attributes[i], max_max_local_lipschitz)
    df = pd.DataFrame({'Attributes': classes, 'Local Lipschitz Constants': constants, 'Distribution': distribution})
    print("saving csv")
    df.to_csv("Lipschitz_summary.csv")
    plot = sns.boxplot(y="Local Lipschitz Constants", x="Attributes", hue="Distribution", data=df, showfliers=False)
    leg = plot.legend()
    leg.set_bbox_to_anchor([0.55, 1.03])
    #plt.ylim(0, 0.80*10e9)
    plt.tight_layout()
    sns.despine()
    plot.get_figure().savefig(os.path.join(Config.DATA, 'embeddings', 'plots', 'lipschitz_violin-{}{}-{}-{}-{}.png'.format(ALL_LAYERS, DATASET, LAYERBOUND, JACBOUND, FLAG_TYPE)), bbox_inches='tight')

else:
    constants = []
    distribution = []
    classes = []
    attributes = {1: 'Majority', 2: 'Majority', 4: 'Majority', 5: 'Majority', 8: 'Majority', 0: 'Minority', 3: 'Minority', 6: 'Minority', 7: 'Minority', 9: 'Minority'}
    for b in ['imbal', 'bal']:
        for labels in [[1, 2, 4, 5, 8], [0, 3, 6, 7, 9]]:
            for label in labels:
                for sample in range(0, 30):
                    with open(os.path.join(Config.DATA, 'fileio', 'lipschitz_constant-{}{}-{}-{}-{}-{}-{}-{}-{}.txt'.format(ALL_LAYERS, DATASET, b, LAYERBOUND, JACBOUND, LIPSTEPS, EPS, label, sample)), 'r') as infile:
                        for line in infile:
                            if line.startswith('[L0]'):
                                split = line.split('numimage = 1,')[1].split('avg_lipschitz[')[1:-1]
                                for l in split:
                                    if FLAG_TYPE == 'avg':
                                        constants.append(float(l.split('] = ')[1].split(', ')[0]))
                                        distribution.append(propernames[b].capitalize())
                                        classes.append(attributes[label])
                                if FLAG_TYPE == 'global':
                                    constants.append(float(line.split('opnorm_global_lipschitz = ')[1].strip()))
                                    distribution.append(propernames[b].capitalize())
                                    classes.append(attributes[label])
                            elif line.startswith('[L1]'):
                                split = line.split('lipschitz_min = ')[1].split(', lipschitz_max = ')
                                if FLAG_TYPE == 'min':
                                    constants.append(float(split[0]))
                                    distribution.append(propernames[b].capitalize())
                                    classes.append(attributes[label])
                                if FLAG_TYPE == 'max':
                                    constants.append(float(split[1].split(', margin = ')[0]))
                                    distribution.append(propernames[b].capitalize())
                                    classes.append(attributes[label])
    df = pd.DataFrame({'Classes': classes, 'Local Lipschitz Constants': constants, 'Distribution': distribution})
    print("SAVING CSV")

    df.to_csv("Lipschitz_summary_else.csv")

    plot = sns.boxplot(y="Local Lipschitz Constants", x="Classes", hue="Distribution", data=df, showfliers=False)
    lgnd = plt.legend(loc='upper left')
    plt.tight_layout()
    sns.despine()
    plot.get_figure().savefig(os.path.join(Config.DATA, 'embeddings', 'plots', 'lipschitz_violin-{}{}-{}-{}-{}.png'.format(ALL_LAYERS, DATASET, LAYERBOUND, JACBOUND, FLAG_TYPE)), bbox_inches='tight')
