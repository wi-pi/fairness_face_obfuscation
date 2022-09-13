import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from random import randrange
from utils.eval_utils import *
import Config
import os
import csv
import math
import seaborn as sns


MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
ALL_PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4], MUTED[8]])
RACE_PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3]])
SEX_PAL = sns.color_palette([MUTED[4], MUTED[8]])


def runthings(params, df_data):
    if TOPN:
        if params['adversarial_flag']:
            csvfile = open(os.path.join(Config.DATA, 'fileio', 'performance_topn_{}-{}.csv'.format(params['adv_dir_folder'],
                    params['target_model'])), 'r', newline='')
            reader = csv.reader(csvfile)

            adversarial = {}
            names = []
            for i, row in tqdm(enumerate(reader)):
                if i == 0:
                    for j, col in enumerate(row):
                        names.append(col)
                        adversarial[col] = []
                elif i == 1:
                    for j, col in enumerate(row):
                        adversarial[names[j]] = float(col)

            width = 0.25
            for i in names:
                if 'skewpos' in i:
                    df_data['distribution'].append('K-Majority')
                elif 'balance' in i:
                    df_data['distribution'].append('Balance')
                elif 'skewneg' in i:
                    df_data['distribution'].append('K-Minority')
                df_data['topk_accuracy'].append(adversarial[i])
                attribute = i.split('-')[1]
                df_data['labels'].append(attribute)
                if params['different_flag']:
                    df_data['targets'].append('Diff')
                else:
                    df_data['targets'].append('Same')
            return df_data
    else:
        if params['adversarial_flag']:
            csvfile = open(os.path.join(Config.DATA, 'fileio', 'performance_{}-{}.csv'.format(params['adv_dir_folder'],
                    params['target_model'])), 'r', newline='')
            reader = csv.reader(csvfile)

            adversarial = {}
            names = {}
            for i, row in tqdm(enumerate(reader)):
                if i == 0:
                    for j, col in enumerate(row):
                        if MEAN and 'adv' in col:
                            adversarial[col] = []
                        elif not MEAN and 'adv' not in col:
                            adversarial[col] = []
                        names[j] = col
                else:
                    for j, col in enumerate(row):
                        if MEAN and 'adv' in names[j]:
                            adversarial[names[j]].append(float(col))
                        elif not MEAN and 'adv' not in names[j]:
                            adversarial[names[j]].append(float(col))

            for key, val in adversarial.items():
                if 'adv' in key:
                    # k = key.replace('adv', 'mean')
                    k = ' Mean '
                else:
                    # k = key
                    k = ''
                # plt.plot(params['thresholds'], val, label = '{}_adv'.format(k))
                df_data['thresholds'].extend(params['thresholds'])
                df_data['matching_accuracy'].extend(val)
                df_data['labels'].extend([key] * len(val))
                if params['different_flag']:
                    df_data['targets'].extend(['Diff'] * len(val))
                else:
                    df_data['targets'].extend(['Same'] * len(val))
            return df_data
        else:
            csvfile = open(os.path.join(Config.DATA, 'fileio', 'performance_{}.csv'.format(params['all_name'])), 'r', newline='')
            # csvfile = open(os.path.join(Config.DATA, 'fileio', 'performance_{}.csv'.format(params['all_name'].replace(params['attack_name'], '').replace(params['model'], params['target_model']))), 'r', newline='')
            reader = csv.reader(csvfile)

            natural = {}
            names = {}
            for i, row in tqdm(enumerate(reader)):
                if i == 0:
                    for j, col in enumerate(row):
                        if MEAN and 'adv' in col:
                            natural[col] = []
                        elif not MEAN and 'adv' not in col:
                            natural[col] = []
                        names[j] = col
                else:
                    for j, col in enumerate(row):
                        if MEAN and 'adv' in names[j]:
                            natural[names[j]].append(float(col))
                        elif not MEAN and 'adv' not in col:
                            natural[names[j]].append(float(col))
            for key, val in natural.items():
                if 'adv' in key:
                    k = key.replace('adv', 'mean')
                else:
                    k = key
                plt.plot(params['thresholds'], val, label = k)
            plt.xlabel('Thresholds')
            plt.ylabel('Accuracy')
            plt.title(params['all_name'])
            plt.legend()
            plt.savefig(os.path.join(Config.DATA, 'embeddings', 'plots', 'performance-{}.png'.format(params['all_name'])))


def plotthings(df):
    if TOPN:
        g = sns.catplot(data=df, col='Targets', x='Distribution', y='Top-K Accuracy', hue='Attribute', order=['K-Majority', 'Balance', 'K-Minority'],
            kind='bar', legend_out=False, hue_order=HUE_ORDER)
        low = max(min(df['Top-K Accuracy']) - 0.05, 0)
        high = max(df['Top-K Accuracy'])
        plt.ylim([low, high])
        plt.tight_layout()
        sns.despine()
        g.fig.get_axes()[0].legend(loc='upper left')
        g.fig.get_axes()[0].set_title('')
        g.fig.get_axes()[1].set_title('')
        plt.savefig(os.path.join(Config.DATA, 'embeddings', 'plots', 'performance-topn-{}-{}.png'.format(params['adv_dir_folder'], params['target_model'])), bbox_inches='tight')
    else:
        g = sns.relplot(data=df, col='Targets', x='Thresholds', y='Matching Accuracy', hue='Attribute', kind='line', legend='auto', hue_order=HUE_ORDER)
        plt.tight_layout()
        sns.despine()
        leg = g._legend
        leg.set_bbox_to_anchor([0.08, 0.9])
        leg._loc = 2
        # g.fig.get_axes()[0].legend(loc='upper left')
        g.fig.get_axes()[0].set_title('')
        g.fig.get_axes()[1].set_title('')
        plt.savefig(os.path.join(Config.DATA, 'embeddings', 'plots', 'performance-{}-{}.png'.format(params['adv_dir_folder'], params['target_model'])), bbox_inches='tight')


TOPN = False
MEAN = False

args = Config.parse_arguments('embedding')
params = Config.set_parameters(model=args.model,
                               target_model=args.target_model,
                               attack=args.attack,
                               mean_loss=args.mean_loss,
                               amplification=args.amplification,
                               folder=args.folder,
                               adversarial_flag=args.adversarial_flag,
                               whitebox_target=args.whitebox_target,
                               attribute='race',
                               different_flag='true',
                               correct_flag=args.correct_flag)
df_data = {'labels': [], 'topk_accuracy': [], 'distribution': [], 'attribute_names': [], 'targets': [], 'demographic': [], 'matching_accuracy': [], 'thresholds': []}
df_data = runthings(params, df_data)
params = Config.set_parameters(model=args.model,
                               target_model=args.target_model,
                               attack=args.attack,
                               mean_loss=args.mean_loss,
                               amplification=args.amplification,
                               folder=args.folder,
                               adversarial_flag=args.adversarial_flag,
                               whitebox_target=args.whitebox_target,
                               attribute='race',
                               different_flag='false',
                               correct_flag=args.correct_flag)
df_data = runthings(params, df_data)
params = Config.set_parameters(model=args.model,
                               target_model=args.target_model,
                               attack=args.attack,
                               mean_loss=args.mean_loss,
                               amplification=args.amplification,
                               folder=args.folder,
                               adversarial_flag=args.adversarial_flag,
                               whitebox_target=args.whitebox_target,
                               attribute='sex',
                               different_flag='true',
                               correct_flag=args.correct_flag)
df_data = runthings(params, df_data)
params = Config.set_parameters(model=args.model,
                               target_model=args.target_model,
                               attack=args.attack,
                               mean_loss=args.mean_loss,
                               amplification=args.amplification,
                               folder=args.folder,
                               adversarial_flag=args.adversarial_flag,
                               whitebox_target=args.whitebox_target,
                               attribute='sex',
                               different_flag='false',
                               correct_flag=args.correct_flag)
df_data = runthings(params, df_data)
plt.style.use('seaborn-muted')
color_pal = ALL_PAL
HUE_ORDER = ['Asian', 'Black', 'Indian', 'White', 'Male', 'Female']
sns.set(font_scale=1.5, style='ticks', palette=color_pal)
propernames = {'asian': 'Asian', 'Asian': 'Asian', 'indian': 'Indian', 'Indian': 'Indian', 'black': 'Black', 'Black': 'Black',
               'white': 'White', 'White': 'White', 'male': 'Male', 'Male': 'Male', 'female': 'Female', 'NOTMale': 'Female',
               'middle': 'Mid East', 'latino': 'Latino', 'ALL': 'All', 'all': 'All'}
for i in df_data['labels']:
    df_data['attribute_names'].append(propernames[i])

if TOPN:
    df = pd.DataFrame({'Top-K Accuracy': df_data['topk_accuracy'], 'Distribution': df_data['distribution'], 'Attribute': df_data['attribute_names'],
        'Targets': df_data['targets']})
else:
    df = pd.DataFrame({'Matching Accuracy': df_data['matching_accuracy'], 'Attribute': df_data['attribute_names'],
        'Targets': df_data['targets'], 'Thresholds': df_data['thresholds']})
plotthings(df)

