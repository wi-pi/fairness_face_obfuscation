import numpy as np
import os
import time
import Config
import argparse
from FaceAPI.microsoft_face import verify_API
from FaceAPI.aws_rekognition import aws_compare
from FaceAPI.facepp import facepp
from FaceAPI import credentials
import csv
from utils.eval_utils import *
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2", "#AED4C9", "#59599C"]
ALL_PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4], MUTED[8]])
RACE_PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3]])
SEX_PAL = sns.color_palette([MUTED[4], MUTED[8]])
# OTHER_PAL = sns.color_palette([MUTED[10], MUTED[11]])
plt.style.use('seaborn-muted')
# LOWKEY_ATTRIBUTE = 'race'
LOWKEY_ATTRIBUTE = 'race'


def runthings(df_data, params, attributes):
    propernames = {'asian': 'Asian', 'Asian': 'Asian', 'indian': 'Indian', 'Indian': 'Indian', 'black': 'Black', 'Black': 'Black',
                   'white': 'White', 'White': 'White', 'male': 'Male', 'Male': 'Male', 'female': 'Female', 'NOTMale': 'Female',
                   'middle': 'Mid East', 'latino': 'Latino', 'ALL': 'All'}
    all_attributes = params['attributes']
    people_attr = params['sources']
    if params['attack'] != 'lowkey' and params['targeted_success'] == True:
        infile = open(os.path.join(Config.DATA, 'fileio', '{}_targeted_{}.csv'.format(params["api_name"], params['adv_dir_folder'])), 'r', newline='')
    if params['targeted_success'] == False:
        infile = open(os.path.join(Config.DATA, 'fileio', '{}_untargeted_{}.csv'.format(params["api_name"], params['adv_dir_folder'])), 'r', newline='')
    reader = csv.reader(infile)

    successes_dict = {}
    for attr in HUE_ORDER:
        successes_dict[attr] = []
    for i, row in tqdm(enumerate(reader)):
        if i > 0:
            if params['attack'] == 'lowkey':
                for attr in HUE_ORDER:
                    if row[0] in attributes[attr.replace('Female', 'NOTMale')]:
                        if params['api_name'] == 'awsverify' or params['api_name'] == 'facepp':
                            successes_dict[attr].append(float(row[2]) / 100.0)
                        else:
                            successes_dict[attr].append(float(row[2]))
            else:
                for j, attr in enumerate(all_attributes):
                    if row[0] in people_attr[j]:
                        if params['api_name'] == 'awsverify' or params['api_name'] == 'facepp':
                            successes_dict[propernames[attr]].append(float(row[2]) / 100.0)
                        else:
                            successes_dict[propernames[attr]].append(float(row[2]))

    for thresh in tqdm(params['thresholds']):
        for key, val in successes_dict.items():
            total = 0
            count = 0
            for i in val:
                if not params['targeted_success'] or params['attack'] == 'lowkey':
                    if i <= thresh:
                        count += 1
                else:
                    if i >= thresh:
                        count += 1
                total += 1
            df_data['attributes'].append(key)
            df_data['scores'].append(count / total)
            df_data['thresholds'].append(thresh)

    return df_data


def plotthings(df, targeted_string, params):
    g = sns.relplot(data=df, x='Confidence', y='Obfuscation Success', hue='Attribute', kind='line', hue_order=HUE_ORDER)
    # g = sns.displot(data=df, col='Targets', x='L2 Norm', kind='ecdf', hue='Attribute', hue_order=HUE_ORDER)
    # sns.displot(data=df, col='Targets', x='L2 Norm', kind='hist', hue='Attribute', hue_order=HUE_ORDER)
    plt.tight_layout()
    sns.despine()
    sns.move_legend(g, 'center left', bbox_to_anchor=[0.2, 0.5])
    # if params['attack'] == 'lowkey':
    #     g.set(xlabel=None)
    #     g.tick_params(bottom=False)
    #     g.set(xticklabels=[])
    # g.fig.get_axes()[1].set_title('')
    # plt.title('ECDF {} {} {}'.format(params['folder'].upper(), params['attribute'].capitalize(), samediff))
    # plt.ylim(0, 15)
    # if params['api_name'] == 'facepp':
    #     g.axhline(0.69101, ls='--', c='#DC7EC0')
    # else:
    #     g.axhline(0.5, ls='--', c='#DC7EC0')
    # plt.legend([],[], frameon=False)
    plt.savefig(os.path.join(Config.DATA, 'embeddings', 'plots', '{}-success-{}-{}.png'.format(params["api_name"], targeted_string, params['adv_dir_folder'])), bbox_inches='tight')


if __name__ == "__main__":
    params = Config.parse_and_configure_arguments()
    tf_config = Config.set_gpu(params['gpu'])
    df_data = {'scores': [], 'attributes': [], 'targets': [], 'thresholds': []}
    params['thresholds'] = np.arange(0.0, 1, 0.01)
    if params['attack'] == 'lowkey':
        if LOWKEY_ATTRIBUTE == 'race':
            color_pal = RACE_PAL
            HUE_ORDER = ['Asian', 'Black', 'Indian', 'White']
        elif LOWKEY_ATTRIBUTE == 'sex':
            color_pal = SEX_PAL
            HUE_ORDER = ['Male', 'Female']
    else:
        if params['attribute'] == 'race':
            color_pal = RACE_PAL
            HUE_ORDER = ['Asian', 'Black', 'Indian', 'White']
        elif params['attribute'] == 'sex':
            color_pal = SEX_PAL
            HUE_ORDER = ['Male', 'Female']
        else:
            color_pal = ALL_PAL
            HUE_ORDER = ['Asian', 'Black', 'Indian', 'White', 'Male', 'Female']
    attributes, ppl = load_attributes(params)
    df_data = runthings(df_data, params, attributes)

    sns.set(font_scale=1.5, style='ticks', palette=color_pal)
    
    if params['targeted_success']:
        targeted_string = 'targeted'
    else:
        targeted_string = 'untargeted'
    df = pd.DataFrame({'Obfuscation Success': df_data['scores'], 'Attribute': df_data['attributes'], 'Confidence': df_data['thresholds']})
    # print(df)
    plotthings(df, targeted_string, params)

