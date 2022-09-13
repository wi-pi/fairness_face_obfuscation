import os
import ast
import Config
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
ALL_PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4], MUTED[5]])
# LAYERS = ['all', 'Block35_3_Activation', 'Block35_1_Activation', 'add_10']
# LAYERS = ['all', 'Block35_3_Activation', 'Block35_1_Activation', 'add_10', 'Block35_4_Activation',
# 'add_6', 'Block17_3_Activation', 'Block8_3_Activation', 'Mixed_6a', 'Block17_4_Activation']
#LAYERS = ['Conv2d_1a_3x3_Activation', 'Conv2d_2a_3x3_Activation', 'Conv2d_2a_3x3_Activation', 'Conv2d_2b_3x3_Activation', 
#          'Conv2d_3b_1x1_Activation', 'Conv2d_4a_3x3_Activation', 'Conv2d_4b_3x3_Activation', 'Block35_1_Activation', 'Block35_2_Activation',
#          'Block35_3_Activation', 'Block35_4_Activation', 'Block35_5_Activation', 'Block17_1_Activation', 'Block17_2_Activation',
#          'Block17_3_Activation', 'Block17_4_Activation', 'Block17_5_Activation', 'Block17_6_Activation', 'Block17_7_Activation',
#          'Block17_8_Activation', 'Block17_9_Activation', 'Block17_10_Activation', 'Block8_1_Activation', 'Block8_2_Activation',
#          'Block8_3_Activation', 'Block8_4_Activation', 'Block8_5_Activation']
LAYERS = ['Block17_9_Activation', 'Block35_3_Activation', 'Block35_5_Activation', 'Conv2d_2a_3x3_Activation']

identities = {}
with open('vggskins.csv', 'r') as infile:
    for line in infile:
        split = line.strip().split(',')
        identities[split[0]] = int(split[1].replace('type', ''))

count = 0
df_data = {'Skin Tone': [], 'Identity': [], 'Layer': [], 'TCAV Score': [], 'mode': []}
for file in os.listdir('tcav/tcav_scores/'):
    if '-early-gradient-relative' in file:
        mode = '-early-gradient-relative'
    elif '-best_early-gradient-relative' in file:
        mode = '-best_early-gradient-relative'
    elif '-early_single-gradient-relative' in file:
        mode = '-early_single-gradient-relative'
    elif '-good-gradient-relative' in file:
        mode = '-good-gradient-relative'
    else:
        mode = '-all-gradient-relative'
    identity = file.replace('Facenet-skin-skin-', '').replace(mode, '')
    if identity not in identities:
        continue
    skintone = identities[identity]
    with open(os.path.join('tcav/tcav_scores', file), 'r') as infile:
        for line in infile:
            line = line.replace('dict_items', '').strip()
            layer_dict = ast.literal_eval(line)
            for item in layer_dict:
                df_data['Layer'].append(item[0])
                df_data['mode'].append(mode)

                significants = item[1]['significant']
                df_data['TCAV Score'].append(item[1]['bn_vals'][skintone - 1])
                
                #if significants[skintone - 1]:
                #    df_data['TCAV Score'].append(item[1]['bn_vals'][skintone - 1])
                #elif significants[min(skintone, len(significants) - 1)]:
                #    df_data['TCAV Score'].append(item[1]['bn_vals'][min(skintone, len(significants) - 1)])
                #elif significants[max(skintone - 2, 0)]:
                #    df_data['TCAV Score'].append(item[1]['bn_vals'][max(skintone - 2, 0)])
                #else:
                #    df_data['TCAV Score'].append(0)

                df_data['Identity'].append(identity)
                df_data['Skin Tone'].append('Type {}'.format(skintone))
            break

for layer in LAYERS:
    df = pd.DataFrame(df_data)
    if layer != 'all':
        df = df.where(df['Layer'] == layer)
        # df = df.where(df['mode'] == '-best_early-gradient-relative')
        # df = df.where(df['mode'] == '-all-gradient-relative')
        df = df.where(df['mode'] == '-good-gradient-relative')

    sns.set(font_scale=3.5, style='ticks', palette=ALL_PAL)
    plt.figure(figsize=(16, 12))
    plot = sns.histplot(data=df, x='TCAV Score', hue='Skin Tone', stat='percent', hue_order=['Type 1', 'Type 2', 'Type 3', 'Type 4', 'Type 5', 'Type 6'], multiple='dodge')
    plt.tight_layout()
    old_legend = plot.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    plot.legend(handles, labels, loc='upper right', title=title).get_frame().set_alpha(0.5)
    sns.despine()
    plot.get_figure().savefig(os.path.join(Config.DATA, 'embeddings', 'plots', 'tcav-skintones-{}.png'.format(layer)), bbox_inches='tight')
