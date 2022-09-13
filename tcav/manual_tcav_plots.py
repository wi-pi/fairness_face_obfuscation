import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import Config


MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
ALL_PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[5], MUTED[9]])
RACE_PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[6], MUTED[9]])
SEX_PAL = sns.color_palette([MUTED[4], MUTED[8]])

name_dict = {'Brown Male': 'n000157', 'Medium Female': 'n006265', 'Black Female': 'n005234', 'Light Male': 'n008647'}

names = np.load(os.path.join(Config.ROOT, 'tcav', 'rcmalli_vggface_labels_v2.npy'))
indices = []
with open(os.path.join(Config.ROOT, 'tcav', 'vgglabels.txt'), 'r') as infile:
    for line in infile:
        indices.append(line.strip())

plt.style.use('seaborn-muted')

sns.set(font_scale=1.5, style='ticks', palette=RACE_PAL)

tcav_scores = [0.25, 0, 0, 0, 0,
               0.34, 0, 0, 0, 0,
               0, 0, 0, 0.43, 0,
               0, 0, 0, 0.65, 0,
               0, 0.66, 0, 0, 0.32,
               0.31, 0, 0, 0, 0,
               0, 0, 0, 0, 0.62,
               0.37, 0, 0, 0.6, 0,
               0.38, 0, 0, 0, 0,
               0, 0, 0, 0.68, 0,
               0.28, 0, 0, 0, 0,
               0, 0, 0, 0.67, 0,
               0, 0, 0, 0, 0.62,]
attributes = ['n000157', 'n000157', 'n000157', 'n000157', 'n000157', 
              'n005234', 'n005234', 'n005234', 'n005234', 'n005234', 
              'n006265', 'n006265', 'n006265', 'n006265', 'n006265',
              'n008647', 'n008647', 'n008647', 'n008647', 'n008647',
              'n009084', 'n009084', 'n009084', 'n009084', 'n009084',
              'n000359', 'n000359', 'n000359', 'n000359', 'n000359',
              'n000769', 'n000769', 'n000769', 'n000769', 'n000769',
              'n001372', 'n001372', 'n001372', 'n001372', 'n001372',
              'n002150', 'n002150', 'n002150', 'n002150', 'n002150',
              'n003536', 'n003536', 'n003536', 'n003536', 'n003536',
              'n004011', 'n004011', 'n004011', 'n004011', 'n004011',
              'n004066', 'n004066', 'n004066', 'n004066', 'n004066',
              'n006362', 'n006362', 'n006362', 'n006362', 'n006362',]
skin_tone = ['6', '4', '3', '2', '1',
             '6', '4', '3', '2', '1',
             '6', '4', '3', '2', '1',
             '6', '4', '3', '2', '1',
             '6', '4', '3', '2', '1',
             '6', '4', '3', '2', '1',
             '6', '4', '3', '2', '1',
             '6', '4', '3', '2', '1',
             '6', '4', '3', '2', '1',
             '6', '4', '3', '2', '1',
             '6', '4', '3', '2', '1',
             '6', '4', '3', '2', '1',
             '6', '4', '3', '2', '1',]
yerr = [0.044, 0, 0, 0, 0,
        0.083, 0, 0, 0, 0,
        0, 0, 0, 0.045, 0,
        0, 0, 0, 0.04, 0,
        0, 0.042, 0, 0, 0.026,
        0.033, 0, 0, 0, 0,
        0, 0, 0, 0, 0.029,
        0.025, 0, 0, 0.025, 0,
        0.061, 0, 0, 0, 0,
        0, 0, 0, 0.098, 0,
        0.054, 0, 0, 0, 0,
        0, 0, 0, 0.023, 0,
        0, 0, 0, 0, 0.02,]

df = pd.DataFrame({'Attributes': attributes,'Skin Tone': skin_tone, 'TCAV Scores': tcav_scores, 'precomputed_std': yerr})

dfCopy = df.loc[df.index.repeat(1000)].copy()
dfCopy['TCAV Scores'] = np.random.normal(dfCopy['TCAV Scores'].values, dfCopy['precomputed_std'].values)
# print(dfCopy[:5])
plot = sns.barplot(data=dfCopy, x='Skin Tone', y='TCAV Scores', estimator=np.mean, hue='Attributes', ci='sd')
plot.legend().remove()
plt.ylim(0.15, 0.8)
for bar in plot.patches:
    print(bar)
    if bar.get_height() < 0.1:
        plot.text(bar.xy[0], bar.get_height(), '*', fontdict = {'weight': 'bold', 'size': 16,
            'color': bar.get_facecolor()})
sns.despine()
plt.savefig(os.path.join(Config.DATA, 'embeddings', 'plots', 'face-tcav.png'), bbox_inches='tight')
