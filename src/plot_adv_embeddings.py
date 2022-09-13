import numpy as np
import pandas as pd
import sklearn.manifold
import joblib
# from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
from models.face_models import get_model
from utils.crop import *
from utils.eval_utils import *
import argparse


MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2", "#AED4C9", "#59599C"]
# ALL_PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[5], MUTED[9]])
RACE_PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[5], MUTED[6], MUTED[7], MUTED[9]])
SEX_PAL = sns.color_palette([MUTED[4], MUTED[8], MUTED[10], MUTED[11]])
# RACE_HUE = ['Asian', 'Black', 'Indian', 'White', 'AsianAdv', 'BlackAdv', 'IndianAdv', 'WhiteAdv']
RACE_HUE = ['Asian', 'Black', 'Indian', 'White']
# SEX_HUE = ['Male', 'Female', 'MaleAdv', 'FemaleAdv']
SEX_HUE = ['Male', 'Female']


plt.style.use('seaborn-muted')
params = Config.parse_and_configure_arguments()

lfw, lfw_all = load_target_embeddings(params)
lfw_adv, lfw_all_adv = load_embeddings(params)
attributes, _ = load_attributes(params)
embeddings = []
identities = []
adversarial = []
for key in lfw.keys():
    if len(lfw[key]) > 0:
        identities.append(key)
        # identities.extend([key,]*len(lfw[key]))
        embeddings.append(lfw[key])
        adversarial.append('Nat')
for key in lfw_adv.keys():
    if len(lfw_adv[key]) > 0:
        identities.append(key)
        embeddings.append(lfw_adv[key])
        adversarial.append('Adv')
# all_embeddings = np.concatenate(embeddings,axis=0)
tsne = TSNE(n_components=2, verbose=10, perplexity=40, n_iter=300, n_jobs=4)
embeddings_2d = tsne.fit_transform(embeddings)
df = pd.DataFrame({'identity': identities,'x': embeddings_2d[:,0], 'y': embeddings_2d[:,1], 'adversarial': adversarial})
df['Race'] = None
df['Sex'] = None
if params['folder'] == 'lfw':
    ethnicities = ['Asian', 'Indian', 'Black', 'White']
    sexes = ['Male', 'NOTMale']
    size = 8
elif params['folder'] == 'vggface2':
    ethnicities = ['asian', 'indian', 'black', 'white', 'middle', 'latino']
    sexes = ['male', 'female']
    size = 8

propernames = {'asian': 'Asian', 'Asian': 'Asian', 'indian': 'Indian', 'Indian': 'Indian', 'black': 'Black', 'Black': 'Black',
               'white': 'White', 'White': 'White', 'male': 'Male', 'Male': 'Male', 'female': 'Female', 'NOTMale': 'Female',
               'middle': 'Mid East', 'latino': 'Latino'}

# advnames = {'Asian': 'AsianAdv', 'Indian': 'IndianAdv', 'Black': 'BlackAdv', 'White': 'WhiteAdv', 'Male': 'MaleAdv', 'NOTMale': 'FemaleAdv'}

for ethnicity in ethnicities:
    df.loc[df.identity.isin(attributes[ethnicity]), 'Race']= propernames[ethnicity]
    # df.loc[df.adversarial.isin(['Adv']) & df.identity.isin(attributes[ethnicity]), 'Race'] = advnames[ethnicity]
for sex in sexes:
    df.loc[df.identity.isin(attributes[sex]), 'Sex']= propernames[sex]
    # df.loc[df.adversarial.isin(['Adv']) & df.identity.isin(attributes[sex]), 'Sex'] = advnames[sex]

sns.set(font_scale=1.5, style='ticks', palette=RACE_PAL)
# plt.figure(figsize=(16, 12))
plot = sns.relplot(data=df, x='x', y='y', hue='Race', col='adversarial', alpha=0.5, s=size, hue_order=RACE_HUE, legend='auto')
plot.fig.get_axes()[0].set_title('')
plot.fig.get_axes()[1].set_title('')
# leg = plot._legend
# leg.set_bbox_to_anchor([0.08, 0.94])
# leg._loc = 2
# leg.frameon = True
# leg.fancybox = True
# leg.get_frame().set_facecolor('white')

# lgnd = plt.legend(loc='upper left')
# lgnd = plot.fig.get_axes()[0].legend(loc='upper left')
# for lh in lgnd.legendHandles: 
#     lh.set_alpha(1)
#     lh.set_sizes([100])
# plt.title('T-SNE, {}: {}'.format(params['folder'].upper(), 'Race'))
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(Config.DATA, 'embeddings', 'plots', '{}.png'.format('tsne-{}-race'.format(params['all_name']))), bbox_inches='tight')

sns.set(font_scale=1.5, style='ticks', palette=SEX_PAL)
# plt.figure(figsize=(16, 12))
plot = sns.relplot(data=df, x='x', y='y', hue='Sex', col='adversarial', alpha=0.5,s=size, hue_order=SEX_HUE, legend='auto')
plot.fig.get_axes()[0].set_title('')
plot.fig.get_axes()[1].set_title('')
# leg = plot._legend
# leg.set_bbox_to_anchor([0.08, 0.94])
# leg._loc = 2
# leg.frameon = True
# leg.fancybox = True
# leg.get_frame().set_facecolor('white')
lgnd = plot.legend(loc='upper left')
# lgnd = plot.fig.get_axes()[0].legend(loc='upper left')
# for lh in lgnd.legendHandles: 
#     lh.set_alpha(1)
#     lh.set_sizes([100])
# plt.title('T-SNE, {}: {}'.format(params['folder'].upper(), 'Sex'))
plt.tight_layout()
sns.despine()
plt.savefig(os.path.join(Config.DATA, 'embeddings', 'plots', '{}.png'.format('tsne-{}-sex'.format(params['all_name']))), bbox_inches='tight')
