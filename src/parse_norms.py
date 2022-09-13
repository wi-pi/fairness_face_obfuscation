import csv, os
import numpy as np
import pandas as pd
from tqdm import tqdm
import Config
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns


MUTED=["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2", "#AED4C9", "#59599C"]
ALL_PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3], MUTED[4], MUTED[8]])
RACE_PAL = sns.color_palette([MUTED[0], MUTED[1], MUTED[2], MUTED[3]])
SEX_PAL = sns.color_palette([MUTED[4], MUTED[8]])
OTHER_PAL = sns.color_palette([MUTED[10], MUTED[11]])


def runthings(params, df_data):
    ATTRIBUTES = {'Asian': [], 'White': [], 'Black': [], 'Indian': [], 'Male': [], 'NOTMale': [], 'ALL': []}
    fieldnames = ['demographic', 'norm']
    print("filename", os.path.join(Config.DATA, 'fileio', 'norm_distance_{}.csv'.format(params['adv_dir_folder'])))
    outfile = open(os.path.join(Config.DATA, 'fileio', 'norm_out_{}.csv'.format(params['adv_dir_folder'])), 'w')
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    attributes = np.load(os.path.join(Config.DATA, 'embeddings', 'attributes', 'attributes-{}.npz'.format(params['folder'])), allow_pickle=True)['attributes'].item()
    
    with open(os.path.join(Config.DATA, 'fileio', 'norm_distance_{}.csv'.format(params['adv_dir_folder'])), 'r') as infile:
        reader = csv.reader(infile)
        for i, row in tqdm(enumerate(reader)):
            if i != 0:
                ATTRIBUTES['ALL'].append(float(row[3]))
                for a, _ in ATTRIBUTES.items():
                    if a != 'ALL':
                        if row[4] in attributes[a]:
                            ATTRIBUTES[a].append(float(row[3]))

    for key, val in ATTRIBUTES.items():
        average = 0
        average_same = 0
        average_different = 0
        for i in val:
            average += i
        if len(val) > 0:
            print('{}, {}'.format(key, average / len(val)))
            writer.writerow({'demographic': key,
                             'norm': average / len(val),})
    for key, val in ATTRIBUTES.items():
        if key != 'ALL' and ((params['attribute'] == 'sex' and key in ['NOTMale', 'Male']) or (params['attribute'] == 'race' and key in ['Asian', 'White', 'Black', 'Indian'])):
            for i in val:
                df_data['labels'].append(key)
                df_data['examples'].append(i)
                if params['different_flag']:
                    df_data['targets'].append('Diff')
                else:
                    df_data['targets'].append('Same')
    return df_data


def plotthings(df):
    kwargs = {'cumulative': True}
    g = sns.boxplot(data=df, x='Attribute', y='L2 Norm', hue='Targets', hue_order=HUE_ORDER, showfliers=False)
    # g = sns.displot(data=df, col='Targets', x='L2 Norm', kind='ecdf', hue='Attribute', hue_order=HUE_ORDER)
    # sns.displot(data=df, col='Targets', x='L2 Norm', kind='hist', hue='Attribute', hue_order=HUE_ORDER)
    plt.tight_layout()
    sns.despine()
    plt.legend(frameon=False)
    # g.legend().frameon = False
    # g.fig.get_axes()[0].set_title('')
    # g.fig.get_axes()[1].set_title('')
    # plt.title('ECDF {} {} {}'.format(params['folder'].upper(), params['attribute'].capitalize(), samediff))
    #plt.ylim(0, 4)
    plt.savefig(os.path.join(Config.DATA, 'embeddings', 'plots', 'perturbation-norm-cdf-{}.png'.format(params['adv_dir_folder'])), bbox_inches='tight')


OUTLIER = False
plt.style.use('seaborn-muted')

if __name__ == '__main__':
    params = Config.parse_and_configure_arguments()
    if params['source'] != None:
        adv_out_folder = '{}_{}_diff_{}_{}'.format(params['all_name'], params['attribute'], Config.LFWRACE_ATTRIBUTES[int(params['source'])], params['amplification'])
        ATTRIBUTES = {'Asian': [], 'White': [], 'Black': [], 'Indian': [], 'Male': [], 'NOTMale': [], 'ALL': []}
        fieldnames = ['demographic', 'norm']
        attributes = np.load(os.path.join(Config.DATA, 'embeddings', 'attributes', 'attributes-{}.npz'.format(params['folder'])), allow_pickle=True)['attributes'].item()
        for target in range(0, 4):
            print("TARGET",target)
            if target != int(params['source']):
                params['adv_dir_folder'] = '{}_{}_diff_{}_{}_{}'.format(params['all_name'], params['attribute'], Config.LFWRACE_ATTRIBUTES[target], Config.LFWRACE_ATTRIBUTES[int(params['source'])], params['amplification'])
                outfile = open(os.path.join(Config.DATA, 'fileio', 'norm_out_{}.csv'.format(params['adv_dir_folder'])), 'w')
                with open(os.path.join(Config.DATA, 'fileio', 'norm_distance_{}.csv'.format(params['adv_dir_folder'])), 'r') as infile:
                    reader = csv.reader(infile)
                    for i, row in tqdm(enumerate(reader)):
                        if i != 0:
                            ATTRIBUTES['ALL'].append(float(row[3]))
                            for a, _ in ATTRIBUTES.items():
                                if a != 'ALL':
                                    if row[4] in attributes[a]:
                                        ATTRIBUTES[a].append(float(row[3]))
            else:
                params['adv_dir_folder'] = '{}_{}_same_{}'.format(params['all_name'], params['attribute'], params['amplification'])
                with open(os.path.join(Config.DATA, 'fileio', 'norm_distance_{}.csv'.format(params['adv_dir_folder'])), 'r') as infile:
                    reader = csv.reader(infile)
                    for i, row in tqdm(enumerate(reader)):
                        if i != 0:
                            if row[4] in attributes[Config.LFWRACE_ATTRIBUTES[target]]:
                                ATTRIBUTES[Config.LFWRACE_ATTRIBUTES[target]].append(float(row[3]))

        def plot_stuff(key, val, title, pdf):
            if OUTLIER:
                mn, mx = 70, 120
            else:
                mn, mx = 0, 20
            # mn, mx = plt.xlim()
            if pdf:
                kde_xs = np.linspace(mn, mx, 300)
                kde = st.gaussian_kde(val)
                plt.plot(kde_xs, kde.pdf(kde_xs), label='PDF-{}'.format(key))
                plt.title('PDF-{}'.format(title))
                plt.ylim(0, 0.4)
            else:
                plt.figure()
                plt.hist(val, density=True, histtype='bar', bins=100, label=key)
                plt.title(title)
                if OUTLIER:
                    plt.ylim(0, 0.1)
                else:
                    plt.ylim(0, 1)
            plt.xlim(mn, mx)        
            plt.ylabel('Probability')
            plt.xlabel('L2 Norm Perturbation')
            plt.legend(loc='upper right')

        # labels = []
        # examples = []
        # for key, val in ATTRIBUTES.items():
        #     labels.append(key)
        #     examples.append(np.array(val))
        #     if not OUTLIER:
        #         plot_stuff(key, val, adv_out_folder, pdf=True)
        # if not OUTLIER:
        #     plt.savefig(os.path.join(Config.DATA, 'embeddings', 'plots', 'perturbation-norm-pdf-{}.png'.format(adv_out_folder)))
        # plot_stuff(labels, examples, adv_out_folder, pdf=False)
        # if OUTLIER:
        #     plt.savefig(os.path.join(Config.DATA, 'embeddings', 'plots', 'perturbation-norm-outliers-{}.png'.format(adv_out_folder)))
        # else:
        #     plt.savefig(os.path.join(Config.DATA, 'embeddings', 'plots', 'perturbation-norm-{}.png'.format(adv_out_folder)))
        labels = []
        examples = []
        for key, val in ATTRIBUTES.items():
            for i in val:
                labels.append(key)
                examples.append(i)

            # if not OUTLIER:
            #     plot_stuff(key, val, params['adv_dir_folder'], pdf=True)
        propernames = {'asian': 'Asian', 'Asian': 'Asian', 'indian': 'Indian', 'Indian': 'Indian', 'black': 'Black', 'Black': 'Black',
                       'white': 'White', 'White': 'White', 'male': 'Male', 'Male': 'Male', 'female': 'Female', 'NOTMale': 'Female',
                       'middle': 'Mid East', 'latino': 'Latino', 'ALL': 'All'}
        df = pd.DataFrame({'L2 Norm Perturbation': examples, 'y': labels})
        df['Attribute'] = None
        # if params['folder'] == 'lfw':
        #     attribute_names = ['Asian', 'Indian', 'Black', 'White', 'Male', 'NOTMale']
        # elif params['folder'] == 'vggface2':
        #     attribute_names = ['asian', 'indian', 'black', 'white', 'middle', 'latino', 'male', 'female']
        attribute_names = []
        for i in df.y:
            attribute_names.append(propernames[i])
        df['Attribute'] = attribute_names
        # sns.set(font_scale=2)
        # plt.figure(figsize=(16, 12))
        plot = sns.displot(data=df, x='L2 Norm Perturbation', kind='ecdf', hue='Attribute')
        if params['different_flag']:
            samediff = 'Different'
        else:
            samediff = 'Same'
        plt.title('ECDF {} {} {}'.format(params['folder'].upper(), params['attribute'].capitalize(), samediff))
        plt.xlim(0, 10)

        if not OUTLIER:
            plt.savefig(os.path.join(Config.DATA, 'embeddings', 'plots', 'perturbation-norm-cdf-{}.png'.format(adv_out_folder)))
    else:
        df_data = {'labels': [], 'examples': [], 'attribute_names': [], 'targets': []}
        params['adv_dir_folder'] = '{}_{}_diff_{}'.format(params['all_name'], params['attribute'], params['amplification'])
        params['different_flag'] = True
        df_data = runthings(params, df_data)
        params['adv_dir_folder'] = '{}_{}_same_{}'.format(params['all_name'], params['attribute'], params['amplification'])
        params['different_flag'] = False
        df_data = runthings(params, df_data)
        # if params['attribute'] == 'race':
        #     color_pal = RACE_PAL
        #     HUE_ORDER = ['Asian', 'Black', 'Indian', 'White']
        # elif params['attribute'] == 'sex':
        #     color_pal = SEX_PAL
        #     HUE_ORDER = ['Male', 'Female']
        # else:
        #     color_pal = ALL_PAL
        #     HUE_ORDER = ['Asian', 'Black', 'Indian', 'White', 'Male', 'Female']
        color_pal = OTHER_PAL
        HUE_ORDER = ['Diff', 'Same']
        sns.set(font_scale=1.5, style='ticks', palette=color_pal)
        propernames = {'asian': 'Asian', 'Asian': 'Asian', 'indian': 'Indian', 'Indian': 'Indian', 'black': 'Black', 'Black': 'Black',
                       'white': 'White', 'White': 'White', 'male': 'Male', 'Male': 'Male', 'female': 'Female', 'NOTMale': 'Female',
                       'middle': 'Mid East', 'latino': 'Latino', 'ALL': 'All'}
        for i in df_data['labels']:
            df_data['attribute_names'].append(propernames[i])
        df = pd.DataFrame({'L2 Norm': df_data['examples'], 'Attribute': df_data['attribute_names'], 'Targets': df_data['targets']})
        df.where(df['Attribute'] == 'Asian').dropna().to_csv('test_asian.csv')
        plotthings(df)
