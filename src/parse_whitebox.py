import csv, os
import numpy as np
from tqdm import tqdm
import Config


# look at the specific targets (and their demographics) and see whether they make a difference


if __name__ == '__main__':
    params = Config.parse_and_configure_arguments()
    tf_config = Config.set_gpu(params['gpu'])
   
    filename = 'whitebox_eval_{}-{}-{}.csv'.format(params['model'], params['target_model'], params['attack_name'])

    UNTARGETED = {'Asian': 0, 'White': 0, 'Black': 0, 'Indian': 0, 'Male': 0, 'NOTMale': 0}
    TARGETED = {'Asian': 0, 'White': 0, 'Black': 0, 'Indian': 0, 'Male': 0, 'NOTMale': 0}
    TOP5 = {'Asian': 0, 'White': 0, 'Black': 0, 'Indian': 0, 'Male': 0, 'NOTMale': 0}
    UNSUCCESSFUL = {'Asian': 0, 'White': 0, 'Black': 0, 'Indian': 0, 'Male': 0, 'NOTMale': 0}
    TOP5_AT_LEAST_1= {'Asian': 0, 'White': 0, 'Black': 0, 'Indian': 0, 'Male': 0, 'NOTMale': 0}
    TOP5_ALL_SAME = {'Asian': 0, 'White': 0, 'Black': 0, 'Indian': 0, 'Male': 0, 'NOTMale': 0}
    TOTALS = {'Asian': 0, 'White': 0, 'Black': 0, 'Indian': 0, 'Male': 0, 'NOTMale': 0}
    untargeted = 0
    targeted = 0
    topn = 0
    total = 0

    meta_fields = ['untargeted_overall', 'targeted_overall', 'top5_overall', 'male_count', 'NOTMale_count', 'white_count', 'asian_count', 'black_count', 'indian_count']
    fieldnames = ['demographic', 'untargeted', 'targeted', 'top5', 'unsuccessful_occurrence', 'top5_at_least_1_same_occurrence', 'top5_all_same_occurrence']

    infile = open(os.path.join(Config.DATA, 'fileio/whitebox_eval_{}'.format(filename)), 'r')
    metafile = open(os.path.join(Config.DATA, 'fileio/whitebox_out_meta_{}'.format(filename)), 'w')
    outfile = open(os.path.join(Config.DATA, 'fileio/whitebox_out_{}'.format(filename)), 'w')
    attributes = np.load(os.path.join(Config.DATA, 'embeddings/attributes/attributes-{}.npz'.format(params['all_name'])), allow_pickle=True)['attributes'].item()

    reader = csv.reader(infile)
    meta_writer = csv.DictWriter(metafile, fieldnames=meta_fields)
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    meta_writer.writeheader()
    writer.writeheader()

    for i, row in tqdm(enumerate(reader)):
        if i != 0:
            if row[5] == 'False':
                untargeted += 1
            if row[6] == 'True':
                targeted += 1
            if row[4] != row[13] and row[4] != row[17] and row[4] != row[21] and row[4] != row[25] and row[4] != row[29]:
                topn += 1
            for a, _ in TOTALS.items():
                if row[4] in attributes[a]:
                    TOTALS[a] += 1
                    if row[4] == row[13] or row[4] == row[17] or row[4] == row[21] or row[4] == row[25] or row[4] == row[29]:
                        TOP5[a] += 1
                    if row[13] in attributes[a] and row[17] in attributes[a] and row[21] and attributes[a] and row[25] and attributes[a] and row[29] in attributes[a]:
                      TOP5_ALL_SAME[a] += 1
                    if row[5] == 'False':
                        UNTARGETED[a] += 1
                    if row[6] == 'True':
                        TARGETED[a] += 1
                    if row[13] in attributes[a] or row[17] in attributes[a] or row[21] or attributes[a] or row[25] or attributes[a] or row[29] in attributes[a]:
                        TOP5_AT_LEAST_1[a] += 1
                    if row[4] != row[13] and row[4] != row[17] and row[4] != row[21] and row[4] != row[25] and row[4] != row[29]:
                        UNSUCCESSFUL[a] += 1
            total += 1
        else:
            print(row)
    print(untargeted/total)
    print(targeted/total)
    print(topn/total)

    meta_writer.writerow({'untargeted_overall': untargeted/total*100,
                          'targeted_overall': targeted/total*100,
                          'top5_overall': topn/total*100,
                          'male_count': TOTALS['Male'],
                          'NOTMale_count': TOTALS['NOTMale'],
                          'white_count': TOTALS['White'],
                          'asian_count': TOTALS['Asian'],
                          'black_count': TOTALS['Black'],
                          'indian_count': TOTALS['Indian']})

    for key, val in TOTALS.items():
        writer.writerow({'demographic': key,
                         'untargeted': UNTARGETED[key]/val*100,
                         'targeted': TARGETED[key]/val*100,
                         'top5': TOP5[key]/val*100,
                         'unsuccessful_occurrence': UNSUCCESSFUL[key]/val*100,
                         'top5_at_least_1_same_occurrence': TOP5_AT_LEAST_1[key]/val*100,
                         'top5_all_same_occurrence': TOP5_ALL_SAME[key]/val*100})
