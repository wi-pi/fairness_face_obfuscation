import os
from shutil import copy2
from tqdm import tqdm
import Config


if __name__ == '__main__':
    params = Config.parse_and_configure_arguments()
    with open('tcav/vggskins.csv', 'r') as infile:
        for line in tqdm(infile):
            split = line.strip().split(',')
            person_path = os.path.join(params['align_dir'], split[0])
            file_list = os.listdir(person_path)
            if not os.path.exists(os.path.join('tcav/concepts', split[0])):
                os.mkdir(os.path.join('tcav/concepts', split[0]))
                if len(file_list) >= 100:
                    for i in file_list[:100]:
                        copy2(os.path.join(person_path, i), os.path.join('tcav/concepts', split[0], i))
