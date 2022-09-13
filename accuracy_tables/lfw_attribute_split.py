import os
import Config
from shutil import copy2
import numpy as np


args = Config.parse_arguments('embedding')
params = Config.set_parameters(model_type=args.model_type,
                               loss_type=args.loss_type,
                               dataset_type=args.dataset_type,
                               adversarial_flag=args.adversarial_flag,
                               folder=args.folder)

attributes = np.load(os.path.join(Config.ROOT, 'data', 'embeddings', 'attributes', 'attributes-{}.npz'.format(params['all_name'])), allow_pickle=True)['attributes'].item()
src = os.path.join(Config.ROOT, 'data', params['folder'], params['folder'])
dst = os.path.join(Config.ROOT, 'data', params['folder'], 'attributes')

if not os.path.exists(dst):
	os.mkdir(dst)
for attr, people in attributes.items():
	if not os.path.exists(os.path.join(dst, attr)):
		os.mkdir(os.path.join(dst, attr))
	for p in people:
		if not os.path.exists(os.path.join(dst, attr, p)):
			os.mkdir(os.path.join(dst, attr, p))
		for i in os.listdir(os.path.join(src, p)):
			if os.path.isfile(os.path.join(src, p, i)):
				copy2(os.path.join(src, p, i), os.path.join(dst, attr, p, i))
