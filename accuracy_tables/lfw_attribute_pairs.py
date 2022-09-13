import os
import Config
from shutil import copy2
import numpy as np
from random import randrange
from tqdm import tqdm
import argparse

args = Config.parse_arguments('embedding')
params = Config.set_parameters(model_type=args.model_type,
                               loss_type=args.loss_type,
                               dataset_type=args.dataset_type,
                               batch_size=args.batch_size,
                               adversarial_flag=args.adversarial_flag,
                               folder=args.folder,
                               all_flag=args.all_flag,
                               uniform_flag=args.uniform_flag)

attributes = np.load(os.path.join(Config.ROOT, 'data', 'embeddings', 'attributes', 'attributes-{}.npz'.format(params['all_name'])), allow_pickle=True)['attributes'].item()
lfw = np.load(os.path.join(Config.ROOT, 'data', 'embeddings', 'attributes', 'embeddings-{}.npz'.format(params['all_name'])), allow_pickle=True)['embeddings'].item()

startstring = os.path.join(Config.ROOT, 'data', 'pairs', params['folder'])
if not os.path.exists(startstring):
	os.mkdir(startstring)
startstring = os.path.join(startstring, '{}_pairs_'.format(params['folder']))
if params['all_flag'] and params['uniform_flag']:
	outstring = startstring + 'all_uniform_{}.txt'
elif params['all_flag']:
	outstring = startstring + 'all_{}.txt'
elif params['uniform_flag']:
	outstring = startstring + 'uniform_{}.txt'
else:
	outstring = startstring + '{}.txt'


def test_loop(size, same, people):
	for j in range(0, size):
		p = randrange(len(people))
		p1 = randrange(len(lfw[people[p]]))
		p2 = randrange(len(lfw[people[p]]))
		count = 0
		while '{}-{}-{}'.format(p, p1, p2) in same or p1 == p2:
			p = randrange(len(people))
			p1 = randrange(len(lfw[people[p]]))
			p2 = randrange(len(lfw[people[p]]))
			count += 1
			if count > 100000:
				return True
		same['{}-{}-{}'.format(p, p1, p2)] = 0
		same['{}-{}-{}'.format(p, p2, p1)] = 0
	return False


def loop_same(size, same, people):
	for j in range(0, size):
		p = randrange(len(people))
		p1 = randrange(len(lfw[people[p]]))
		p2 = randrange(len(lfw[people[p]]))
		count = 0
		while '{}-{}-{}'.format(p, p1, p2) in same or p1 == p2:
			p = randrange(len(people))
			p1 = randrange(len(lfw[people[p]]))
			p2 = randrange(len(lfw[people[p]]))
		same['{}-{}-{}'.format(p, p1, p2)] = 0
		same['{}-{}-{}'.format(p, p2, p1)] = 0
		outfile.write('{}\t{}\t{}\n'.format(people[p], p1+1, p2+1))


def loop_diff(size, diff, people, all_people=None):
	if all_people == None:
		people1 = people
	else:
		people1 = all_people
	for j in range(0, size):
		r1 = randrange(len(people))
		r2 = randrange(len(people1))
		r11 = randrange(len(lfw[people[r1]]))
		r22 = randrange(len(lfw[people1[r2]]))
		while '{}-{}-{}-{}'.format(r1, r11, r2, r22) in diff or people[r1] == people1[r2]:
			r1 = randrange(len(people))
			r2 = randrange(len(people1))
			r11 = randrange(len(lfw[people[r1]]))
			r22 = randrange(len(lfw[people1[r2]]))
		diff['{}-{}-{}-{}'.format(r1, r11, r2, r22)] = 0
		diff['{}-{}-{}-{}'.format(r2, r22, r1, r11)] = 0
		outfile.write('{}\t{}\t{}\t{}\n'.format(people[r1], r11+1, people1[r2], r22+1))


# get only pairs between people in the same attribute
# get all possible pairs involving people within an attribute with all other people
all_people = list(lfw.keys())
for attr, people in attributes.items():
	with open(outstring.format(attr.replace(' ', '_')), 'w') as outfile:
		size = 1000
		print(attr)
		failure = True
		while failure:
			failure = False
			same = {}
			for i in range(0, 10):
				if test_loop(size, same, people):
					failure = True
					break
			size = int(size/2)

		if params['uniform_flag']:
			if size < 62:
				continue
			size = 50

		outfile.write('{}\t{}\n'.format(10, size))
		same = {}
		diff = {}
		for i in range(0, 10):
			loop_same(size, same, people)
			if params['all_flag']:
				loop_diff(size, diff, people, all_people)
			else:
				loop_diff(size, diff, people)
