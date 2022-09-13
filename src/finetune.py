# from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
# import torch
# from torch.utils.data import DataLoader, SubsetRandomSampler
# from torch import optim
# from torch.optim.lr_scheduler import MultiStepLR
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms
import numpy as np
import os
import imageio
import cv2
import Config
import argparse
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from keras_vggface.vggface import VGGFace
from models.keras_vggface.vggface import BiasVGGFace
import tensorflow as tf


CLASSES = ['n000157', 'n001198', 'n002483', 'n004066', 'n004806', 'n005774', 'n006628', 'n007233', 'n008342', 'n000223', 'n001866', 'n002560', 'n004088',
'n004843', 'n006059', 'n006717', 'n007239', 'n008394', 'n000254', 'n001985', 'n002837', 'n004185', 'n004976', 'n006212', 'n006725', 'n007470', 'n008647',
'n000359', 'n002127', 'n003308', 'n004514', 'n005234', 'n006265', 'n006752', 'n008004', 'n009058', 'n000875', 'n002150', 'n004011', 'n004551', 'n005679',
'n006469', 'n006758', 'n008293', 'n009084']


def extract_data(dataset='vggface2'):
    data = []
    labels = []
    new_labels = []
    one_hot = []
    if dataset == 'vggface2':
        labelfile = 'vgglabels.txt'
    elif dataset == 'celeba':
        labelfile = 'celebalabels.txt'
    elif dataset == 'vggracebalanced' or 'vggraceimbalanced':
        labelfile = 'vggracelabels.txt'
    elif dataset == 'vggsexbalanced' or 'vggseximbalanced':
        labelfile = 'vggsexlabels.txt'
    elif dataset == 'vggbalanced' or 'vggimbalanced':
        labelfile = 'vgg12labels.txt'
    elif dataset == 'vgg-bal-race':
        labelfile = ''
    with open(os.path.join(Config.DATA, dataset, labelfile), 'r') as infile:
      for line in infile:
        labels.append(line.strip())
    ROOT = os.path.join(Config.DATA, dataset, '{}-align-160'.format(dataset))
    for person in tqdm(os.listdir(ROOT)):
        path = os.path.join(ROOT, person)
        target = labels.index(person)
        file_list = []
        for file in os.listdir(path):
            file_list.append(os.path.join(path, file))
        data.extend(read_face_from_aligned(file_list))
        new_labels.extend([target] * len(os.listdir(path)))
    data = np.array(data)
    for l in new_labels:
        temp = np.zeros(8631)
        temp[int(target)] = 1
        one_hot.append(temp)
    one_hot = np.array(one_hot)
    return data, one_hot


# def generate_batches(image_gen, path, class_mode, color_mode, batch_size, target_size, shuffle, seed):
#     for x, y in image_gen.flow_from_directory(path,
#         class_mode=class_mode,
#         color_mode=color_mode,
#         batch_size=batch_size,
#         target_size=target_size,
#         shuffle=shuffle,
#         seed=seed):
#         x = x[..., ::-1]
#         x[..., 0] -= CHANNEL0
#         x[..., 1] -= CHANNEL1
#         x[..., 2] -= CHANNEL2
#         yield (x, y)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='vggface2')
parser.add_argument('--model', default='vgg16')
parser.add_argument('--gpu', default='4,5,6,7')
parser.add_argument('--from-scratch', default='false')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

#custom parameters
if args.dataset == 'vggface2':
    nb_class = 8631
elif args.dataset == 'celeba':
    nb_class = 8192
elif args.dataset == 'vggracebalanced' or args.dataset == 'vggraceimbalanced' or args.dataset == 'vggsexbalanced' or args.dataset == 'vggseximbalanced':
    nb_class = 20
elif args.dataset == 'vggbalanced' or args.dataset == 'vggimbalanced':
    nb_class = 12
elif args.dataset == 'vgg-imbal-race' or args.dataset == 'vgg-bal-race' or args.dataset == 'vgg-imbal-sex' or args.dataset == 'vgg-bal-sex':
    nb_class = 600

hidden_dim = 512

if args.from_scratch == 'false':
    print('\n\n\nTRAINING WITH PRETRAINED\n\n\n')
    if args.model == 'vgg16':
        vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
        last_layer = vgg_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(hidden_dim, activation='relu', name='fc6')(x)
        x = Dense(hidden_dim, activation='relu', name='fc7')(x)
        out = Dense(nb_class, activation='softmax', name='classifier')(x)
        model = Model(vgg_model.input, out)
        CHANNEL0 = 93.5940
        CHANNEL1 = 104.7624
        CHANNEL2 = 129.1863
    elif args.model == 'resnet50':
        vgg_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))
        last_layer = vgg_model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(hidden_dim, activation='relu', name='fc6')(x)
        x = Dense(hidden_dim, activation='relu', name='fc7')(x)
        out = Dense(nb_class, activation='softmax', name='classifier')(x)
        model = Model(vgg_model.input, out)
        CHANNEL0 = 91.4953
        CHANNEL1 = 103.8827
        CHANNEL2 = 131.0912
else:
    print('\n\n\nTRAINING FROM SCRATCH\n\n\n')
    if args.model == 'vgg16':
        vgg_model = BiasVGGFace(weights=None, include_top=False, input_shape=(224, 224, 3), use_bias=True)
        last_layer = vgg_model.get_layer('pool5').output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(hidden_dim, activation='relu', name='fc6')(x)
        x = Dense(hidden_dim, activation='relu', name='fc7')(x)
        out = Dense(nb_class, activation='softmax', name='classifier')(x)
        model = Model(vgg_model.input, out)
        CHANNEL0 = 93.5940
        CHANNEL1 = 104.7624
        CHANNEL2 = 129.1863
    elif args.model == 'resnet50':
        vgg_model = BiasVGGFace(weights=None, model='resnet50', include_top=False, input_shape=(224, 224, 3), use_bias=True)
        last_layer = vgg_model.get_layer('avg_pool').output
        x = Flatten(name='flatten')(last_layer)
        out = Dense(nb_class, activation='softmax', name='classifier')(x)
        model = Model(vgg_model.input, out)
        CHANNEL0 = 91.4953
        CHANNEL1 = 103.8827
        CHANNEL2 = 131.0912

csv_logger = CSVLogger('{}-{}.csv'.format(args.dataset, args.model), append=True, separator=';')

callback = ModelCheckpoint(
    filepath='weights/best/{}-{}.hdf5'.format(args.dataset, args.model),
    monitor="val_categorical_accuracy",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
)

# sgd = Adam(lr = 0.01)
# model.compile(
#     optimizer=sgd,
#     loss=CategoricalCrossentropy(),
#     metrics=[CategoricalAccuracy()],
# )

ds = ImageDataGenerator()

x_train = ds.flow_from_directory(
    os.path.join(Config.DATA, args.dataset, '{}-align-160'.format(args.dataset)),
    class_mode='categorical',
    color_mode="rgb",
    batch_size=32,
    target_size=(224, 224),
    shuffle=False,
    seed=2222
)
x_val = ds.flow_from_directory(
    os.path.join(Config.DATA, args.dataset, '{}-val-align-160'.format(args.dataset)),
    class_mode='categorical',
    color_mode="rgb",
    batch_size=32,
    target_size=(224, 224),
    shuffle=False,
    seed=2222
)

# history = model.fit_generator(
#     x_train,
#     epochs=200,
#     validation_data=x_val,
#     callbacks=[callback]
# )

# x_train = generate_batches(ds,
#     os.path.join(Config.DATA, args.dataset, '{}-align-224'.format(args.dataset)),
#     class_mode='categorical',
#     color_mode="rgb",
#     batch_size=32,
#     target_size=(224, 224),
#     shuffle=False,
#     seed=2222
# )
# x_val = generate_batches(ds,
#     os.path.join(Config.DATA, args.dataset, '{}-val-align-224'.format(args.dataset)),
#     class_mode='categorical',
#     color_mode="rgb",
#     batch_size=32,
#     target_size=(224, 224),
#     shuffle=False,
#     seed=2222
# )

#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.5, decay_steps=10000, decay_rate=0.9)

print(x_train.class_indices)

with open(os.path.join(Config.DATA, args.dataset, 'finetune-{}.txt'.format(args.dataset)), 'w') as outfile:
    count = 0
    for key, val in sorted(x_train.class_indices.items()):
        outfile.write('{}\n'.format(key))

epochs = 35
steps_per_epoch = x_train.samples//32
validation_steps = x_val.samples//32
train = {}

opt_adm = Adadelta()
model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=[CategoricalAccuracy()])
train["part_1"] = model.fit_generator(x_train, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                                    verbose=1, validation_data=x_val, callbacks=[callback, csv_logger])
model.save("weights/{}_generic_first.h5".format(args.dataset))
print(train["part_1"].history)

# Training for 25 epochs more - (50-75)
opt_adm = Adadelta(lr=0.7, rho=0.9)
model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=[CategoricalAccuracy()])
train["part_2"] = model.fit_generator(x_train, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                                    verbose=1, validation_data=x_val, callbacks=[callback, csv_logger])
model.save("weights/{}_generic_second.h5".format(args.dataset))
print(train["part_2"].history)

# Training for 25 epochs more - (75-100)
opt_adm = Adadelta(lr=0.5, rho=0.85)
model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=[CategoricalAccuracy()])
train["part_3"] = model.fit_generator(x_train, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                                    verbose=1, validation_data=x_val, callbacks=[callback, csv_logger])
model.save("weights/{}_generic_third.h5".format(args.dataset))
print(train["part_3"].history)

# Training for 25 epochs more  - (100-125)
opt_adm = Adadelta(lr=0.3, rho=0.75)
model.compile(loss='categorical_crossentropy', optimizer=opt_adm, metrics=[CategoricalAccuracy()])
train["part_4"] = model.fit_generator(x_train, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                                    verbose=1, validation_data=x_val, callbacks=[callback, csv_logger])
model.save("weights/{}_generic_fourth.h5".format(args.dataset))
print(train["part_4"].history)
print("\n \n Final Logs: ", train)




# data_dir = '../data/test_images'

# batch_size = 32
# epochs = 8
# workers = 0 if os.name == 'nt' else 8



# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# print('Running on device: {}'.format(device))
# mtcnn = MTCNN(
#     image_size=160, margin=0, min_face_size=20,
#     thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
#     device=device
# )
# dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
# dataset.samples = [
#     (p, p.replace(data_dir, data_dir + '_cropped'))
#         for p, _ in dataset.samples
# ]
        
# loader = DataLoader(
#     dataset,
#     num_workers=workers,
#     batch_size=batch_size,
#     collate_fn=training.collate_pil
# )

# for i, (x, y) in enumerate(loader):
#     mtcnn(x, save_path=y)
#     print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
    
# # Remove mtcnn to reduce GPU memory usage
# del mtcnn
# resnet = InceptionResnetV1(
#     classify=True,
#     pretrained='vggface2',
#     num_classes=len(dataset.class_to_idx)
# ).to(device)

# optimizer = optim.Adam(resnet.parameters(), lr=0.001)
# scheduler = MultiStepLR(optimizer, [5, 10])

# trans = transforms.Compose([
#     np.float32,
#     transforms.ToTensor(),
#     fixed_image_standardization
# ])
# dataset = datasets.ImageFolder(data_dir + '_cropped', transform=trans)
# img_inds = np.arange(len(dataset))
# np.random.shuffle(img_inds)
# train_inds = img_inds[:int(0.8 * len(img_inds))]
# val_inds = img_inds[int(0.8 * len(img_inds)):]

# train_loader = DataLoader(
#     dataset,
#     num_workers=workers,
#     batch_size=batch_size,
#     sampler=SubsetRandomSampler(train_inds)
# )
# val_loader = DataLoader(
#     dataset,
#     num_workers=workers,
#     batch_size=batch_size,
#     sampler=SubsetRandomSampler(val_inds)
# )

# loss_fn = torch.nn.CrossEntropyLoss()
# metrics = {
#     'fps': training.BatchTimer(),
#     'acc': training.accuracy
# }

# writer = SummaryWriter()
# writer.iteration, writer.interval = 0, 10

# print('\n\nInitial')
# print('-' * 10)
# resnet.eval()
# training.pass_epoch(
#     resnet, loss_fn, val_loader,
#     batch_metrics=metrics, show_running=True, device=device,
#     writer=writer
# )

# for epoch in range(epochs):
#     print('\nEpoch {}/{}'.format(epoch + 1, epochs))
#     print('-' * 10)

#     resnet.train()
#     training.pass_epoch(
#         resnet, loss_fn, train_loader, optimizer, scheduler,
#         batch_metrics=metrics, show_running=True, device=device,
#         writer=writer
#     )

#     resnet.eval()
#     training.pass_epoch(
#         resnet, loss_fn, val_loader,
#         batch_metrics=metrics, show_running=True, device=device,
#         writer=writer
#     )

# writer.close()
