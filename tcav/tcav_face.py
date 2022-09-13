import absl
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile
import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.model  as model
import tcav.tcav as tcav
import tcav.utils_plot as utils_plot # utils_plot requires matplotlib
from tcav import utils
import os
import Config
import argparse
from models.face_models import get_model


CONCEPTS = {'race': ['asian', 'black', 'white', 'indian'],
            'gender': ['male', 'female'],
            #'skin': ['brown', 'tan', 'gold', 'peach', 'white']
            'skin': ['type1', 'type2', 'type3', 'type4', 'type5', 'type6']
}
# CONCEPTS = {'race': ['black'],
#             'gender': ['male', 'female']
# }
BOTTLNECKS = {'many': {
                'Facenet': ['Block35_1_Activation', 'add_1', 'Block35_4_Activation', 'Mixed_6a_Branch_1_Conv2d_0a_1x1',
                            'add_6', 'Block17_3_Activation', 'add_10', 'Block17_7_Activation', 'lambda_14', 'Block8_2_Concatenate',
                            'Block8_3_Activation', 'Block8_6_Concatenate', 'lambda_20', 'AvgPool', 'Bottleneck', 'Bottleneck_BatchNorm'],
                'resnet50': ['activation', 'max_pooling2d', 'activation_4', 'conv2_3_3x3', 'activation_14', 'add_4',
                             'activation_23', 'conv4_3_1x1_increase', 'activation_33', 'add_11', 'conv5_2_1x1_increase',
                             'add_15', 'activation_48', 'avg_pool', 'flatten', 'classifier']
              },
              'some': {
                'Facenet': ['Block35_4_Activation', 'Block17_7_Activation', 'Block8_2_Concatenate', 'lambda_20', 'Bottleneck_BatchNorm'],
                'resnet50': ['activation', 'activation_14', 'activation_33', 'add_15', 'classifier']
              },
              'good': {
                'Facenet': ['Block17_9_Activation', 'Block35_3_Activation', 'Block35_5_Activation', 'Conv2d_2a_3x3_Activation'],
                'resnet50': ['activation_33', 'add_4', 'activation_48', 'add_11']
              },
              'small': {
                'Facenet': ['Block17_4_Activation'],
                'resnet50': ['add_4']
              },
              'early': {
                'Facenet': ['Block35_1_Activation', 'Block35_4_Activation', 'add_6', 'Block17_3_Activation', 'add_10',
                            'Block8_3_Activation', 'Mixed_6a', 'Block17_4_Activation', 'Block35_3_Activation'],
                'resnet50': [],
                'vggimbalanced': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'conv3_3', 'pool3', 'conv4_2', 'convt5_1'],
                'vggbalanced': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'conv3_3', 'pool3', 'conv4_2', 'convt5_1'],
                'vggraceimbalanced': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'conv3_3', 'pool3', 'conv4_2', 'convt5_1'],
                'vggracebalanced': ['conv1_1', 'conv1_2', 'pool1', 'conv2_1', 'conv2_2', 'conv3_3', 'pool3', 'conv4_2', 'convt5_1']
              },
              'all': {
                'Facenet': ['Conv2d_1a_3x3_Activation', 'Conv2d_2a_3x3_Activation', 'Conv2d_2a_3x3_Activation', 'Conv2d_2b_3x3_Activation', 
                            'Conv2d_3b_1x1_Activation', 'Conv2d_4a_3x3_Activation', 'Conv2d_4b_3x3_Activation', 'Block35_1_Activation', 'Block35_2_Activation',
                            'Block35_3_Activation', 'Block35_4_Activation', 'Block35_5_Activation', 'Block17_1_Activation', 'Block17_2_Activation',
                            'Block17_3_Activation', 'Block17_4_Activation', 'Block17_5_Activation', 'Block17_6_Activation', 'Block17_7_Activation',
                            'Block17_8_Activation', 'Block17_9_Activation', 'Block17_10_Activation', 'Block8_1_Activation', 'Block8_2_Activation',
                            'Block8_3_Activation', 'Block8_4_Activation', 'Block8_5_Activation'],
                'resnet50': []
              },
              'early_single': {
                'Facenet': ['Block17_4_Activation']
              },
              'best_early': {
                'Facenet': ['Block35_3_Activation', 'Block35_1_Activation', 'add_10']
              }
}

SOURCE_DIR = 'tcav/concepts'
CAV_DIR = 'tcav/cavs'
ACTIVATION_DIR = 'tcav/activations'


def set_tcav_params(sess,
                    params,
                    relative_random,
                    bottlenecks,
                    alpha,
                    target_class,
                    feature,
                    concept,
                    mode):
    settings = {}
    settings['outname'] = '{}-{}-{}-{}-{}-{}-{}'.format(params['model'], concept, feature, target_class, bottlenecks, mode, relative_random)
    tcav_model = get_model(params)
    tcav_concepts = []
    if feature == 'all':
        for i in CONCEPTS[concept]:
            for j in ['mouth', 'forehead', 'eyes', 'eye', 'cheek']:
                tcav_concepts.append('{}-{}'.format(i, j))
    else:
        for i in CONCEPTS[concept]:
            tcav_concepts.append('{}-{}'.format(i, feature))
    
    settings['concepts'] = tcav_concepts
    settings['target'] = target_class
    settings['alpha'] = alpha
    settings['mode'] = mode

    if relative_random == 'relative':
        settings['random_concepts'] = tcav_concepts
        settings['relative'] = True
    else:
        settings['relative'] = False
        settings['random_concepts'] = ['male-forehead', 'male-mouth', 'black-eyes', 'white-mouth', 'white-eye', 'female-cheek', 'female-mouth', 'indian-eyes']
        # settings['random_concepts'] = ['white-forehead', 'indian-forehead']
    settings['num_random_exp'] = 50

    settings['bottlenecks'] = BOTTLNECKS[bottlenecks][params['model']]
    if params['model'] == 'resnet50':
        settings['model'] = model.KerasModelWrapper(sess,
                                                    'tcav/vgglabels.txt',
                                                    tcav_model,
                                                    settings['bottlenecks'])
    elif params['model'] == 'vggimbalanced':
        settings['model'] = model.KerasModelWrapper(sess,
                                                    'tcav/finetune-vggimbalanced.txt',
                                                    tcav_model,
                                                    settings['bottlenecks'])
    elif params['model'] == 'vggbalanced':
        settings['model'] = model.KerasModelWrapper(sess,
                                                    'tcav/finetune-vggbalanced.txt',
                                                    tcav_model,
                                                    settings['bottlenecks'])
    elif params['model'] == 'vggraceimbalanced':
        settings['model'] = model.KerasModelWrapper(sess,
                                                    'tcav/finetune-vggracebalanced.txt',
                                                    tcav_model,
                                                    settings['bottlenecks'])
    elif params['model'] == 'vggracebalanced':
        settings['model'] = model.KerasModelWrapper(sess,
                                                    'tcav/finetune-vggraceimbalanced.txt',
                                                    tcav_model,
                                                    settings['bottlenecks'])
    else:
        settings['model'] = model.KerasFaceModelWrapper(sess,
                                                        'weights/facenet_weights.h5',
                                                        'nothing',
                                                        tcav_model,
                                                        settings['bottlenecks'])
        # settings['model'] = model.KerasFaceModelWrapper(sess,
        #                                                 'weights/facenet_weights.h5',
        #                                                 'nothing',
        #                                                 tcav_model)
    return settings


if __name__ == '__main__':
    args = Config.parse_arguments()
    params = Config.set_parameters(args)
    Config.set_gpu(args.gpu)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            Config.BM.mark('set params')
            settings = set_tcav_params(sess=sess,
                                       params=params,
                                       relative_random=args.relative_random,
                                       bottlenecks=args.bottlenecks,
                                       alpha=args.alpha,
                                       target_class=args.target_class,
                                       feature=args.feature,
                                       concept=args.concept,
                                       mode=args.mode)
            Config.BM.mark('set params')
            Config.BM.mark('set activations')
            act_generator = act_gen.ImageActivationGenerator(settings['model'], SOURCE_DIR, ACTIVATION_DIR, max_examples=100)
            Config.BM.mark('set activations')
            absl.logging.set_verbosity(0)
            Config.BM.mark('init tcav')

            if settings['relative']:
                mytcav = tcav.TCAV(sess,
                                   settings['target'],
                                   settings['concepts'],
                                   settings['bottlenecks'],
                                   act_generator,
                                   [settings['alpha']],
                                   cav_dir=CAV_DIR,
                                   num_random_exp=settings['num_random_exp'],
                                   random_concepts=settings['random_concepts'])
            else:
                mytcav = tcav.TCAV(sess,
                                   settings['target'],
                                   settings['concepts'],
                                   settings['bottlenecks'],
                                   act_generator,
                                   [settings['alpha']],
                                   cav_dir=CAV_DIR,
                                   num_random_exp=settings['num_random_exp'],
                                   random_counterpart='white-forehead',
                                   random_concepts=settings['random_concepts'])
            Config.BM.mark('init tcav')
            Config.BM.mark('run tcav')
            results = mytcav.run(run_parallel=False, mode=settings['mode'])
            Config.BM.mark('run tcav')
            # print(results)
            count = 0
            # with open(os.path.join(Config.DATA, 'fileio', 'tcav_out_{}.txt'.format(settings['outname'])), 'w') as outfile:
            #     for i in results:
            #         outfile.write('{}\n===========================\n'.format(count))
            #         for key, val in i.items():
            #             outfile.write('{}, {}\n'.format(key, val))
            #         count += 1
    utils_plot.plot_results(results, num_random_exp=settings['num_random_exp'], random_concepts=settings['random_concepts'], outfile=settings['outname'])
