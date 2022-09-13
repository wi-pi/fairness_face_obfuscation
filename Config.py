import os, cv2, argparse
import tensorflow as tf
import numpy as np
from timeit import default_timer as timer
from utils.constants import *

class Benchmark:
    """Stores the start and end times of execution for performance metrics."""
    def __init__(self):
        self.start = {}
        self.end = {}

    def mark(self, message=''):
        """
        Stores the start or end time depending on which call.
        Prints the execution time.
        Usage: 
            Benchmark.mark('message to print')
            Code to benchmark...
            Benchmark.mark('message to print')

        Keyword arguments:
        message -- a key for the dict and message to print
        """
        if message not in self.start:
            self.start[message] = -1
            self.end[message] = -1
        if self.start[message] == -1:
            self.start[message] = timer()
        else:
            if self.end[message] == -1:
                self.end[message] = timer()
            print('{message:{fill}{align}{width}}-{time}'.format(message=message,
                fill='-', align='<', width=50,
                time=(self.end[message] - self.start[message])))
            self.start[message] = -1
            self.end[message] = -1


S3_DIR = 'https://YOURBUCKET.s3.us-east-2.amazonaws.com'
S3_BUCKET = 'YOURBUCKET'
ROOT = os.path.abspath('.')
DATA = os.path.abspath('../data')
OUT_DIR = 'new_adv_imgs'
API_DIR = 'new_api_results'
DEVICE = 'cuda'
BM = Benchmark()

CASIA_MODEL_PATH = 'weights/facenet_casia.h5'
VGGSMALL_MODEL_PATH = 'weights/small_facenet_center.h5'
VGGADV_MODEL_PATH = 'weights/facenet_vggsmall.h5'
CENTER_MODEL_PATH = 'weights/facenet_keras_center_weights.h5'
TRIPLET_MODEL_PATH = 'weights/facenet_keras_weights.h5'


np.arange(0.0, 2, 0.04)
np.arange(0.0, 2, 0.008)
MODELS = {'Facenet': (160, 160), 'VGG-Face': (224, 224), 'OpenFace': (96, 96), 'DeepFace': (152, 152), 'DeepID': (47, 66), 'Dlib': (150, 150), 'ArcFace': (112, 112), 'pytorch': (160, 160), 'resnet50': (224, 224), 'vggimbalanced': (224, 224), 'vggbalanced': (224, 224), 'vggraceimbalanced': (224, 224), 'vggracebalanced': (224, 224),'vggface2_default_facenet': (160, 160),'vggface2_race_balanced_facenet': (160, 160),'vggface2_sex_balanced_facenet': (160, 160),'vggface2_standard_facenet': (160,160),'vggface2_casia_facenet':(160,160), 'torch_Xu_facenet':(160,160)}
THRESHOLDS = {'Facenet': np.arange(0.0, 20, 0.4), 'VGG-Face': np.arange(0.0, 1.5, 0.03), 'OpenFace': np.arange(0.0, 2.0, 0.04), 'DeepFace': np.arange(40, 80, 0.8), 'DeepID': np.arange(20, 60, 0.8), 'Dlib': np.arange(0.0, 20, 0.4), 'ArcFace': np.arange(10.0, 40, 0.6), 'pytorch': np.arange(6.0, 20, 0.28),  'resnet50': np.arange(0.0, 2.0, 0.04), 'vggimbalanced': np.arange(0.0, 2.0, 0.04), 'vggbalanced': np.arange(0.0, 2.0, 0.04), 'vggraceimbalanced': np.arange(0.0, 2.0, 0.04), 'vggracebalanced': np.arange(0.0, 2.0, 0.04),'vggface2_default_facenet': np.arange(0.0, 2, 0.04),'vggface2_race_balanced_facenet': np.arange(0.0, 2, 0.04),'vggface2_sex_balanced_facenet': np.arange(0.0, 2, 0.04),'vggface2_standard_facenet': np.arange(0.0, 20, 0.4),'vggface2_casia_facenet': np.arange(0.0, 20, 0.4),'torch_Xu_facenet':np.arange(0.0, 2, 0.04) }
SMALL_THRESHOLDS = {'Facenet': np.arange(0.0, 20, 0.08), 'VGG-Face': np.arange(0.0, 1.5, 0.006), 'OpenFace': np.arange(0.0, 2.0, 0.008), 'DeepFace': np.arange(40, 80, 0.16), 'DeepID': np.arange(20, 60, 0.16), 'Dlib': np.arange(0.0, 20, 0.08), 'ArcFace': np.arange(10.0, 40, 0.12), 'pytorch': np.arange(6.0, 20, 0.056),  'resnet50': np.arange(0.0, 2.0, 0.008), 'vggimbalanced': np.arange(0.0, 2.0, 0.008), 'vggbalanced': np.arange(0.0, 2.0, 0.008), 'vggraceimbalanced': np.arange(0.0, 2.0, 0.008), 'vggracebalanced': np.arange(0.0, 2.0, 0.008),'vggface2_default_facenet': np.arange(0.0, 2, 0.008),'vggface2_race_balanced_facenet': np.arange(0.0, 2, 0.008),'vggface2_sex_balanced_facenet': np.arange(0.0, 2, 0.008),'vggface2_standard_facenet': np.arange(0.0, 20, 0.08), 'torch_Xu_facenet':np.arange(0.0, 2, 0.008)}

def bool_cmd_arg(x):
    
    if x.strip().upper() == 'FALSE':
        return False
    if x.strip().upper() == 'TRUE':
        return True
    
def attack_cmd_arg(x):
    return x.lower()
    

def parse_and_configure_arguments():
    args = parse_arguments()
    
    set_gpu(args.gpu)
    params = set_parameters(args)
    return params

def construct_parser():
    parser = argparse.ArgumentParser()
    # Generic args
    parser.add_argument('--gpu', type=str, default="0", help='GPU(s) to run the code on')

    parser.add_argument('--model', type=str, default="Facenet", help='Type of model to generate adversarial examples with', choices=['Facenet','VGG-Face', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace', 'Emotion', 'Age', 'Gender', 'Race', 'pytorch', 'resnet50', 'vggimbalanced', 'vggbalanced', 'vggraceimbalanced', 'vggracebalanced','vggface2_default_facenet','vggface2_race_balanced_facenet','vggface2_sex_balanced_facenet','vggface2_standard_facenet','vggface2_casia_facenet','torch_Xu_facenet'])
    parser.add_argument('--target-model', type=str, default="Facenet", help='Type of model to evaluate adversarial examples on', choices=['Facenet','VGG-Face', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace', 'Emotion', 'Age', 'Gender', 'Race','vggface2_default_facenet','vggface2_race_balanced_facenet','vggface2_sex_balanced_facenet','vggface2_standard_facenet','vggface2_casia_facenet','torch_Xu_facenet'])

    # Attack args
    parser.add_argument('--attack', type=attack_cmd_arg, default='cw', help='Attack algorithm',choices=['pgd', 'cw', 'foggysight', 'lowkey'])
    parser.add_argument('--norm', type=str, default='2', help='Lp-norm of attack', choices=['inf', '2'])
    parser.add_argument('--targeted-flag', type=bool_cmd_arg, default='TRUE', help='Targeted (true) or untargeted (false) attack')
    parser.add_argument('--tv-flag', type=bool_cmd_arg, default='FALSE', help='Use tv_loss term (true) or do not use it (false)')
    parser.add_argument('--hinge-flag', type=bool_cmd_arg, default='TRUE', help='Use hinge loss (true) or target loss (false)')
    parser.add_argument('--cos-flag', type=bool_cmd_arg, default='FALSE', help='Use cosine similarity (true) instead of l2 (false) for loss')
    parser.add_argument('--margin', type=float, default=6.0, help='Separation parameter for adversarial example transferability')
    parser.add_argument('--amplification', type=float, default=5.1, help='Parameter for amplifying adversarial examples (minimum 1.0)')
    parser.add_argument('--granularity', type=str, default='normal', help='Parameter specifies margin and amplification intervals', choices=['fine', 'normal', 'coarse', 'coarser', 'coarsest', 'single', 'fine-tuned', 'no-amp', 'coarse-single', 'api-eval'])
    parser.add_argument('--mean-loss', type=str, default='embeddingmean', help='(embedding): computes distances to each image. (embeddingmean): computes distances to each centroid.', choices=['embeddingmean', 'embedding'])
    parser.add_argument('--interpolation', type=str, default='bilinear', help='Interpolation method for upscaling and downscaling', choices=['nearest', 'bilinear', 'bicubic', 'lanczos'])
    parser.add_argument('--batch-size', type=int, default=-1, help='Batch size for attack and evaluation')
    parser.add_argument('--folder', type=str, default='celeb', help='Dataset or folder to load')
    parser.add_argument('--adversarial-flag', type=bool_cmd_arg, default='FALSE', help='Evaluate adversarial or natural dataset')
    parser.add_argument('--attribute', type=str, default=None, help='Demographic attribute to evaluate')
    parser.add_argument('--whitebox-target', type=bool_cmd_arg, default="FALSE", help='Use the target model (true) or source model (false) for evaluation')
    parser.add_argument('--different-flag', type=bool_cmd_arg, default='FALSE', help='Source target pairs are a different demographic (true) or same demographic (false)')
    parser.add_argument('--correct-flag', type=bool_cmd_arg, default='FALSE', help='')
    parser.add_argument('--source', type=str, default=None, help='Batches for sources with attack generation', choices=[None, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
    parser.add_argument('--target', type=str, default=None, help='Batches for targets with attack generation', choices=[None, '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
    parser.add_argument('--targeted-success', type=bool_cmd_arg, default='FALSE', help='Evaluate using targeted success metric (true), or untargeted success metric (false)')
    parser.add_argument('--topn-flag', type=bool_cmd_arg, default='FALSE', help='Perform evaluation with top-n metric (true) or not (false)')
    # Attack algorithm args
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value needed for PGD')
    parser.add_argument('--iterations', type=int, default=20, help='Number of inner step iterations for CW and number of iterations for PGD')
    parser.add_argument('--binary-steps', type=int, default=5, help='Number of binary search steps for CW')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate for CW')
    parser.add_argument('--epsilon-steps', type=float, default=0.01, help='Epsilon per iteration for PGD')
    parser.add_argument('--init-const', type=float, default=0.3, help='Initial const value for CW')
    # Face recognition API args
    parser.add_argument('--api-name', type=str, default='azure', help='API to evaluate images with', choices=['azure', 'awsverify', 'facepp'])
    # TCAV args
    parser.add_argument('--relative-random', type=str, default='relative', help='Use relative concepts or random concepts for statistical significance')
    parser.add_argument('--bottlenecks', type=str, default='many', help='Parameter for the layers to evaluate', choices=['many', 'some', 'good', 'small', 'early', 'all', 'early_single', 'best_early'])
    parser.add_argument('--alpha', type=float, default=0.1, help='')
    parser.add_argument('--target-class', type=str, default='n000875', help='Name of the class/label to evaluate')
    parser.add_argument('--feature', type=str, default='mouth', help='Facial feature to evaluate', choices=['mouth', 'forehead', 'eyes', 'eye', 'cheek', 'all', 'skin'])
    parser.add_argument('--concept', type=str, default='race', help='Concept type to evaluate', choices=['race', 'gender', 'skin'])
    parser.add_argument('--mode', type=str, default='cav', help='Gradient computation method for embedding networks', choices=['cav', 'gradient', 'jacobian', 'neither'])
    parser.add_argument('--pair-flag',type=bool_cmd_arg, default='FALSE')
    return parser

def parse_arguments():
    parser = construct_parser()
    args = parser.parse_args()
    return args


def string_to_bool(arg):
    """Converts a string into a returned boolean."""
    if arg.lower() == 'true':
        arg = True
    elif arg.lower() == 'false':
        arg = False
    else:
        raise ValueError('ValueError: Argument must be either "true" or "false".')
    return arg


def set_gpu(gpu):
    """Configures CUDA environment variable and returns tensorflow GPU config."""
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')

    print('gpu',gpu,gpu_devices)
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    DEVICE = 'cuda:{}'.format(gpu)

def set_parameters(kwargs):
    params = vars(kwargs)
    """
    Initializes params dictionary to be used in most functions.

    Keyword arguments:
    api_name -- API to evaluate against (azure, awsverify, facepp)
    targeted_flag -- true: use targeted attack
    tv_flag -- true: use tv loss
    hinge_flag -- true: use hinge loss (defined in paper)
    cos_flag -- true: use cosine similarity along with l2 norm
    interpolation -- type of interpolation to use in resizing delta
    model_type -- input size used in training model (small, large)
    loss_type -- loss type used in training model (center, triplet)
    dataset_type -- dataset used in training model (vgg, casia, vggsmall, vggadv)
    target_model -- target model size (whitebox transferability eval)
    target_loss -- target loss type (whitebox transferability eval)
    target_dataset -- target dataset type (whitebox transferability eval)
    attack -- attack type to use (CW, PGD)
    norm -- attack loss norm (2, inf)
    epsilon -- PGD epsilon upper bound
    iterations -- number of epochs for CW and PGD
    binary_steps -- number of outer binary search steps in CW
    learning_rate -- learning rate to use in attack
    epsilon_steps -- epsilon update value
    init_const -- initial CW constant
    mean_loss -- whether to use mean of embeddings or non-mean loss (embeddingmean, embedding)
    batch_size -- batch size used in attack (embedding: must be 1)
    margin -- margin or kappa value used in attack
    amplification -- amplification or alpha value used in amplifying perturbation
    granularity -- granularity of intervals for margin and amplification values
        (fine, normal, coarse, coarser, coarsest, single, fine-tuned, coarse-single, api-eval)
    whitebox_target -- true: using target model for whitebox transferability evaluation
    pair_flag -- true: use Config.PAIRS to determine source-target pairs
    """
    
    params['folder_dir'] = os.path.join(DATA, params['folder'])
    params['adversarial_dir'] = os.path.join(DATA, 'adversarial')

    if params['folder'] == 'lfw':
        params['targets'] = LFW_TARGETS
        params['sources'] = LFW_SOURCES
        params['flats'] = LFW_FLAT
        params['attributes'] = LFW_ATTRIBUTES
        if params['attribute'] == 'race':
            if params['correct_flag']:
                params['targets'] = LFWBESTRACE_TARGETS
                params['sources'] = LFWBESTRACE_SOURCES
                params['flats'] = LFWBESTRACE_FLAT
                params['attributes'] = LFWRACE_ATTRIBUTES
                params['other'] = LFWBESTRACE_OTHER
            else:
                params['targets'] = LFWRACE_TARGETS
                params['sources'] = LFWRACE_SOURCES
                params['flats'] = LFWRACE_FLAT
                params['attributes'] = LFWRACE_ATTRIBUTES
                params['other'] = LFWRACE_OTHER
        elif params['attribute'] == 'sex':
            if params['correct_flag']:
                params['targets'] = LFWBESTSEX_TARGETS
                params['sources'] = LFWBESTSEX_SOURCES
                params['flats'] = LFWBESTSEX_FLAT
                params['attributes'] = LFWSEX_ATTRIBUTES
                params['other'] = LFWBESTSEX_OTHER
            else:
                params['targets'] = LFWSEX_TARGETS
                params['sources'] = LFWSEX_SOURCES
                params['flats'] = LFWSEX_FLAT
                params['attributes'] = LFWSEX_ATTRIBUTES
                params['other'] = LFWSEX_OTHER
    elif params['folder'] == 'celeb':
        params['targets'] = CELEB_TARGETS
        params['sources'] = CELEB_SOURCES
        params['flats'] = CELEB_FLAT
    elif params['folder'] == 'vgg':
        params['targets'] = VGG_TARGETS
        params['sources'] = VGG_SOURCES
        params['flats'] = VGG_FLAT
    elif params['folder'] == 'vggface2':
        if params['attribute'] == 'race':
            params['targets'] = VGGRACE_TARGETS
            params['sources'] = VGGRACE_SOURCES
            params['flats'] = VGGRACE_FLAT
            params['attributes'] = VGGRACE_ATTRIBUTES
            params['other'] = VGGRACE_OTHER
        elif params['attribute'] == 'sex':
            params['targets'] = VGGSEX_TARGETS
            params['sources'] = VGGSEX_SOURCES
            params['flats'] = VGGSEX_FLAT
            params['attributes'] = VGGSEX_ATTRIBUTES
            params['other'] = VGGSEX_OTHER
    elif params['folder'] == 'tcav':
        if params['attribute'] == 'race':
            params['targets'] = TCAVRACE_TARGETS
            params['sources'] = TCAVRACE_SOURCES
            params['flats'] = TCAVRACE_FLAT
            params['attributes'] = TCAVRACE_ATTRIBUTES
            params['other'] = TCAVRACE_OTHER
        elif params['attribute'] == 'sex':
            params['targets'] = TCAVSEX_TARGETS
            params['sources'] = TCAVSEX_SOURCES
            params['flats'] = TCAVSEX_FLAT
            params['attributes'] = TCAVSEX_ATTRIBUTES
            params['other'] = TCAVSEX_OTHER

    params['pixel_max'] = 1.0
    params['pixel_min'] = 0.0
    if params['model'] == 'Race' or params['model'] == 'Age' or params['model'] == 'Emotion' or params['model'] == 'Gender':
        params['image_dims'] = (224, 224)
    else:
        if params['whitebox_target']:
            params['image_dims'] = MODELS[params['target_model']]
            params['thresholds'] = THRESHOLDS[params['target_model']]
            params['small_thresholds'] = SMALL_THRESHOLDS[params['target_model']]
        else:
            params['image_dims'] = MODELS[params['model']]
            params['thresholds'] = THRESHOLDS[params['model']]
            params['small_thresholds'] = SMALL_THRESHOLDS[params['model']]
    
    params['align_dir'] = '{}/{}/{}-align-160'.format(DATA, params['folder'], params['folder'])
    # params['align_dir'] = '{}/{}/{}'.format(DATA, params['folder'], params['folder'])
    
    if params['interpolation'] == 'nearest':
        params['interpolation'] = cv2.INTER_NEAREST
    elif params['interpolation'] == 'bilinear':
        params['interpolation'] = cv2.INTER_LINEAR
    elif params['interpolation'] == 'bicubic':
        params['interpolation'] = cv2.INTER_CUBIC
    elif params['interpolation'] == 'lanczos':
        params['interpolation'] = cv2.INTER_LANCZOS4
    elif params['interpolation'] == 'super':
        ValueError('ValueError: Super interpolation not yet implemented.')
    else:
        raise ValueError('ValueError: Argument must be of the following, [nearest, bilinear, bicubic, lanczos, super].')

    if params['granularity'] == 'fine':
        params['margin_list'] = np.arange(0.0, params['margin'], params['margin'] / 20.0)
        params['amp_list'] = np.arange(1.0, params['amplification'], 0.2)
    elif params['granularity'] == 'normal':
        params['margin_list'] = np.arange(0.0, params['margin'], params['margin'] / 10.0)
        params['amp_list'] = np.arange(1.0, params['amplification'], 0.5)
    elif params['granularity'] == 'coarse':
        params['margin_list'] = np.arange(0.0, params['margin'], params['margin'] / 5.0)
        params['amp_list'] = np.arange(1.0, params['amplification'], 1.0)
    elif params['granularity'] == 'coarser':
        params['margin_list'] = np.arange(0.0, params['margin'], params['margin'] / 3.0)
        params['amp_list'] = np.arange(1.0, params['amplification'], 0.2)
    elif params['granularity'] == 'coarsest':
        params['margin_list'] = np.arange(0.0, params['margin'], params['margin'] / 3.0)
        params['amp_list'] = np.arange(1.0, params['amplification'], 1.0)
    elif params['granularity'] == 'single':
        params['margin_list'] = np.array([params['margin']])
        params['amp_list'] = np.array([params['amplification']])
    elif params['granularity'] == 'fine-tuned':
        params['margin_list'] = np.arange(10.0, params['margin'], 1.0)
        params['amp_list'] = np.arange(1.0, params['amplification'], 0.2)
    elif params['granularity'] == 'coarse-single':
        params['margin_list'] = np.arange(0.0, params['margin'], params['margin'] / 3.0)
        params['amp_list'] = np.array([1.0])
    elif params['granularity'] == 'api-eval':
        params['margin_list'] = np.arange(0.0, params['margin'], params['margin'] / 3.0)
        params['amp_list'] = np.arange(1.0, params['amplification'], 0.8)
    elif params['granularity'] == 'no-amp':
        params['margin_list'] = np.arange(0.0, params['margin'], params['margin'] / 3.0)
        params['amp_list'] = np.array([params['amplification']])
    else:
        raise ValueError('ValueError: Argument must be of the following, [fine, normal, coarse, coarser, single].')

    if params['hinge_flag']:
        params['attack_loss'] = 'hinge'
    else:
        params['attack_loss'] = 'target'
    if not params['targeted_flag']:
        params['attack_loss'] = 'target'
    if params['norm'] == 'inf':
        norm_name = 'i'
    else:
        norm_name = '2'
    if params['tv_flag']:
        tv_name = '_tv'
    else:
        tv_name = ''
    if params['cos_flag']:
        cos_name = '_cos'
    else:
        cos_name = ''

    if params['attack'] == 'lowkey' or params['attack'] == 'foggysight':
        params['attack_name'] = params['attack'].lower()
    else:
        params['attack_name'] = '{}_l{}{}{}'.format(params['attack'].lower(), norm_name, tv_name, cos_name)
    params['directory_path'] = os.path.join(DATA,
                                            OUT_DIR,
                                            params['attack_name'],
                                            params['model'],
                                            '{}_loss/full'.format(params['attack_loss']))
    params['directory_path_crop'] = os.path.join(DATA,
                                                 OUT_DIR,
                                                 params['attack_name'],
                                                 params['model'],
                                                 '{}_loss/crop'.format(params['attack_loss']))
    params['directory_path_npz'] = os.path.join(DATA,
                                                OUT_DIR,
                                                params['attack_name'],
                                                params['model'],
                                                '{}_loss/npz'.format(params['attack_loss']))
    params['api_path'] = os.path.join(DATA,
                                      API_DIR,
                                      params['attack_name'],
                                      params['model'],
                                      '{}_loss/npz'.format(params['attack_loss']))
    if params['adversarial_flag']:
        params['all_name'] = '{}_{}_{}'.format(params['folder'], params['attack_name'], params['model'])
    else:
        params['all_name'] = '{}_{}'.format(params['folder'], params['model'])

    if params['attribute'] == None:
        params['adv_dir_folder'] = '{}_{}'.format(params['all_name'], params['amplification'])
    else:
        if params['attack'] == 'lowkey' or params['attack'] == 'foggysight':
            if params['correct_flag']:
                params['adv_dir_folder'] = '{}_{}_correct_{}'.format(params['all_name'], params['attribute'], params['amplification'])
            else:
                params['adv_dir_folder'] = '{}_{}_{}'.format(params['all_name'], params['attribute'], params['amplification'])
        else:
            if params['correct_flag']:
                if params['different_flag']:
                    if params['target'] != None and params['source'] != None:
                        params['adv_dir_folder'] = '{}_{}_correct_diff_{}_{}_{}'.format(params['all_name'], params['attribute'], LFWRACE_ATTRIBUTES[int(params['source'])], LFWRACE_ATTRIBUTES[int(params['target'])], params['amplification'])
                    else:
                        params['adv_dir_folder'] = '{}_{}_correct_diff_{}'.format(params['all_name'], params['attribute'], params['amplification'])
                else:
                    params['adv_dir_folder'] = '{}_{}_correct_same_{}'.format(params['all_name'], params['attribute'], params['amplification'])
            else:
                if params['different_flag']:
                    if params['target'] != None and params['source'] != None:
                        params['adv_dir_folder'] = '{}_{}_diff_{}_{}_{}'.format(params['all_name'], params['attribute'], LFWRACE_ATTRIBUTES[int(params['source'])], LFWRACE_ATTRIBUTES[int(params['target'])], params['amplification'])
                    else:
                        params['adv_dir_folder'] = '{}_{}_diff_{}'.format(params['all_name'], params['attribute'], params['amplification'])
                else:
                    params['adv_dir_folder'] = '{}_{}_same_{}'.format(params['all_name'], params['attribute'], params['amplification'])
    params['directory_path_crop'] = os.path.join(params['adversarial_dir'], params['adv_dir_folder'])

    return params
