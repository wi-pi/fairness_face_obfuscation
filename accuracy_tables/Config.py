import os
import argparse
import numpy as np


ROOT = os.path.abspath('./accuracy_tables/')
OUT_DIR = 'data/new_adv_imgs'
API_DIR = 'data/new_api_results'

ATTR = ['imagenum', 'Male', 'Asian', 'White', 'Black', 'Baby',
        'Child', 'Youth', 'Middle Aged', 'Senior', 'Black Hair', 'Blond Hair',
        'Brown Hair', 'Bald', 'No Eyewear', 'Eyeglasses', 'Sunglasses', 'Mustache',
        'Smiling', 'Frowning', 'Chubby', 'Blurry', 'Harsh Lighting', 'Flash', 'Soft Lighting',
        'Outdoor', 'Curly Hair', 'Wavy Hair', 'Straight Hair', 'Receding Hairline', 'Bangs',
        'Sideburns', 'Fully Visible Forehead', 'Partially Visible Forehead', 'Obstructed Forehead',
        'Bushy Eyebrows', 'Arched Eyebrows', 'Narrow Eyes', 'Eyes Open', 'Big Nose', 'Pointy Nose',
        'Big Lips', 'Mouth Closed', 'Mouth Slightly Open', 'Mouth Wide Open', 'Teeth Not Visible',
        'No Beard', 'Goatee', 'Round Jaw', 'Double Chin', 'Wearing Hat', 'Oval Face', 'Square Face',
        'Round Face', 'Color Photo', 'Posed Photo', 'Attractive Man', 'Attractive Woman', 'Indian',
        'Gray Hair', 'Bags Under Eyes', 'Heavy Makeup', 'Rosy Cheeks', 'Shiny Skin', 'Pale Skin',
        '5 o Clock Shadow', 'Strong Nose-Mouth Lines', 'Wearing Lipstick', 'Flushed Face',
        'High Cheekbones', 'Brown Eyes', 'Wearing Earrings', 'Wearing Necktie', 'Wearing Necklace']


def string_to_bool(arg):
    """Converts a string into a returned boolean."""
    if arg.lower().strip() == 'true':
        arg = True
    elif arg.lower().strip() == 'false':
        arg = False
    else:
        print(arg)
        raise ValueError('ValueError: Argument must be either "true" or "false".')
    return arg


def parse_arguments(flag):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0", help='GPU(s) to run the code on')
    parser.add_argument('--model', type=str, default="small", help='type of model')
    parser.add_argument('--model-type', type=str, default="small", help='type of model', choices=['small','large'])
    parser.add_argument('--loss-type', type=str, default="center", help='loss function used to train the model',choices=['center','triplet'])
    parser.add_argument('--dataset-type', type=str, default='vgg', help='dataset used in training model', choices=['vgg', 'vggsmall', 'casia', 'lfw'])
    parser.add_argument('--attack', type=str, default='CW', help='attack type',choices=['PGD', 'CW'])
    parser.add_argument('--norm', type=str, default='2', help='p-norm', choices=['inf', '2'])
    parser.add_argument('--targeted-flag', type=str, default='true', help='targeted (true) or untargeted (false)', choices=['true', 'false'])
    parser.add_argument('--tv-flag', type=str, default='false', help='do not use tv_loss term (false) or use it (true)', choices=['true', 'false'])
    parser.add_argument('--hinge-flag', type=str, default='true', help='hinge loss (true) or target loss (false)', choices=['true', 'false'])
    parser.add_argument('--cos-flag', type=str, default='false', help='use cosine similarity instead of l2 for loss', choices=['true', 'false'])
    parser.add_argument('--margin', type=float, default=6.0, help='needed for determining goodness of transferability')
    parser.add_argument('--amplification', type=float, default=5.1, help='needed for amplifying adversarial examples')
    parser.add_argument('--granularity', type=str, default='normal', help='add more or less margin and amplification intervals', choices=['fine', 'normal', 'coarse', 'coarser', 'coarsest', 'single', 'fine-tuned', 'no-amp', 'coarse-single', 'api-eval'])
    parser.add_argument('--mean-loss', type=str, default='embeddingmean', help='old:(embedding) new formulation:(embeddingmean) WIP formulation:(distancemean)', choices=['embeddingmean', 'embedding', 'distancemean'])
    parser.add_argument('--batch-size', type=int, default=-1, help='batch size for evaluation')
    parser.add_argument('--pair-flag', type=str, default='false', help='optimal source target pairs')
    parser.add_argument('--folder', type=str, default='celeb', help='Dataset or folder to load')
    if flag == 'whitebox':
        parser.add_argument('--topn', type=int, default=5, help='do top-n evaluation of closest faces')
        parser.add_argument('--target-model-type', type=str, default="small", help='type of model', choices=['small','large'])
        parser.add_argument('--target-loss-type', type=str, default="center", help='loss function used to train the model',choices=['center','triplet'])
        parser.add_argument('--target-dataset-type', type=str, default='vgg', help='dataset used in training model', choices=['vgg', 'vggsmall', 'casia'])
    elif flag == 'attack':
        parser.add_argument('--epsilon', type=float, default=0.1, help='epsilon value needed for PGD')
        parser.add_argument('--iterations', type=int, default=20, help='number of inner step iterations for CW and number of iterations for PGD')
        parser.add_argument('--binary-steps', type=int, default=5, help='number of binary search steps for CW')
        parser.add_argument('--learning-rate', type=float, default=0.01, help='learning rate for CW')
        parser.add_argument('--epsilon-steps', type=float, default=0.01, help='epsilon per iteration for PGD')
        parser.add_argument('--init-const', type=float, default=0.3, help='initial const value for CW')
        parser.add_argument('--source', type=str, default='none', help='', choices=['barack', 'bill', 'jenn', 'leo', 'mark', 'matt', 'melania', 'meryl', 'morgan', 'taylor', 'myface', 'none', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'])
    elif flag == 'api':
        parser.add_argument('--api-name', type=str, default='azure', help='API to evaluate against', choices=['azure', 'awsverify', 'facepp'])
        parser.add_argument('--topn-flag', type=str, default='true', help='topn evaluation or not', choices=['true', 'false'])
        parser.add_argument('--credentials', type=str, default='0', help='api keys')
    elif flag == 'embedding':
        parser.add_argument('--adversarial-flag', type=str, default='false')
        parser.add_argument('--attribute', type=str, default='')
        parser.add_argument('--all-flag', type=str, default='false')
        parser.add_argument('--uniform-flag', type=str, default='false')
    args = parser.parse_args()
    return args


def set_parameters(api_name='',
                   targeted_flag='true',
                   tv_flag='false',
                   hinge_flag='true',
                   cos_flag='false',
                   model_type='large',
                   loss_type='triplet',
                   dataset_type='vgg',
                   target_model='large',
                   target_loss='center',
                   target_dataset='VGG',
                   attack='CW',
                   norm='2',
                   epsilon=0.1,
                   iterations=20,
                   binary_steps=5,
                   learning_rate=0.01,
                   epsilon_steps=0.01,
                   init_const=0.3,
                   mean_loss='embeddingmean',
                   batch_size=-1,
                   margin=15.0,
                   amplification=6.0,
                   granularity='normal',
                   whitebox_target=False,
                   pair_flag='false',
                   adversarial_flag='false',
                   all_flag='false',
                   uniform_flag='false',
                   attribute='',
                   folder=''):
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
    
    params = {}
    params['model_type'] = model_type
    params['loss_type'] = loss_type
    params['dataset_type'] = dataset_type
    params['target_model'] = target_model
    params['target_loss'] = target_loss
    params['target_dataset'] = target_dataset
    params['attack'] = attack
    params['norm'] = norm
    params['epsilon'] = epsilon
    params['iterations'] = iterations
    params['binary_steps'] = binary_steps
    params['learning_rate'] = learning_rate
    params['epsilon_steps'] = epsilon_steps
    params['init_const'] = init_const
    params['mean_loss'] = mean_loss
    params['batch_size'] = batch_size
    params['whitebox_target'] = whitebox_target
    params['targeted_flag'] = string_to_bool(targeted_flag)
    params['tv_flag'] = string_to_bool(tv_flag)
    params['hinge_flag'] = string_to_bool(hinge_flag)
    params['cos_flag'] = string_to_bool(cos_flag)
    params['pair_flag'] = string_to_bool(pair_flag)
    params['adversarial_flag'] = string_to_bool(adversarial_flag)
    params['all_flag'] = string_to_bool(all_flag)
    params['uniform_flag'] = string_to_bool(uniform_flag)
    params['attribute'] = attribute
    params['api_name'] = api_name
    params['folder'] = folder
    params['folder_dir'] = os.path.join(ROOT, folder)

    if model_type == 'small' and loss_type == 'center':
        params['pixel_max'] = 1.0
        params['pixel_min'] = -1.0
    else:
        params['pixel_max'] = 1.0
        params['pixel_min'] = 0.0

    if model_type == 'large' or dataset_type == 'casia':
        params['align_dir'] = '{}/{}/{}-align-160'.format(ROOT, params['folder'], params['folder'])
    elif model_type == 'small':
        params['align_dir'] = '{}/{}/{}-align-96'.format(ROOT, params['folder'], params['folder'])
    else:
        ValueError('ValueError: Argument must be either "small" or "large".')
    
    if granularity == 'fine':
        params['margin_list'] = np.arange(0.0, margin, margin / 20.0)
        params['amp_list'] = np.arange(1.0, amplification, 0.2)
    elif granularity == 'normal':
        params['margin_list'] = np.arange(0.0, margin, margin / 10.0)
        params['amp_list'] = np.arange(1.0, amplification, 0.5)
    elif granularity == 'coarse':
        params['margin_list'] = np.arange(0.0, margin, margin / 5.0)
        params['amp_list'] = np.arange(1.0, amplification, 1.0)
    elif granularity == 'coarser':
        params['margin_list'] = np.arange(0.0, margin, margin / 3.0)
        params['amp_list'] = np.arange(1.0, amplification, 0.2)
    elif granularity == 'coarsest':
        params['margin_list'] = np.arange(0.0, margin, margin / 3.0)
        params['amp_list'] = np.arange(1.0, amplification, 1.0)
    elif granularity == 'single':
        params['margin_list'] = np.array([margin])
        params['amp_list'] = np.array([amplification])
    elif granularity == 'fine-tuned':
        params['margin_list'] = np.arange(10.0, margin, 1.0)
        params['amp_list'] = np.arange(1.0, amplification, 0.2)
    elif granularity == 'coarse-single':
        params['margin_list'] = np.arange(0.0, margin, margin / 3.0)
        params['amp_list'] = np.array([1.0])
    elif granularity == 'api-eval':
        params['margin_list'] = np.arange(0.0, margin, margin / 3.0)
        params['amp_list'] = np.arange(1.0, amplification, 0.8)
    elif granularity == 'no-amp':
        params['margin_list'] = np.arange(0.0, margin, margin / 3.0)
        params['amp_list'] = np.array([amplification])
    else:
        raise ValueError('ValueError: Argument must be of the following, [fine, normal, coarse, coarser, single].')

    if params['hinge_flag']:
        params['attack_loss'] = 'hinge'
    else:
        params['attack_loss'] = 'target'
    if not params['targeted_flag']:
        params['attack_loss'] = 'target'
    if norm == 'inf':
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

    params['model_name'] = '{}_{}'.format(model_type, loss_type)
    if dataset_type == 'casia' or dataset_type == 'vggsmall':
        params['model_name'] = dataset_type
    params['target_model_name'] = '{}_{}_{}'.format(target_model, target_loss, target_dataset)
    params['attack_name'] = '{}_l{}{}{}'.format(attack.lower(), norm_name, tv_name, cos_name)
    params['directory_path'] = os.path.join(ROOT,
                                            OUT_DIR,
                                            params['attack_name'],
                                            params['model_name'],
                                            '{}_loss/full'.format(params['attack_loss']))
    params['directory_path_crop'] = os.path.join(ROOT,
                                                 OUT_DIR,
                                                 params['attack_name'],
                                                 params['model_name'],
                                                 '{}_loss/crop'.format(params['attack_loss']))
    params['directory_path_npz'] = os.path.join(ROOT,
                                                OUT_DIR,
                                                params['attack_name'],
                                                params['model_name'],
                                                '{}_loss/npz'.format(params['attack_loss']))
    params['api_path'] = os.path.join(ROOT,
                                      API_DIR,
                                      params['attack_name'],
                                      params['model_name'],
                                      '{}_loss/npz'.format(params['attack_loss']))
    if params['mean_loss'] == 'embedding':
        params['directory_path'] += '_mean'
        params['directory_path_crop'] += '_mean'
        params['directory_path_npz'] += '_mean'
        params['api_path'] += '_mean'
    if params['adversarial_flag']:
        params['all_name'] = '{}_{}_{}'.format(params['folder'], params['attack_name'], params['model_name'])
    else:
        params['all_name'] = '{}_{}'.format(params['folder'], params['model_name'])

    return params
