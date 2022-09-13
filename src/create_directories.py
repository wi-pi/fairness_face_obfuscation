from os.path import abspath, join, exists
from os import mkdir
import Config


def create(path):
    """
    Description

    Keyword arguments:
    """
    if not exists(path):
        mkdir(path)
        print('Creating directory: {}'.format(path))


def check_depth(depth):
    """
    Description

    Keyword arguments:
    """
    if depth == -1:
        directory_list = API_NAMES
    if depth == 0:
        directory_list = ATTACKS
    if depth == 1:
        directory_list = MODEL_SIZE
    if depth == 2:
        directory_list = ATTACK_LOSS
    if depth == 3:
        directory_list = CROP
    return directory_list


def doit(depth, limit, cur, prev_path):
    """
    Description

    Keyword arguments:
    """
    new_path = join(prev_path, cur)
    create(new_path)
    if depth < limit:
        recurse_directories(depth + 1, limit, new_path)


def recurse_directories(depth, limit, prev_path):
    """
    Description

    Keyword arguments:
    """
    directory_list = check_depth(depth)
    for i in directory_list:
        doit(depth, limit, i, prev_path)


ROOT = Config.DATA
create(ROOT)

ADV_IMGS = join(ROOT, 'new_adv_imgs')
API_RESULTS = join(ROOT, 'new_api_results')
ADVERSARIAL = join(ROOT, 'adversarial')
EMBEDDINGS = join(ROOT, 'embeddings')
FILEIO = join(ROOT, 'fileio')
MYFACE = join(ROOT, 'myface')
LFW = join(ROOT, 'lfw')
AGE_GAN = join(ROOT, 'age-gan')
AGE_GAN_STYLE = join(ROOT, 'age-gan-style')
CELEB = join(ROOT, 'celeb')
VGGFACE2 = join(ROOT, 'vggface2')
CELEBA = join(ROOT, 'celeba')
UTKFACE = join(ROOT, 'utkface')

API_NAMES =   ['azure', 'awsverify', 'facepp']
ATTACKS =     ['cw_l2', 'cw_li', 'pgd_l2', 'pgd_li', 'lowkey', 'foggysight']
MODEL_SIZE =  ['Facenet', 'ArcFace', 'DeepID', 'Dlib', 'DeepFace', 'OpenFace', 'VGGFace']
ATTACK_LOSS = ['hinge_loss']
CROP =        ['crop', 'full', 'npz']

create(ADV_IMGS)
recurse_directories(0, 3, ADV_IMGS)

create(API_RESULTS)
recurse_directories(-1, 3, API_RESULTS)

create(ADVERSARIAL)
create(EMBEDDINGS)
create(FILEIO)
create(MYFACE)
create(LFW)
create(AGE_GAN)
create(AGE_GAN_STYLE)
create(CELEB)
create(VGGFACE2)
create(CELEBA)
create(UTKFACE)
 
print('SUCCESS!')