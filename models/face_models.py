import Config
import os
from utils.fr_utils import *
from models.inception_big import *
from tensorflow.keras.models import load_model, model_from_json
from models.inception_resnet_v1 import *
from models.torch_inception_resnet_v1 import *
from deepface import DeepFace
import torch

def get_model(params):
    """
    """
    if params['whitebox_target']:
        model_name = params['target_model']
    else:
        model_name = params['model']
    if model_name == 'pytorch':
        return FacenetLarge(Config.TRIPLET_MODEL_PATH, classes=128)
    elif model_name == 'resnet50':
        return VGGFace(model=model_name)
    elif model_name == 'torch_Xu_facenet':
        model = InceptionResnetV1(
            classify=False,
            pretrained='vggface2'
        )
        model.load_state_dict(torch.load(os.path.join(Config.DATA, 'weights/torch_trained_model_0'),map_location=torch.device('cpu')))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        wrapped_model = model_wrapper(model,device)
        return wrapped_model
        
    elif model_name == 'vggraceimbalanced':
        return load_model(os.path.join(Config.ROOT, 'facenet/customrecurjac/models/vggraceimbalanced-vgg16.hdf5'))
    elif model_name == 'vggracebalanced':
        return load_model(os.path.join(Config.ROOT, 'lipschitz/customrecurjac/models/vggracebalanced-vgg16.hdf5'))
    elif model_name == 'vggimbalanced':
        return load_model(os.path.join(Config.ROOT, 'lipschitz/customrecurjac/models/simplenet_vggimbalanced_generic_fourth.h5'))
    elif model_name == 'vggbalanced':
        return load_model(os.path.join(Config.ROOT, 'lipschitz/customrecurjac/models/simplenet_vggbalanced_generic_fourth.h5'))
    elif model_name == 'vggface2_default_facenet':
        model = InceptionResnetV1(
            classify=False,
            pretrained='vggface2'
        )
        model.load_state_dict(torch.load(os.path.join(Config.DATA, 'weights/torch_trained_model'),map_location=torch.device('cpu')))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        wrapped_model = model_wrapper(model,device)
        return wrapped_model
    elif model_name == 'vggface2_race_balanced_facenet':
        model = InceptionResnetV1(
            classify=False,
            num_classes=1842
        )
        model.load_state_dict(torch.load(os.path.join(Config.DATA, 'weights/torch_race_balanced_trained_model'),map_location=torch.device('cpu')))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        wrapped_model = model_wrapper(model,device)
        return wrapped_model
    elif model_name == 'vggface2_sex_balanced_facenet':
        model = InceptionResnetV1(
            classify=False,
            num_classes=4866
        )
        model.load_state_dict(torch.load(os.path.join(Config.DATA, 'weights/torch_sex_balanced_trained_model'),map_location=torch.device('cpu')))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        wrapped_model = model_wrapper(model,device)
        return wrapped_model
    else:
        return DeepFace.build_model(model_name)


class CenterModel:
    """
    """
    def __init__(self, session=None):
        self.num_channels = 3
        self.image_height = 112
        self.image_width = 96
        self.model = load_model(os.path.join(Config.ROOT, 'weights/face_model_caffe_converted.h5'))

    def predict(self, im):
        return self.model(im)


class TripletModel:
    """
    """
    def __init__(self, session=None):
        self.num_channels = 3
        self.image_height = 96
        self.image_width = 96
        self.image_size = 96

        json_file = open(os.path.join(Config.ROOT, 'models/FRmodel.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        FRmodel = model_from_json(loaded_model_json)
        FRmodel.load_weights(os.path.join(Config.ROOT, "weights/FRmodel.h5"))
        self.model = FRmodel
    
    def predict(self, im):
        return self.model(im)


class FacenetLarge:
    """
    """
    def __init__(self, path, classes=512, session=None):
        self.num_channels = 3
        self.image_height = 160
        self.image_width = 160
        self.image_size = 160

        FRmodel = faceRecoModel((self.image_height, self.image_width, self.num_channels), classes=classes)
        FRmodel.load_weights(os.path.join(Config.ROOT, path))
        self.model = FRmodel
    
    def predict(self, im):
        return self.model(im)

    
def model_wrapper(model,device):
    model.eval()
    model = model.to(device)
    def wrapped_model(x):
        #print(x.shape)
        x_rs = np.transpose(x,[0,3,1,2])
        tensor_x = torch.tensor(x_rs)
        tensor_x = tensor_x.type(torch.FloatTensor)
        tensor_x_gpu = tensor_x.to(device)
        tmp = model(tensor_x_gpu)
        return tmp.cpu().detach().numpy()
    return wrapped_model
