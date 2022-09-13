import os
import numpy as np
import imageio
import cv2
import Config


def pre_proc(img, params):
    """
    Pre-processes images to fit the appropriate model input dimensions.

    Keyword arguments:
    img -- individual image to be processed
    params -- parameter dict (Config)
    """

    interpolation = params['interpolation']
    img_resize = cv2.resize(img, params['image_dims'], interpolation)
    img_BGR = img_resize[...,::-1]
    img_CHW = np.around(img_BGR / 255.0, decimals=12)
    return img_CHW


def crop_face(img, params, detector, known_det=None):
    """
    Crops a single face from an image using MTCNN.

    Keyword arguments:
    img -- image to detect faces from
    params -- parameter dict (Config)
    pnet -- pnet of MTCNN
    rnet -- rnet of MTCNN
    onet -- onet of MTCNN
    known_det -- existing bounding box coordinates
    """

    if params['whitebox_target']:
        model = params['target_model']
    else:
        model = params['model']
    interpolation = params['interpolation']
    
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    margin = 4
    image_width, image_height = params['image_dims']

    if(known_det is not None):
        det = known_det
        img_size = np.asarray(img.shape)[0:2]
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(known_det[0]-margin/2, 0)
        bb[1] = np.maximum(known_det[1]-margin/2, 0)
        bb[2] = np.minimum(known_det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(known_det[3]+margin/2, img_size[0])
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        scaled = cv2.resize(cropped, (image_height, image_width), interpolation)
    else:
        try: 
            bounding_boxes = detector.detect_faces(img)
            nrof_faces = len(bounding_boxes)
            # bounding_boxes, _ = detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            # nrof_faces = bounding_boxes.shape[0]
        except:
            print('Error detecting')
            return None, None, None

        if nrof_faces > 1:
            bounding_boxes = np.expand_dims(bounding_boxes[0], axis=0)

        if nrof_faces == 0:
            return None, None, None
        
        det = bounding_boxes[0]['box']
        features = bounding_boxes[0]['keypoints']
        det[2] += det[0]
        det[3] += det[1]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        det_arr.append(np.squeeze(det))

        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            scaled = cv2.resize(cropped, (image_height, image_width), interpolation)

    face = np.around(scaled / 255.0, decimals=12)
    
    face = np.array(face)
    
    return face, det, features


def apply_delta(delta, img, det, params):
    """
    Resizes and applies delta perturbation layer onto original image.

    Keyword arguments:
    delta -- delta layer with shape of face
    img -- original image containing face
    det -- bounding box coordinates of face
    params -- parameter dict (Config)
    """

    interpolation = params['interpolation']

    adv_img = img * 1
    
    margin = 4
    img_size = np.asarray(img.shape)[0:2]
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])

    orig_dim = [bb[3]-bb[1], bb[2]-bb[0]]

    delta_up = cv2.resize(delta, (orig_dim[1], orig_dim[0]), interpolation)
    adv_img[bb[1]:bb[3],bb[0]:bb[2],:] += delta_up
    adv_img[bb[1]:bb[3],bb[0]:bb[2],:] = np.maximum(adv_img[bb[1]:bb[3],bb[0]:bb[2],:], 0)
    adv_img[bb[1]:bb[3],bb[0]:bb[2],:] = np.minimum(adv_img[bb[1]:bb[3],bb[0]:bb[2],:], 1)

    return adv_img


def read_face_from_aligned(file_list, params):
    """
    Reads and creates numpy array of aligned face images

    Keyword arguments:
    file_list -- list of files within a directory
    params -- parameter dict (Config)
    """

    result = []
    useless = file_list[0]
    for file_name in file_list:
        face = imageio.imread(file_name)
        face = pre_proc(face, params)
        result.append(face)
    result = np.array(result)
    return result
