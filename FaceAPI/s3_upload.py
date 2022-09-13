import boto3
import os
import Config
import argparse
from utils.eval_utils import *
from FaceAPI import credentials
import Config
from tqdm import tqdm


s3 = boto3.resource('s3', aws_access_key_id=credentials.aws_access_key_id, aws_secret_access_key=credentials.aws_secret_access_key)


def upload_api(params, target_folder_name, file_list):
    for person in tqdm(file_list):
        for file in person:
            s3.meta.client.upload_file(file, Config.S3_BUCKET, target_folder_name + file.replace(Config.DATA, ''))


if __name__ == '__main__':
    params = Config.parse_and_configure_arguments()
    _, people_targets, file_names = load_images(params, only_files=True)
    target_folder_name = 'fairnessfaces'
    if params['adversarial_flag']:
        source_folder_name = params['adversarial_dir']
    else:
        source_folder_name = params['align_dir']
    upload_api(params, target_folder_name, file_names)
