# Fairness and Face Obfuscation

## Background
Supporting code for the paper: Fairness Properties of Face Recognition and Obfuscation Systems -- [https://arxiv.org/abs/2108.02707](https://arxiv.org/abs/2108.02707). We evaluated the demographic fairness properties of several face privacy systems.

## Installation
Using python version 3.8.8
```
git clone https://github.com/wi-pi/fairness_face_obfuscation.git
conda update -n base -c defaults conda
conda create -n fairnessfaces python=3.8
conda activate fairnessfaces
conda install -c anaconda tensorflow-gpu
pip install -r requirements.txt
pip install -e .
```
or
```
conda env create -f environment.yml
```

## Required weights, embeddings, datasets
Download the following folders and place them in the `./data` directory: https://drive.google.com/drive/folders/1xWoGESQEtRPZhIEut-9ID2Yyu25lJDaz?usp=share_link

To download the all embeddings, download `embeddings/all_embeds/` and move the files to the `./data/embeddings/attributes/` folder. To download only the non-adversarial embeddings, download `embeddings/embeds/` and move the files to the `./data/embeddings/attributes/` folder.

## Resources
Results reported in the paper were obtained using a server with 40 CPU cores, 2 Nvidia TITAN Xp's, and 1 Quadro P6000, 125 GB Memory, Ubuntu version 16.04 LTS, CUDA 10.0, NVIDIA Driver 410.104.

Disclaimer: The code has not yet been tested on a variety of platforms.

See https://stackoverflow.com/questions/50622525/which-tensorflow-and-cuda-version-combinations-are-compatible for CUDA - TensorFlow compatibility.

## API evaluation
Note that any API evaluation requires accounts, keys, and an AWS S3 bucket. Below are some links to resources helpful for setting up keys. Follow the step-by-step instructions found in the below links.

AWS S3: https://docs.aws.amazon.com/AmazonS3/latest/dev/access-points.html

AWS S3 - Public Read Access: https://aws.amazon.com/premiumsupport/knowledge-center/read-access-objects-s3-bucket/

Follow the instructions in these 3 links to obtain your public and private keys. Add them to the `FaceAPI/credentials.py` file.

AWS Rekognition: https://docs.aws.amazon.com/rekognition/latest/dg/getting-started.html

Azure Face: https://azure.microsoft.com/en-us/services/cognitive-services/face/#get-started

Face++: https://console.faceplusplus.com/documents/7079083

## Development
#### Setup
```
./scripts/create_directories.sh
```
Creates the necessary subdirectories. Code creates subdirectories in `data/new_adv_imgs`.

#### Generation
```
./scripts/fairness_attack.sh
```
Generates perturbations on a single image-class pair. Code generates perturbations on faces, and outputs results in `data/new_adv_imgs` and `data/adversarial`. Successful generation will be printed through each iteration.

NOTE: If you want to use hinge loss, you must align (detect, crop, and resize) a bucket of your own faces to sizes 160x160 or 96x96. You can use MTCNN to do so. We will integrate support for this shortly.

#### Amplification
```
./scripts/amplify.sh
```
Amplifies the perturbations by an amplification factor (constant at the top of `src/amplify.py`).

#### Uploading
```
./scripts s3_upload.sh
```
Uploads specified images to AWS S3 bucket for API evaluation.

#### Embeddings
```
./scripts/store_embeddings.sh
./scripts/store_adv_embeddings.sh
```
Creates embeddings using a given model and outputs to a .npz file in `data/embeddings/attributes`

#### Evaluation
```
./scripts/validate_on_lfw.sh
./scripts/norm_distance.sh
./scripts/api_eval.sh
./scripts/tcav_all.sh
./lipschitz/customrecurjac/scripts/newvgg<1-12>.sh
python src/read_lipschitz_constants.py
```
Evaluates natural and adversarial accuracy metrics. Also evaluates L2 norms of perturbations. Evaluates adversarial examples on APIs like Face++. Runs TCAV evaluation of skin tones on face recognition model. Runs Lipschitz constant estimations. Finally parses Lipschitz constant estimation results and outputs to csv.

#### Plotting
```
./scripts/plot.sh
./scripts/plot_accuracy.sh
./scripts/parse_norms.sh
python src/plot_tcav.py
./scripts/plot_lipschitz.sh
```
Plots TSNE of embeddings. Plots adversarial and natural accuracy metrics. Plots distribution of L2 perturbation norms. Plots TCAV distributions for identities and skin tones. Plots distribution of estimated Lipschitz constants.

## Link to the papers
https://arxiv.org/abs/2108.02707

## Citation
```
@misc{rosenberg2021fairness,
      title={Fairness Properties of Face Recognition and Obfuscation Systems}, 
      author={Harrison Rosenberg and Brian Tang and Kassem Fawaz and Somesh Jha},
      year={2021},
      eprint={2108.02707},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact
Questions? Contact bjaytang@umich.edu or byron123t@gmail.com with subject: Fairness Faces
