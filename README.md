# AUTOMATIC RECONSTRUCTION OF 3-DIMENSIONAL MODELS FROM SINGLE VIEW FACIAL IMAGES

We propose a method to train a deep learning model which produces a 3D models from raw single-view images, without ground-truth 3D, multiple views, 2D/3D keypoints, prior shape models or any other supervision.


## Setup (with [Anaconda](https://www.anaconda.com/))

### 1. Install dependencies:
```
conda env create -f environment.yml
```
OR manually:
```
conda install -c conda-forge scikit-image matplotlib opencv moviepy pyyaml tensorboardX
```


### 2. Install [PyTorch](https://pytorch.org/):
```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```
*Note*: The code is tested with PyTorch 1.2.0 and CUDA 9.2 on CentOS 7. A GPU version is required for training and testing, since the [neural_renderer](https://github.com/daniilidis-group/neural_renderer) package only has GPU implementation. You are still able to run the demo without GPU.


### 3. Install [neural_renderer](https://github.com/daniilidis-group/neural_renderer):
This package is required for training and testing, and optional for the demo. It requires a GPU device and GPU-enabled PyTorch.
```
pip install neural_renderer_pytorch
```

## Datasets
[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) face dataset. Please download the original images (`img_celeba.7z`) from their [website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and run `celeba_crop.py` in `data/` to crop the images.

## Pretrained Models
Download pretrained models using the scripts provided in `pretrained/`, eg:
```
cd pretrained && sh download_pretrained_celeba.sh
```
## Training and Testing
Check the configuration files in `experiments/` and run experiments, eg:
```
python run.py --config experiments/train_celeba.yml --gpu 0 --num_workers 4
```

## To Run
```
python -m demo.demo --input demo/images/human_face --result demo/results/human_face --checkpoint pretrained/pretrained_celeba/checkpoint030.pth
```
