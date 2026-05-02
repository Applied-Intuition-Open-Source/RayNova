# Infinity $\infty$: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis

## Installation
```bash
conda create -n inf python=3.10
conda activate inf
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
mkdir weights && cd weights
wget https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_vae_d32reg.pth
wget https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_2b_reg.pth

# 125 MB
wget https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_vae_d16.pth
wget https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_125M_256x256.pth

curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/google/flan-t5-xl

sudo ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so # if this file does not exist

cd ..
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .
```

```bash
# link dataset
cd Infinity
ln -s /media/training_data/yichen_xie/weights ./
ln -s /media/training_data/nuscenes_golden/nuscenes/ data
ln -s /media/training_data/chensheng_peng/whl ./
unlink weights
lsof -i :12345
conda activate inf_test

lilypad login

# get ip
``` hostname -I | awk '{print $1}' ```

export ARNOLD_ID=0
./train.sh

export ARNOLD_ID=1
./train.sh
pip install lilypad-py==2.5.0 --extra-index-url https://ursa.pypi.applied.dev/simple

# launch a job
lilypad workload launch model_config.yaml

# check the status of training
lilypad cluster status --verbose

# stop a job
lilypad workload stop <workload-id>
```


```bash
# torch 211
conda create -n inf_211 python=3.10
conda activate inf_211

pip install torch==2.1.1
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements_211.txt
```

```bash
# Minimum reproducable error script, debug.py
PYTHONPATH=$(pwd) lilypad experiment launch debug_model_config.yaml

```

```bash
bash scripts/train_debug.sh
bash scripts/train_debug_single.sh
bash scripts/debug.sh
```

# lilypad command
```bash
lilypad cluster status


# baloo
conda activate inf_211
cd Infinity
unlink weights
unlink gen_datasets
PYTHONPATH=$(pwd) lilypad experiment launch model_config.yaml

PYTHONPATH=$(pwd) lilypad experiment launch model_config_vae.yaml

lilypad experiment list

lilypad experiment stop <experiment id>
```


ln -s /media/training_data/yichen_xie/gen_datasets ./
# FID score 

## Generate synthetic dataset
```
# Please create a new environment with nuscenes-devkit after cloning current environment! It has conflicts with some existing packages
cd eval_tools/
python val_image_gen.py  # For single-frame, change img_gen_dir and model_path
python val_video_gen.py  # For multi-frame video, change img_gen_dir and model_path
```

## Get FID
```
cd eval_tools/
python fid_score.py --rootb=/path/to/synthetic/dataset
```

# FVD score 

## Generate synthetic dataset (same as FID)
## Get FVD
```
cd eval_tools/
python get_fvd_features_gen.py  # change dataset_name and gen_data_path
python fvd_from_npy.py ../gen_datasets/fvd_outputs/fvd_feats_ori.npy ../gen_datasets/fvd_outputs/fvd_feats_gen_nuscenes_125M_full_ablation_imgs_299_ep_video.npy
```

# Object condition
## Generate synthetic dataset (same as FID)
## Link it to a specific path
```
ln -s  /path/to/synthetic/dataset /home/applied/yichen_xie/src/Infinity/gen_datasets/synthetic_nuscenes
```
## BEVFusion-C inference
```
# Using oski
tmux a -t bevfusion
torchpack dist-run -np 1 python tools/test.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/default.yaml pretrained/camera-only-det.pth --eval bbox
```

# Motion Planning
```
cd eval_tools/
python ade_metric.py #  change model path
```

# VAE Encode/Decode Example

This repository contains a script for loading a VAE (Variational Autoencoder) model from the Infinity codebase, encoding an image to latent space, and then decoding it back to image space.

## Prerequisites

- Python 3.6+
- PyTorch
- PIL
- NumPy
- Infinity codebase installed

## Installation

Make sure you have the required dependencies:

```bash
pip install torch torchvision pillow numpy
```

## Usage

1. Update the `example_usage.sh` script with the path to your VAE checkpoint and input image:

```bash
# Path to your VAE checkpoint
VAE_CKPT="/path/to/your/vae_checkpoint.pth"

# Path to input image
INPUT_IMAGE="/path/to/your/input_image.jpg"
```

2. Make the script executable:

```bash
chmod +x example_usage.sh
```

3. Run the script:

```bash
./example_usage.sh
```

Alternatively, you can run the Python script directly:

```bash
python vae_encode_decode.py \
    --vae_ckpt /path/to/your/vae_checkpoint.pth \
    --image_path /path/to/your/input_image.jpg \
    --output_path reconstructed_image.png \
    --vae_type 18 \
    --device cuda
```

## Parameters

- `--vae_ckpt`: Path to the VAE checkpoint file
- `--image_path`: Path to the input image
- `--output_path`: Path to save the reconstructed output image
- `--vae_type`: VAE codebook dimension (8, 16, 18, 20, 24, 32, 64, 128)
- `--device`: Device to run on ('cuda' or 'cpu')
- `--apply_spatial_patchify`: Flag to apply spatial patchify

## How it Works

1. The script loads a pre-trained VAE model
2. It processes an input image into the format expected by the model
3. The image is encoded into a latent representation
4. The latent representation is decoded back into an image
5. The reconstructed image is saved to disk

For advanced usage, you can also decode directly from indices by uncommenting the relevant section in the script.
