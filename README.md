# [CVPR 2026] RayNova: Scale-Temporal Autoregressive World Modeling in Ray Space

**[Project Page](https://raynova-ai.github.io/) | [Paper](https://arxiv.org/abs/2602.20685)**

World foundation models aim to simulate the evolution of the real world with physically plausible behavior. Unlike prior methods that handle spatial and temporal correlations separately, we propose RAYNOVA, a geometry-agonistic multiview world model for driving scenarios that employs a dual-causal autoregressive framework. It follows both scale-wise and temporal topological orders in the autoregressive process, and leverages global attention for unified 4D spatio-temporal reasoning. Different from existing works that impose strong 3D geometric priors, RAYNOVA constructs an isotropic spatio-temporal representation across views, frames, and scales based on relative Plücker-ray positional encoding, enabling robust generalization to diverse camera setups and ego motions. We further introduce a recurrent training paradigm to alleviate distribution drift in long-horizon video generation.

---

## Highlights

- **Versatile World Foundation Model.**: Supporting diverse input and output formats for various conditional generation use cases with a single model.
- **Scalable Data-Driven Framework.**: Ingesting heterogeneous training data from diverse sources with different sensor configurations.
- **Extendable Position Embedding.**: Our relative ray-level positional encoding supports extrapolation beyond the training range
- **Efficient Video Generation.**: Rapid progression from coarse abstractions to fine-grained details

<img width="1398" height="1392" alt="image" src="https://github.com/user-attachments/assets/af2c698a-bfb8-48c3-9054-6223e60e4a2f" />


---

## Installation

```bash
conda create -n raynova python=3.10
conda activate raynova

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Install [ScenarioNet](https://github.com/metadriverse/scenarionet) (required for data conversion):
```bash
# Install MetaDrive Simulator
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e.

# Install ScenarioNet
cd ~/  # Go to the folder you want to host these two repos.
git clone https://github.com/metadriverse/scenarionet.git
cd scenarionet
pip install -e .
```

---

## Weights

Download model weights into the `weights/` directory:

```bash
mkdir -p weights && cd weights

# VAE
wget https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_vae_d32reg.pth

# Pretrained backbone (Infinity 2B)
wget https://huggingface.co/FoundationVision/Infinity/resolve/main/infinity_2b_reg.pth

# Text encoder (Flan-T5-XL)
git clone https://huggingface.co/google/flan-t5-xl

cd ..
```

> **RayNova pretrained checkpoint** (trained on public NuPlan and NuScenes data): *coming soon.*
> We are releasing a model trained exclusively on publicly available data. Performance may be slightly lower than our internal model trained on the full dataset.

---

## Dataset

### Mini Dataset (for demo & training tutorial)

We provide a small sample dataset (`nuplan_sample/`) to demonstrate the training and inference pipeline without requiring access to proprietary data. 

The file structure of sample dataset is shown as follows:
```
nuplan_sample/
└── sample_4/                     # 10 NuPlan scenarios
    ├── dataset_mapping.pkl        # {filename: subdir_name}
    ├── dataset_summary.pkl        # {filename: metadata_dict}
    ├── sample_4_0/
    │   ├── dataset_mapping.pkl
    │   ├── dataset_summary.pkl
    │   └── <scenario_id>.pkl      # pickled ScenarioDescription dict
    ├── sample_4_1/
    └── ...
    └── sensor_blobs/                  # camera images, organised by log/camera/
        └── <log_name>/
            └── <camera>/
                └── <timestamp>.jpg
```

**Data format.** Each scenario `.pkl` file follows the [ScenarioNet](https://github.com/metadriverse/scenarionet) `ScenarioDescription` format. 

You can follow the instructions in [ScenarioNet](https://github.com/metadriverse/scenarionet) to convert nuPlan and other datasets (like nuScenes or WOD) to the desired formats. The fields used by RayNova are:

| Field | Shape / Type | Description |
|-------|-------------|-------------|
| `raw_sensors[t]['images'][cam]['cam_abs_path']` | `str` | Relative path to the camera image (relative to `sensor_blobs/`) |
| `raw_sensors[t]['images'][cam]['cam_intrinsic']` | `[3,3]` | Camera intrinsic matrix |
| `raw_sensors[t]['images'][cam]['sensor2lidar_rotation']` | `[3,3]` | Extrinsic rotation (camera → LiDAR) |
| `raw_sensors[t]['images'][cam]['sensor2lidar_translation']` | `[3]` | Extrinsic translation |
| `raw_sensors[t]['images'][cam]['ego2global_rotation']` | `[4]` | Ego-vehicle → global quaternion |
| `tracks` | `dict` | Per-agent state: position, heading, size, validity |
| `map_features` | `dict` | Map elements with `polyline` / `polygon` geometry |
| `metadata['log_name']` | `str` | Log identifier (used to derive timestamp / location) |
| `language_description` | `list[dict]` | Per-frame text descriptions (optional) |

Camera names: `CAM_F0` (front), `CAM_L0/L1/L2` (left), `CAM_R0/R1/R2` (right), `CAM_B0` (back).

To align heterogeneous data sources, we adopt a unified coordinate system:

1. Ego coordinate: x-axis forward, y-axis left-ward, z-axis upward
2. Camera coordinate: x-axis rightward, y-axis downward, z-axis forward

---

## Training

You can train our model on the mini dataset with the following script:

```bash
bash scripts/train.sh
```

Our default training pipeline includes three stages:

**Stage 1:** Training the model to generate low resolution (192x336) videos:
```bash
# set --pn to '0.06M'
bash scripts/train.sh
```

**Stage 2:** Training the model to generate high resolution (384x672) videos:
```bash
# set "--pn" to '0.25M', "--rush_resume" to the checkpoint of Stage 1
bash scripts/train.sh
```

**Stage 3:** Recurrent training for long-horizonal generation:
```bash
# set --pn to '0.25M', "--tblr" to 1/10 of Stage 2, "--rush_resume" to the checkpoint of Stage 2
bash scripts/train.sh
```

**Training on a single GPU with the mini dataset is primarily intended for verifying the pipeline. For full-scale training, set `--data_path` to your own full dataset.**

---

## Inference

We show the inference prcoess with interactive notebook:

```bash
# Full inference with data loaded from ScenarioNet format
jupyter notebook tools/interactive_infer.ipynb
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{xie2026raynova,
  title={RAYNOVA: Scale-Temporal Autoregressive World Modeling in Ray Space},
  author={Xie, Yichen and Peng, Chensheng and Abdelfattah, Mazen and Hu, Yihan and Yang, Jiezhi and Higgins, Eric and Brigden, Ryan and Tomizuka, Masayoshi and Zhan, Wei},
  journal={arXiv preprint arXiv:2602.20685},
  year={2026}
}
```

---

## Acknowledgements

RayNova builds on [Infinity](https://github.com/FoundationVision/Infinity) and uses data in [ScenarioNet](https://github.com/metadriverse/scenarionet) format. We thank the authors of both projects.
