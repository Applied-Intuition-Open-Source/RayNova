# RayNova: Scalable World Model for Autonomous Driving

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

Install MetaDrive (required for ScenarioNet data loading):
```bash
git clone https://github.com/metadriverse/metadrive.git
cd metadrive && pip install -e . && cd ..
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

> **RayNova pretrained checkpoint** (trained on public NuPlan data): *coming soon.*
> We are releasing a model trained exclusively on publicly available data. Performance may be slightly lower than our internal model trained on the full dataset.

---

## Dataset

### Mini Dataset (for demo & training tutorial)

We provide a small sample dataset (`nuplan_sample/`) to demonstrate the training and inference pipeline without requiring access to proprietary data.

```
nuplan_sample/
└── sample_10/                     # 10 NuPlan scenarios
    ├── dataset_mapping.pkl        # {filename: subdir_name}
    ├── dataset_summary.pkl        # {filename: metadata_dict}
    ├── sample_10_0/
    │   ├── dataset_mapping.pkl
    │   ├── dataset_summary.pkl
    │   └── <scenario_id>.pkl      # pickled ScenarioDescription dict
    ├── sample_10_1/
    └── ...
└── sensor_blobs/                  # camera images, organised by log/camera/
    └── <log_name>/
        └── <camera>/
            └── <timestamp>.jpg
```

**Data format.** Each scenario `.pkl` file follows the [ScenarioNet](https://github.com/metadriverse/scenarionet) `ScenarioDescription` format. The fields used by RayNova are:

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

**Downloading your own data.** To download scenarios from the full NuPlan dataset in this format, use the provided script:

```bash
python scenarionet_tools/download_nuplan_from_s3.py \
    --local_dir nuplan_sample/sample_10 \
    --num_scenarios 10
```

---

## Training

```bash
bash scripts/train.sh
```

Key arguments in `train.sh` (override defaults from `infinity/utils/arg_util.py`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | `nuplan_sample/sample_10` | Path to ScenarioNet dataset |
| `--lbs` | `1` | Local batch size per GPU |
| `--tblr` | `1e-4` | Transformer learning rate |
| `--freeze_backbone` | `1` | Freeze Infinity backbone, train only new modules |
| `--save_model_ep_freq` | `1` | Save checkpoint every N epochs |

Training on a single GPU with the mini dataset is primarily intended for verifying the pipeline. For full-scale training, set `--data_path` to your full ScenarioNet NuPlan dataset.

---

## Inference

### Interactive notebook

```bash
# Full inference with data loaded from ScenarioNet format
jupyter notebook tools/interactive_infer.ipynb

# Pre-processed demo sample (no dataset required)
jupyter notebook for_demo/interactive_infer_demo.ipynb
```

The demo notebook (`for_demo/interactive_infer_demo.ipynb`) loads a pre-processed sample from `for_demo/demo_sample/` and generates a multi-camera video. The sample format is:

```
demo_sample/
└── frame_{t}/
    ├── item_dict.pkl          # metadata + bboxes + map + camera params
    └── frame_images_v{v}.png  # camera image for view v
```

`item_dict.pkl` contains:

```python
{
  'img_metas': {
    'curr_to_first_lidar_rt': Tensor[4,4],   # current → first-frame LiDAR transform
    'curr_to_prev_lidar_rt':  Tensor[4,4],   # current → previous-frame LiDAR transform
    'location': str,                          # map location
    'timeofday': str,                         # HH:MM
    'description': str,                       # text description of the scene
  },
  'gt_bboxes_3d':       Tensor[N,7],   # LiDAR-frame boxes (x,y,z,l,w,h,yaw)
  'gt_labels_3d':       Tensor[N],     # class indices
  'map_sampled_points': Tensor[M,P,3], # M map elements, P sampled points each
  'map_type_labels':    Tensor[M],     # map element type indices
  'camera_params':      tuple,         # (rot, trans, intrins, post_rot, post_trans)
}
```

---

## Roadmap

- [ ] Release RayNova pretrained model (public-data-only)
- [ ] Release full mini dataset on Hugging Face
- [ ] Inference server / Gradio demo

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{raynova2025,
  title   = {RayNova: Scalable World Model for Autonomous Driving},
  author  = {},
  journal = {},
  year    = {2025},
  url     = {https://raynova-ai.github.io/}
}
```

---

## Acknowledgements

RayNova builds on [Infinity](https://github.com/FoundationVision/Infinity) and uses data in [ScenarioNet](https://github.com/metadriverse/scenarionet) format. We thank the authors of both projects.
