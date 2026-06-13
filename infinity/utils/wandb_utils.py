# Copyright (c) 2026 Applied Intuition, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math
import yaml


def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args):
    config_dict = args.state_dict(key_ordered=False)
    with open(os.path.join(args.local_out_path, "config.yaml"), "w") as f:
        yaml.dump(config_dict, f)
    # wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        project=args.project_name,
        name=args.exp_name,
        config=config_dict,
        id=generate_run_id(args.exp_name),
        # resume="allow",
    )


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(name, sample, epoch):
    if is_main_process():
        sample = array2grid(sample) 

        wandb.log({f"{name}": wandb.Image(sample)}, step=epoch)

def log_checkpoint(path, alias="latest"):
    # Create an artifact and add the file
    if is_main_process():
        print(f"Uploading {path} to wandb")
        artifact = wandb.Artifact(name="model-checkpoint", type="model")
        artifact.add_file(path)
        # Log it to wandb and set an alias (e.g., 'latest', 'best', 'epoch10')
        wandb.log_artifact(artifact, aliases=[alias])

def log_config(path, alias="latest"):
    # Create an artifact and add the file
    if is_main_process():
        print(f"Uploading {path} to wandb")
        artifact = wandb.Artifact(name="model-config", type="model")
        artifact.add_file(path)
        # Log it to wandb and set an alias (e.g., 'latest', 'best', 'epoch10')
        wandb.log_artifact(artifact, aliases=[alias])

def array2grid(x):
    V, T, _, H, W = x.shape
    x = x.transpose(0, 1).reshape(-1, 3, H, W) # T*V x 3 x H x W
    x = make_grid(x, nrow=V, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x

def get_wandb_name():
    return wandb.run.name