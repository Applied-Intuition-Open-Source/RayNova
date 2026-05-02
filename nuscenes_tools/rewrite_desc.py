import pickle
import os
import json
from tqdm import tqdm
import time
split = "val"
src_path = f"/media/research-datasets/nuscenes/nuscenes_desc_infos_s3_{split}.pkl"
dest_path = f"nuscenes_desc_infos_s3_omnidrive_{split}.pkl"
root_path = f"/media/training_data/nuscenes_golden/nuscenes/data_nusc/desc/{split}"

with open(src_path, 'rb') as file:
    info = pickle.load(file)



for i, item in tqdm(enumerate(info['infos'])):
    sample_idx=item['token']
    old_desc=item['description']
    filepath = os.path.join(root_path, f"{sample_idx}.json")
    if not os.path.exists(filepath):
        print(f"Cannot find file at {filepath}")
    with open(filepath) as fp:
        desc = json.load(fp)
    better_desc = desc['description']
    item['description'] = better_desc
    item['description'] = better_desc

with open(dest_path, "wb") as file:
    pickle.dump(info, file)
    
# sanity check

with open(dest_path, 'rb') as file:
    info = pickle.load(file)
sample = info['infos'][0]
desc = sample['description']
path = sample["lidar_path"]
print(desc)
print(path)