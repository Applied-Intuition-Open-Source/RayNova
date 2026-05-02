import pickle 


split = "val"
target = "s3"

assert split in ["train", "val"]
assert target in ["media", "s3"]

if target == "media":
    prefix = "/media/research-datasets/"
else:
    prefix = "s3://research-datasets/"


# src_path = "/home/applied/yichen_xie/src/Infinity/data/nuscenes/nuscenes_desc_infos_%s.pkl"%split
# dest_path = "/home/applied/yichen_xie/src/Infinity/data/nuscenes/nuscenes_desc_infos_%s_%s.pkl"%(target, split)

src_path = "/home/applied/yichen_xie/src/bevfusion/data/nuscenes/nuscenes_infos_val.pkl"
dest_path = "/home/applied/yichen_xie/src/bevfusion/data/nuscenes/synthetic_nuscenes_infos_val.pkl"

with open(src_path, "rb") as file:
    info = pickle.load(file)

for item in info["infos"]:
    for key in item["cams"]:
        data_path = item["cams"][key]["data_path"]
        new_data_path = data_path.replace('./data/nuscenes/samples', '/media/training_data/yichen_xie/gen_datasets/synthetic_nuscenes')
        new_data_path = new_data_path.replace('.jpg', '_gen_0.jpg')
        item["cams"][key]["data_path"] = new_data_path
    
    # item["lidar_path"] = item["lidar_path"].replace("./data/", prefix)

    # for sweep in item["sweeps"]:
    #     sweep["data_path"] = sweep["data_path"].replace("./data/", prefix)


with open(dest_path, "wb") as file:
    pickle.dump(info, file)
