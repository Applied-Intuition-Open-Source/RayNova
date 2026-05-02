from nuscenes_tools.nuscenes_utils.collate import collate as nuscenes_collate

def collate(batch, samples_per_gpu=1):
    output_batch = []
    for tid in range(len(batch[0])):
        sample_batch = []
        for sample_id in range(len(batch)):
            sample_batch.append(batch[sample_id][tid])
        output_batch.append(nuscenes_collate(sample_batch))
    
    return output_batch