from metadrive.scenario import ScenarioDescription
import lance
import os
import pickle
from typing import TypedDict
import storage

dataset_path = (
    "s3://research-datasets/unified_datasets/scenarionet_lite_nuplan_full_2025_03_04/"
)
dataset = lance.dataset(
    dataset_path, storage_options=storage.make_s3_storage_options_from_env()
)

for record_batch in dataset.to_batches(
    columns=["scenario_id", "scenario"],
    filter="raw_sensors_valid = true",
    limit=10,
    batch_size=1,
):
    [record] = record_batch.to_pylist()
    scenario_dict = pickle.loads(record["scenario"])
    scenario = ScenarioDescription(scenario_dict)
    import pdb
    pdb.set_trace()
    # embed()