import pyarrow as pa
import lance
from typing import Iterable, NamedTuple, TypedDict
import pathlib
import os
import pickle
from metadrive.scenario import ScenarioDescription

os.environ['AWS_ACCESS_KEY_ID'] = 'fd082c21e10475d60adc85b01be246745860a21a'
os.environ['AWS_ENDPOINT_URL'] = 'https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'rEW3zxhO6sV802Rg7EqgfA+CLTGh0eqHu0cIcgujauw='
os.environ['AWS_DEFAULT_REGION'] = 'us-phoenix-1'
os.environ['PREAUTH_URL'] = 'https://idskhu5vqvtl.objectstorage.us-phoenix-1.oci.customer-oci.com/p/ofkGTeRQaWyr0mNvkheVidOQYGEjr4OmEhEAi3EECl_UjuMeqtvu8mKr-k22ixWw/n/idskhu5vqvtl/b/research_datasets/o/remote_deps'


class S3StorageOptions(TypedDict):
    aws_endpoint: str
    aws_region: str
    aws_access_key_id: str
    aws_secret_access_key: str
    timeout: str


class S3StorageOptionsFromRole(TypedDict):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str
    timeout: str


def make_s3_storage_options_from_env() -> S3StorageOptions | S3StorageOptionsFromRole:
    """Make parameters for lance to authenticate with S3 compatible storage APIs.

    Documentation: https://lancedb.github.io/lance/read_and_write.html#object-store-configuration

    """
    aws_endpoint = os.getenv("AWS_ENDPOINT_URL")
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_DEFAULT_REGION")
    if (
        aws_endpoint is None
        or aws_access_key_id is None
        or aws_secret_access_key is None
        or aws_region is None
    ):
        # This code path should only be hit in integration tests. Usually the env vars should be set.
        import boto3

        session = boto3.Session()
        credentials = session.get_credentials().get_frozen_credentials()
        return {
            "aws_access_key_id": credentials.access_key,
            "aws_secret_access_key": credentials.secret_key,
            "aws_session_token": credentials.token,
            "timeout": "300s",
        }

    return {
        "aws_endpoint": aws_endpoint,
        "aws_region": aws_region,
        "aws_access_key_id": aws_access_key_id,
        "aws_secret_access_key": aws_secret_access_key,
        "timeout": "300s",
    }


SCHEMA = pa.schema(
    [
        pa.field("scenario_id", pa.string()),
        pa.field("scenario", pa.binary()),
        pa.field("raw_sensors_valid", pa.bool_()),
        pa.field("raw_sensors_invalid", pa.bool_()),
    ]
)

class ScenarioToWrite(NamedTuple):
    scenario_id: str
    scenario: ScenarioDescription
    raw_sensors_valid: bool
    raw_sensors_invalid: bool


def _get_record_batch(item: ScenarioToWrite) -> pa.RecordBatch:
    scenario_id = item.scenario[ScenarioDescription.ID]
    serialized_scenario = pickle.dumps(item.scenario.to_dict())
    return pa.RecordBatch.from_pylist(
        [
            {
                "scenario_id": scenario_id,
                "scenario": serialized_scenario,
                # We assume that all scenarios have valid raw sensor data unless otherwise specified.
                "raw_sensors_valid": item.scenario["metadata"].get(
                    "raw_sensors_valid", True
                ),
                "raw_sensors_invalid": item.scenario["metadata"].get(
                    "raw_sensors_invalid", False
                )
            }
        ]
    )


class ScenarioDatasetBuilder:
    def __init__(self, storage_options: S3StorageOptions | None = None):
        self._storage_options = storage_options

    def build(
        self, items: Iterable[ScenarioToWrite], destination: str | pathlib.Path
    ) -> pa.Table:
        def _batches(
            items_in: Iterable[ScenarioToWrite],
        ) -> Iterable[pa.RecordBatch]:
            for item in items_in:
                yield _get_record_batch(item)

        lance.write_dataset(
            _batches(items),
            destination,
            schema=SCHEMA,
            storage_options=self._storage_options,
        )
        

LOCAL_DIR = pathlib.Path("/home/yichen-xie/src/ScenarioNet/data/nuscenes_scenarionet_map_val")
DESTINATION = "s3://research-datasets/unified_datasets/scenarionet_lite_nuscenes_map_infinity/val"


storage_options = make_s3_storage_options_from_env()

def load_all_scenarios(base_dir: pathlib.Path) -> Iterable[ScenarioToWrite]:
    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("nuscenes_scenarionet_map_"):
            continue

        print(f"📂 Processing {subdir.name}")
        for pkl_file in sorted(subdir.glob("*.pkl")):
            if pkl_file.name in ("dataset_mapping.pkl", "dataset_summary.pkl"):
                continue
            try:
                with open(pkl_file, "rb") as f:
                    scenario_data = pickle.load(f)

                scenario = ScenarioDescription(scenario_data)
                yield ScenarioToWrite(
                    scenario_id=scenario[ScenarioDescription.ID],
                    scenario=scenario,
                    raw_sensors_valid=scenario["metadata"].get("raw_sensors_valid", True),
                    raw_sensors_invalid=scenario["metadata"].get("raw_sensors_invalid", False),
                )
            except Exception as e:
                print(f"⚠️ Skipping {pkl_file}: {e}")

builder = ScenarioDatasetBuilder(storage_options=storage_options)
builder.build(load_all_scenarios(LOCAL_DIR), DESTINATION)
print(f"✅ Successfully uploaded dataset to {DESTINATION}")