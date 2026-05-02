import lance
import pyarrow as pa
from typing import Iterable, NamedTuple
import pathlib
import pickle

from metadrive.scenario import ScenarioDescription

from ml.rl.scenarionet import types as scenarionet_types
import storage


_SCHEMA = pa.schema(
    {
        "scenario_id": pa.string(),
        "scenario": pa.binary(),
        "raw_sensors_valid": pa.bool_(),
        "raw_sensors_invalid": pa.bool_(),
        "annotation_status": pa.string(),
        "start_timestamp": pa.int64(),
        "run_uuid": pa.string(),
    }
)


class ScenarioToWrite(NamedTuple):
    scenario: ScenarioDescription
    # Other metadata to write to the dataset.
    run_uuid: str


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
                ),
                "annotation_status": item.scenario["metadata"].get(
                    "annotation_status",
                    scenarionet_types.AnnotationStatus.UNSET.value,
                ),
                "start_timestamp": item.scenario["metadata"]["ts"][0],
                "run_uuid": item.run_uuid,
            }
        ]
    )


class ScenarioDatasetBuilder:
    def __init__(self, storage_options: storage.S3StorageOptions | None = None):
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
            _SCHEMA,
            max_bytes_per_file=100 * int(1e6),
            storage_options=self._storage_options,
        )


class ScenarioDatasetUpdater:
    def __init__(
        self, dataset_uri: str, storage_options: storage.S3StorageOptions | None = None
    ):
        self.dataset_uri = dataset_uri
        self._storage_options = storage_options

    def add_if_not_exists(self, items: Iterable[ScenarioToWrite]) -> None:
        dataset = lance.dataset(self.dataset_uri, storage_options=self._storage_options)
        for item in items:
            batch = _get_record_batch(item)
            dataset.merge_insert("scenario_id").when_not_matched_insert_all().execute(
                batch
            )

    def upsert(self, items: Iterable[ScenarioToWrite]) -> None:
        dataset = lance.dataset(self.dataset_uri, storage_options=self._storage_options)
        for item in items:
            batch = _get_record_batch(item)
            dataset.merge_insert(
                "scenario_id"
            ).when_matched_update_all().when_not_matched_insert_all().execute(batch)