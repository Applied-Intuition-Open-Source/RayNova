import lance
import pathlib
import os
import pickle
import argparse
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Scenario lance dataset (Chicago) ─────────────────────────────────────────
os.environ['AWS_ACCESS_KEY_ID'] = 'cd0146b0fd5c24625a928b242d19f7e0dec18424'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'HN7mDT0pooo+3E40lyab8rrNfIied/33pCbpyrSEDuA='
os.environ['AWS_ENDPOINT_URL'] = 'https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com'
os.environ['AWS_DEFAULT_REGION'] = 'us-chicago-1'

SCENARIO_STORAGE_OPTIONS = {
    "aws_region": "us-chicago-1",
    "aws_endpoint": "https://idskhu5vqvtl.compat.objectstorage.us-chicago-1.oraclecloud.com",
    "aws_access_key_id": os.environ['AWS_ACCESS_KEY_ID'],
    "aws_secret_access_key": os.environ['AWS_SECRET_ACCESS_KEY'],
    "timeout": "300s",
}

# ── Image blobs (Phoenix, research-datasets) ──────────────────────────────────
_IMAGE_ACCESS_KEY = 'cd0146b0fd5c24625a928b242d19f7e0dec18424'
_IMAGE_SECRET_KEY = 'HN7mDT0pooo+3E40lyab8rrNfIied/33pCbpyrSEDuA='
_IMAGE_ENDPOINT   = 'https://idskhu5vqvtl.compat.objectstorage.us-phoenix-1.oraclecloud.com'
_IMAGE_REGION     = 'us-phoenix-1'

SOURCE    = "s3://research-datasets-chicago/unified_datasets/scenarionet_lite_nuplan_full_2025_04_29_lang_desc/"
LOCAL_DIR = pathlib.Path("/media/training_data/yichen_xie/nuplan_sample/sample_10")

_META_FILES = {"dataset_mapping.pkl", "dataset_summary.pkl"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_pkl(path: pathlib.Path, obj) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _make_image_s3_client():
    session = boto3.Session(
        aws_access_key_id=_IMAGE_ACCESS_KEY,
        aws_secret_access_key=_IMAGE_SECRET_KEY,
        region_name=_IMAGE_REGION,
    )
    return session.client("s3", endpoint_url=_IMAGE_ENDPOINT)


def _parse_s3_path(cam_abs_path: str):
    """Return (bucket, key) from an s3://bucket/key path.

    OCI stores images under 'research_datasets' (underscore) in the path but
    the actual bucket name uses a hyphen: 'research-datasets'.
    """
    no_prefix = cam_abs_path[len("s3://"):]
    bucket, key = no_prefix.split("/", 1)
    bucket = bucket.replace("_", "-")   # research_datasets → research-datasets
    return bucket, key


def _download_one_image(
    s3_client,
    cam_abs_path: str,
    cam_rel_path: str,
    images_dir: pathlib.Path,
) -> pathlib.Path:
    """Download a single image; return local path.  Skip if already present."""
    local_path = images_dir / cam_rel_path
    if local_path.exists():
        return local_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    bucket, key = _parse_s3_path(cam_abs_path)
    s3_client.download_file(bucket, key, str(local_path))
    return local_path


def _localize_images(
    scenario_dict: dict,
    images_dir: pathlib.Path,
    s3_client,
    max_workers: int,
) -> dict:
    """Download every image referenced in raw_sensors and rewrite cam_abs_path."""
    raw_sensors = scenario_dict.get("raw_sensors", [])

    # Build task list: (frame_idx, cam_name, cam_abs_path, cam_rel_path)
    tasks = []
    for frame_idx, frame in enumerate(raw_sensors):
        for cam_name, cam_info in frame.get("images", {}).items():
            abs_path = cam_info.get("cam_abs_path", "")
            rel_path = cam_info.get("cam_rel_path", "")
            if abs_path.startswith("s3://") and rel_path:
                tasks.append((frame_idx, cam_name, abs_path, rel_path))

    if not tasks:
        return scenario_dict  # already localized or no images

    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_task = {
            pool.submit(_download_one_image, s3_client, t[2], t[3], images_dir): t
            for t in tasks
        }
        for future in as_completed(future_to_task):
            frame_idx, cam_name, abs_path, rel_path = future_to_task[future]
            try:
                local_path = future.result()
                # Store as a portable relative path (relative to images_dir).
                raw_sensors[frame_idx]["images"][cam_name]["cam_abs_path"] = rel_path
            except Exception as e:
                errors += 1
                print(f"\n  [warn] failed to download {rel_path}: {e}")

    if errors:
        print(f"  [warn] {errors}/{len(tasks)} images failed to download")
    return scenario_dict


def _scenario_needs_localization(scenario_dict: dict) -> bool:
    """Return True if any cam_abs_path still points to S3."""
    for frame in scenario_dict.get("raw_sensors", []):
        for cam_info in frame.get("images", {}).values():
            if cam_info.get("cam_abs_path", "").startswith("s3://"):
                return True
    return False


# ── Main download logic ───────────────────────────────────────────────────────

def download(
    source: str,
    local_dir: pathlib.Path,
    images_dir: pathlib.Path,
    batch_size: int = 16,
    num_scenarios: int | None = None,
    scenarios_per_subdir: int = 1,
    image_workers: int = 32,
) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    subdir_prefix = local_dir.name

    print(f"Opening dataset: {source}")
    ds = lance.dataset(source, storage_options=SCENARIO_STORAGE_OPTIONS)
    total = ds.count_rows()
    limit = num_scenarios if num_scenarios is not None else total
    print(f"Total rows in dataset: {total}, downloading up to: {limit}")
    print(f"Images will be saved to: {images_dir}")

    s3_client = _make_image_s3_client()

    root_mapping: dict[str, str] = {}
    root_summary: dict[str, dict] = {}
    subdir_mapping: dict[str, dict[str, str]] = {}
    subdir_summary: dict[str, dict[str, dict]] = {}

    written = skipped = 0

    for batch in ds.to_batches(batch_size=batch_size, columns=["scenario_id", "scenario"]):
        scenario_ids = batch.column("scenario_id").to_pylist()
        scenario_blobs = batch.column("scenario").to_pylist()

        for scenario_id, blob in zip(scenario_ids, scenario_blobs):
            if written + skipped >= limit:
                break

            subdir_name = f"{subdir_prefix}_{written // scenarios_per_subdir}"
            subdir_path = local_dir / subdir_name
            subdir_path.mkdir(exist_ok=True)

            filename = f"{scenario_id}.pkl"
            out_path = subdir_path / filename

            scenario_dict = pickle.loads(blob)
            metadata = scenario_dict.get("metadata", {})

            if out_path.exists():
                # Re-open existing file to check whether images are already local.
                with open(out_path, "rb") as f:
                    existing = pickle.load(f)
                if _scenario_needs_localization(existing):
                    print(f"\n  [update] localizing images for existing {filename}")
                    existing = _localize_images(existing, images_dir, s3_client, image_workers)
                    _write_pkl(out_path, existing)
                skipped += 1
            else:
                print(f"\n  [{written + skipped + 1}/{limit}] downloading {scenario_id}")
                scenario_dict = _localize_images(scenario_dict, images_dir, s3_client, image_workers)
                _write_pkl(out_path, scenario_dict)
                written += 1

            root_mapping[filename] = subdir_name
            root_summary[filename] = metadata
            subdir_mapping.setdefault(subdir_name, {})[filename] = ""
            subdir_summary.setdefault(subdir_name, {})[filename] = metadata

        done = written + skipped
        if done >= limit:
            break

    print(f"\nWriting index files...")
    _write_pkl(local_dir / "dataset_mapping.pkl", root_mapping)
    _write_pkl(local_dir / "dataset_summary.pkl", root_summary)
    for subdir_name in subdir_mapping:
        subdir_path = local_dir / subdir_name
        _write_pkl(subdir_path / "dataset_mapping.pkl", subdir_mapping[subdir_name])
        _write_pkl(subdir_path / "dataset_summary.pkl", subdir_summary[subdir_name])

    print(f"Done. written={written}  skipped={skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download nuplan lance dataset from S3 to local ScenarioNet format, "
                    "including all referenced camera images."
    )
    parser.add_argument("--source", default=SOURCE, help="S3 lance dataset URI")
    parser.add_argument("--local_dir", type=pathlib.Path, default=LOCAL_DIR,
                        help="Local output directory for scenario pkl files")
    parser.add_argument("--images_dir", type=pathlib.Path,
                        default=LOCAL_DIR / "sensor_blobs",
                        help="Local directory for camera image files")
    parser.add_argument("--batch_size", type=int, default=16, help="Arrow batch size")
    parser.add_argument("--num_scenarios", type=int, default=None,
                        help="Max number of scenarios to download (default: all)")
    parser.add_argument("--scenarios_per_subdir", type=int, default=1,
                        help="Scenarios per subdirectory")
    parser.add_argument("--image_workers", type=int, default=32,
                        help="Parallel threads for image download")
    args = parser.parse_args()

    download(
        args.source, args.local_dir, args.images_dir,
        args.batch_size, args.num_scenarios, args.scenarios_per_subdir,
        args.image_workers,
    )
