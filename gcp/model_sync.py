"""
model_sync.py — Model pre-fetch for GKE workers.

At module load: lists the GCS bucket once and builds a filename → blob_path
lookup cache.

Before each job: parse the workflow, find loader nodes, check each required
model has a sentinel file on disk, download any that are missing. Blocks
until all required models are confirmed on disk.

The GCS directory structure is preserved under LOCAL_MODEL_ROOT, with the
leading "models/" prefix stripped since the mount point IS the models dir.

e.g. gs://bucket/models/unet/wan2.2.safetensors
     → /comfyui/models/unet/wan2.2.safetensors

Downloads use transfer_manager for parallel chunked transfers to maximise
NVMe write throughput on large model files.

Sentinel files (<model_filename>.done) are written only after a successful
download, so a container crash mid-download leaves no sentinel and the file
will be re-downloaded on the next job.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from google.cloud.storage import transfer_manager

# ---------------------------------------------------------------------------
# Config — override via environment variables if needed
# ---------------------------------------------------------------------------

# GCS bucket name only, no gs:// prefix
GCS_BUCKET = os.environ.get("GCS_MODEL_BUCKET", "animaitus-models-europe-west4")

# Local mount point — the NVMe is mounted directly here
LOCAL_MODEL_ROOT = os.environ.get("LOCAL_MODEL_ROOT", "/comfyui/models")

# The prefix all model blobs share in the bucket — stripped when building local path
GCS_MODEL_PREFIX = "models/"

# ---------------------------------------------------------------------------
# Loader node → input field name mapping
# ---------------------------------------------------------------------------

LOADER_MAP = {
    "CheckpointLoaderSimple":  "ckpt_name",
    "CheckpointLoader":        "ckpt_name",
    "UNETLoader":              "unet_name",
    "VAELoader":               "vae_name",
    "CLIPLoader":              "clip_name",
    "DualCLIPLoader":          "clip_name1",
    "LoraLoader":              "lora_name",
    "LoraLoaderModelOnly":     "lora_name",
    "CLIPVisionLoader":        "clip_name",
    "UpscaleModelLoader":      "model_name",
    "ControlNetLoader":        "control_net_name",
}

# DualCLIPLoader has a second clip field
_DUAL_CLIP_EXTRA = "clip_name2"

# ---------------------------------------------------------------------------
# Bucket index — built once at module load
# ---------------------------------------------------------------------------

# filename → full blob path, e.g. "wan2.2.safetensors" → "models/unet/wan2.2.safetensors"
_bucket_index: dict[str, str] = {}


def _build_bucket_index() -> None:
    """
    List all objects in the bucket and build a filename → blob_path lookup.
    Called once at module load so per-job lookups are instant.
    """
    global _bucket_index
    print(f"[model-sync] Building bucket index for gs://{GCS_BUCKET}...")
    t0 = time.time()
    client = storage.Client()
    blobs = client.list_blobs(GCS_BUCKET)
    index = {}
    for blob in blobs:
        filename = blob.name.split("/")[-1]
        if filename:  # skip directory placeholder blobs
            index[filename] = blob.name
    _bucket_index = index
    elapsed = time.time() - t0
    print(f"[model-sync] Bucket index built — {len(_bucket_index)} object(s) in {elapsed:.2f}s")


# Build the index when the module is first imported
_build_bucket_index()


# ---------------------------------------------------------------------------
# Per-job logic
# ---------------------------------------------------------------------------

def extract_required_models(workflow: dict) -> list[tuple[str, str]]:
    """
    Walk every node in the workflow and return a deduplicated list of
    (blob_name, local_path) tuples for every model file referenced.

    blob_name  — full path within the bucket, e.g. "models/unet/wan2.2.safetensors"
    local_path — full local path, e.g. "/comfyui/models/unet/wan2.2.safetensors"
    """
    required = {}  # local_path → blob_name, dedup by local path

    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue

        class_type = node.get("class_type", "")
        inputs = node.get("inputs", {})

        if class_type not in LOADER_MAP:
            continue

        filenames = [inputs.get(LOADER_MAP[class_type])]
        if class_type == "DualCLIPLoader":
            filenames.append(inputs.get(_DUAL_CLIP_EXTRA))

        for filename in filenames:
            if not filename or not isinstance(filename, str):
                continue

            blob_name = _bucket_index.get(filename)
            if not blob_name:
                print(f"[model-sync] WARNING: {filename} not found in bucket index, skipping")
                continue

            # Strip the leading "models/" prefix — the mount point is already
            # the models directory, so we don't want to double it up
            relative_path = blob_name.removeprefix(GCS_MODEL_PREFIX)
            local_path = os.path.join(LOCAL_MODEL_ROOT, relative_path)
            required[local_path] = blob_name

    result = [(blob_name, local_path) for local_path, blob_name in required.items()]
    print(f"[model-sync] Workflow requires {len(result)} model file(s):")
    for blob_name, local_path in result:
        print(f"[model-sync]   gs://{GCS_BUCKET}/{blob_name} → {local_path}")
    return result


def _sentinel_path(local_path: str) -> str:
    return local_path + ".done"


def _ensure_one(client: storage.Client, blob_name: str, local_path: str) -> tuple[bool, str]:
    """
    Ensure a single model file is on disk.
    Checks for a sentinel file — if present, the model is good.
    If absent, downloads the model using transfer_manager (parallel chunked)
    and writes the sentinel on success.
    Returns (success, error_message).
    """
    sentinel = _sentinel_path(local_path)

    if os.path.isfile(sentinel):
        print(f"[model-sync] Sentinel present, skipping: {local_path}")
        return True, ""

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Get file size for speed reporting
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    blob.reload()
    size_bytes = blob.size or 0
    size_mb = size_bytes / (1024 * 1024)

    print(f"[model-sync] Downloading gs://{GCS_BUCKET}/{blob_name} ({size_mb:.1f} MB)...")
    t0 = time.time()

    try:
        transfer_manager.download_chunks_concurrently(
            blob,
            local_path,
        )

        elapsed = time.time() - t0
        speed_mbps = size_mb / elapsed if elapsed > 0 else 0
        print(f"[model-sync] Downloaded OK: {local_path} "
              f"({size_mb:.1f} MB in {elapsed:.1f}s — {speed_mbps:.1f} MB/s)")

        # Write sentinel only after a confirmed successful download
        with open(sentinel, "w") as f:
            f.write("ok")

        return True, ""

    except Exception as e:
        elapsed = time.time() - t0
        print(f"[model-sync] ERROR after {elapsed:.1f}s downloading {blob_name}: {e}")
        # Clean up any partial file so the next attempt starts fresh
        if os.path.isfile(local_path):
            try:
                os.remove(local_path)
            except OSError:
                pass
        return False, str(e)


def ensure_models_on_disk(required_models: list[tuple[str, str]]) -> list[str]:
    """
    Ensure all required models are on disk. Files are downloaded in parallel
    (one thread per file, each using transfer_manager for chunked I/O).
    Returns a list of error strings — empty means all good.
    """
    if not required_models:
        print("[model-sync] No models to fetch.")
        return []

    # One client instance shared across threads (it's thread-safe)
    client = storage.Client()
    errors = []

    t0 = time.time()
    print(f"[model-sync] Checking/fetching {len(required_models)} model(s)...")

    with ThreadPoolExecutor(max_workers=len(required_models)) as executor:
        futures = {
            executor.submit(_ensure_one, client, blob_name, local_path): local_path
            for blob_name, local_path in required_models
        }
        for future in as_completed(futures):
            local_path = futures[future]
            success, err = future.result()
            if not success:
                errors.append(f"{local_path}: {err}")

    elapsed = time.time() - t0
    if errors:
        print(f"[model-sync] Finished with {len(errors)} error(s) in {elapsed:.1f}s")
    else:
        print(f"[model-sync] All models confirmed on disk in {elapsed:.1f}s")

    return errors
