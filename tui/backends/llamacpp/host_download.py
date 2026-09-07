"""Internal cancellable Hugging Face download worker."""
import json
from pathlib import Path
import sys


def main():
    from huggingface_hub import snapshot_download
    spec = json.load(sys.stdin)
    kwargs = {"repo_id": spec["repo"], "allow_patterns": spec["shards"]}
    if spec["workers"] is not None:
        kwargs["max_workers"] = spec["workers"]
    # Prefer a complete local snapshot, including when offline.
    try:
        snapshot = Path(snapshot_download(**kwargs, local_files_only=True))
        if not all((snapshot / shard).is_file() for shard in spec["shards"]):
            raise FileNotFoundError("Incomplete cached GGUF")
    except (OSError, ValueError):
        snapshot = Path(snapshot_download(**kwargs))
    if not all((snapshot / shard).is_file() for shard in spec["shards"]):
        raise RuntimeError("Downloaded snapshot is missing GGUF shards")
    print(json.dumps({"model": str(snapshot / spec["file"])}))


if __name__ == "__main__":
    main()
