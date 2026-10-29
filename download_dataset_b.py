from huggingface_hub import snapshot_download

repo_id = "mismatch-quest/SeeTRUE-Feedback"

print("Downloading Dataset B...")

local_path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir="dataset_b"
)

print("\nDownload completed!")
print("Dataset location:", local_path)
