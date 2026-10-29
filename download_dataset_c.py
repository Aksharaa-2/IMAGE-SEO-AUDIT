
from huggingface_hub import snapshot_download

repo_id = "multilingual-vlm-conflict/coco-counterfactual-conflict"

print("Downloading Dataset C: COCO-Counterfactual Conflict...")
print("Please wait; download time depends on your internet connection.")

try:
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir="dataset_c",
        allow_patterns=[
            "README.md",
            "data/**",
            "data/*",
            ".gitattributes"
        ]
    )

    print("\nDownload completed!")
    print("Dataset location:", local_path)

except Exception as error:
    print("\nDownload failed.")
    print("Error:", error)
    print("Check your internet connection and try again.")
