from pathlib import Path
import zipfile

for filename in ["Flickr8k_Dataset.zip", "Flickr8k_text.zip"]:
    archive_path = Path(filename)

    if not archive_path.exists():
        print("Missing archive:", archive_path)
        continue

    if not zipfile.is_zipfile(archive_path):
        print("Invalid or incomplete ZIP:", archive_path)
        continue

    output_folder = Path(archive_path.stem)
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting {archive_path}...")

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(output_folder)

    print("Extracted to:", output_folder.resolve())

print("\nExtraction process finished.")