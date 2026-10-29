
from pathlib import Path

ROOT = Path(r"E:\image_seo_project")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}

print("=" * 60)
print("IMAGE SEO PROJECT - DATASET VERIFICATION")
print("=" * 60)

if not ROOT.exists():
    print(f"ERROR: Project folder not found: {ROOT}")
    raise SystemExit

image_count = 0
folder_counts = {}
caption_files = []

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    relative = path.relative_to(ROOT)
    dataset_name = relative.parts[0] if relative.parts else "root"

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        image_count += 1
        folder_counts[dataset_name] = (
            folder_counts.get(dataset_name, 0) + 1
        )

    if any(word in path.name.lower()
           for word in ["caption", "token", "metadata"]):
        caption_files.append(path)

print(f"\nProject folder: {ROOT}")
print(f"Total image files found: {image_count}")

print("\n--- Images by dataset/folder ---")
for folder, count in sorted(folder_counts.items()):
    print(f"{folder}: {count} images")

print("\n--- Caption and metadata files ---")
if caption_files:
    for path in caption_files[:40]:
        print(path.relative_to(ROOT))
else:
    print("No caption or metadata files found.")

print("\n--- Sample image paths ---")
shown = 0
for path in ROOT.rglob("*"):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        print(path.relative_to(ROOT))
        shown += 1
        if shown == 10:
            break

print("\nVerification scan completed.")
