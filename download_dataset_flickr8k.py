
from pathlib import Path
import requests

BASE = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k"

FILES = {
    "Flickr8k_Dataset.zip": f"{BASE}/Flickr8k_Dataset.zip",
    "Flickr8k_text.zip": f"{BASE}/Flickr8k_text.zip",
}

def download_file(filename, url):
    destination = Path(filename)
    partial = Path(filename + ".part")

    existing_size = partial.stat().st_size if partial.exists() else 0

    headers = {}
    if existing_size:
        headers["Range"] = f"bytes={existing_size}-"

    print(f"\nDownloading: {filename}")

    try:
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=(30, 120),
        ) as response:

            # Resume only if the server confirms a partial response.
            if existing_size and response.status_code == 206:
                mode = "ab"
                print("Resuming previous download...")
            else:
                mode = "wb"
                existing_size = 0
                response.raise_for_status()

            total = response.headers.get("Content-Length")
            total = int(total) + existing_size if total else None
            downloaded = existing_size

            with open(partial, mode) as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)

                        if total:
                            percent = downloaded * 100 / total
                            print(
                                f"\rProgress: {percent:.1f}% "
                                f"({downloaded / 1024**2:.1f} MB)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\rDownloaded {downloaded / 1024**2:.1f} MB",
                                end="",
                                flush=True,
                            )

        # Rename only after the HTTP download completes.
        partial.replace(destination)
        print(f"\nFinished: {destination.resolve()}")

    except Exception as error:
        print(f"\nDownload interrupted: {error}")
        print(f"Partial data retained at: {partial.resolve()}")
        print("Run this script again to try resuming.")

for filename, url in FILES.items():
    download_file(filename, url)

print("\nDownload script finished. Check the messages above.")