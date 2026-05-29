import os
import calendar
import requests
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
from io import BytesIO

YEAR = 2025
IMAGES_PER_MONTH = 250
OUTPUT_DIR = "Deggendorf_Uniform_Dataset"
CAMERA = "obsp"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_month_timestamps(year, month):
    days = calendar.monthrange(year, month)[1]
    timestamps = []

    start = datetime(year, month, 1, 0, 0)
    end = datetime(year, month, days, 23, 45)

    current = start
    while current <= end:
        timestamps.append(current)
        current += timedelta(minutes=15)

    return timestamps

def uniform_sample(timestamps, n):
    indices = np.linspace(0, len(timestamps) - 1, n, dtype=int)
    return [timestamps[i] for i in indices]

def download_image(timestamp, save_path):
    image_id = timestamp.strftime("%Y%m%d%H%M")

    url = f"https://webcam.deg.net/bildarchiv.php?id={image_id}&s=&w={CAMERA}"

    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return False

        img = Image.open(BytesIO(response.content))
        img.verify()

        with open(save_path, "wb") as f:
            f.write(response.content)

        return True

    except Exception:
        return False

for month in range(1, 13):
    month_folder = os.path.join(OUTPUT_DIR, f"{YEAR}_{month:02d}")
    os.makedirs(month_folder, exist_ok=True)

    all_times = generate_month_timestamps(YEAR, month)
    selected_times = uniform_sample(all_times, IMAGES_PER_MONTH)

    print(f"\nDownloading month {month:02d}...")

    count = 0

    for timestamp in selected_times:
        filename = timestamp.strftime("%Y_%m_%d_%H_%M") + ".jpg"
        save_path = os.path.join(month_folder, filename)

        if os.path.exists(save_path):
            count += 1
            continue

        success = download_image(timestamp, save_path)

        if success:
            count += 1
            print(f"Downloaded {count}/{IMAGES_PER_MONTH}: {filename}")
        else:
            print(f"Failed: {filename}")

    print(f"Finished month {month:02d}: {count} images downloaded")

print("\nDataset download finished.")