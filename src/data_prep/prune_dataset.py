import pandas as pd
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Define paths
RAW_DIR = Path("../../data/raw")
PROCESSED_DIR = Path("../../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def is_video_available(video_id: str) -> bool:
    """Checks if a YouTube video is publicly accessible using the oEmbed API."""
    url = f"https://www.youtube.com/oembed?url=http://www.youtube.com/watch?v={video_id}&format=json"
    try:
        response = requests.get(url, timeout=5)
        # Returns 200 if accessible, 404/Unauthorized if private or deleted
        return response.status_code == 200
    except requests.RequestException:
        return False

def prune_split(file_name: str):
    """Loads a CSV, verifies video availability, and saves the pruned version."""
    input_path = RAW_DIR / file_name
    output_path = PROCESSED_DIR / file_name.replace('.csv', '_pruned.csv')
    
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return

    print(f"Processing {file_name}...")
    df = pd.read_csv(input_path)
    
    # Get unique video IDs to minimize network requests
    unique_videos = df['video_id'].unique()
    
    # Use multi-threading to check URLs concurrently
    valid_videos = set()
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(is_video_available, unique_videos), total=len(unique_videos), desc="Checking URLs"))
    
    for vid, is_valid in zip(unique_videos, results):
        if is_valid:
            valid_videos.add(vid)
            
    # Filter the dataframe to only include rows with valid videos
    pruned_df = df[df['video_id'].isin(valid_videos)]
    pruned_df.to_csv(output_path, index=False)
    print(f"Saved {output_path.name}. Retained {len(pruned_df)} out of {len(df)} annotations.\n")

if __name__ == "__main__":
    for split in ['train.csv', 'val.csv', 'test.csv']:
        prune_split(split)