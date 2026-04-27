import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROCESSED_DIR = Path("../../data/processed")
VIDEO_DIR = Path("../../data/raw/videos") # Assuming downloaded videos are here
FRAME_DIR = PROCESSED_DIR / "frames"
FRAME_DIR.mkdir(parents=True, exist_ok=True)

def extract_frames_for_video(video_id: str, timestamps_ms: list):
    """Extracts precise frames from a video matching the EEV 6Hz timestamps."""
    video_path = VIDEO_DIR / f"{video_id}.mp4"
    
    if not video_path.exists():
        return

    vid_cap = cv2.VideoCapture(str(video_path))
    
    # Create an output directory specific to this video
    vid_out_dir = FRAME_DIR / video_id
    vid_out_dir.mkdir(exist_ok=True)

    for ts in timestamps_ms:
        output_file = vid_out_dir / f"{ts}.jpg"
        if output_file.exists():
            continue # Skip if already processed

        # Set the video reader to the exact millisecond
        vid_cap.set(cv2.CAP_PROP_POS_MSEC, ts)
        success, frame = vid_cap.read()
        
        if success:
            # Resize frame to standardize inputs for CNN (e.g., 224x224 for MobileNet)
            frame_resized = cv2.resize(frame, (224, 224))
            cv2.imwrite(str(output_file), frame_resized)
            
    vid_cap.release()

if __name__ == "__main__":
    train_csv = PROCESSED_DIR / "train_pruned.csv"
    if not train_csv.exists():
        print("Pruned dataset not found. Run prune_dataset.py first.")
        exit()

    df = pd.read_csv(train_csv)
    
    # Group timestamps by video to open each video only once
    video_groups = df.groupby('video_id')['timestamp_ms'].apply(list).to_dict()
    
    print(f"Extracting frames for {len(video_groups)} videos...")
    for vid_id, timestamps in tqdm(video_groups.items(), desc="Processing Videos"):
        extract_frames_for_video(vid_id, timestamps)