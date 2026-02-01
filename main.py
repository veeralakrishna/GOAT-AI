import os
import glob
from src.processor import VideoProcessor
import config

def main():
    print("starting GOAT-AI Framework...")
    
    # search for videos
    video_files = glob.glob(os.path.join(config.VIDEO_DIR, "*.mp4"))
    
    if not video_files:
        print(f"No videos found in {config.VIDEO_DIR}")
        return

    print(f"Found {len(video_files)} videos.")
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        output_path = os.path.join(config.OUTPUT_DIR, f"processed_{filename}")
        
        print(f"--- Processing {filename} ---")
        processor = VideoProcessor(video_path, output_path)
        processor.process()

if __name__ == "__main__":
    main()
