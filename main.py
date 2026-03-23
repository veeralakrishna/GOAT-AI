import os
import glob
from src.bag_processor import BagProcessor
import config

def main():
    print("starting GOAT-AI Framework...")
    
    # search for .bag recordings
    bag_files = glob.glob(os.path.join(config.RECORDINGS_DIR, "*.bag"))
    if not bag_files:
        print(f"No .bag files found in {config.RECORDINGS_DIR}")
        return

    print(f"Found {len(bag_files)} .bag recordings.")
    for bag_path in bag_files:
        filename = os.path.basename(bag_path)
        output_path = os.path.join(config.OUTPUT_DIR, f"processed_{filename}.mp4")
        
        print(f"--- Processing {filename} with PyOrbbecSDK ---")
        processor = BagProcessor(bag_path, output_path)
        processor.process()

if __name__ == "__main__":
    main()
