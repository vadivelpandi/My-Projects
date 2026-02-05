import cv2
import numpy as np
import os
import tempfile
from PIL import Image
from metadata_engine import MetadataEngine
from main import extract_frames_from_video

def create_dummy_video(filename, duration_sec=1, fps=10):
    height, width = 64, 64
    # Use mp4v for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for i in range(duration_sec * fps):
        # Create a frame with some moving noise to simulate video
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Add a shifting rectangle to ensure movement/consistency
        cv2.rectangle(frame, (i*2, 10), (i*2+20, 30), (255, 0, 0), -1)
        
        out.write(frame)
        
    out.release()
    print(f"Created dummy video: {filename}")

def test_video_pipeline():
    video_path = "test_video.mp4"
    if os.path.exists(video_path):
        try: os.remove(video_path)
        except: pass

    create_dummy_video(video_path)
    
    with open(video_path, "rb") as f:
        video_bytes = f.read()
        
    print("\n--- Testing Metadata Engine (Video) ---")
    engine = MetadataEngine()
    # Mock video metadata detection
    meta_report = engine.analyze(video_path, is_video=True)
    
    # print all keys for debug
    # print(meta_report) 
    
    encoder = meta_report.get('software_trace', {}).get('encoder_info', 'N/A')
    print(f"Detected Encoder: {encoder}")
    
    # [CHECK] Detection of Lavf
    # Note: cv2 VideoWriter usually writes 'Lavf...' as encoder tag on many systems
    if "Lavf" in encoder or "Lazf" in encoder: 
        print("SUCCESS: Metadata Engine detected Lavf/ffmpeg encoder!")
    elif encoder != "N/A":
        print(f"NOTE: Encoder detected as '{encoder}'. Check if this aligns with suspicious list.")
    else:
        print("WARNING: No encoder info found (common for short internal writes).")

    print("\n--- Testing Frame Extraction (Robustness) ---")
    try:
        frames = extract_frames_from_video(video_bytes, num_frames=5, file_ext=".mp4")
        if frames and len(frames) == 5:
             print(f"SUCCESS: Extracted {len(frames)} frames.")
        else:
             print(f"FAILURE: Extracted {len(frames) if frames else 0} frames.")
    except Exception as e:
        print(f"FAILURE: Exception during extraction: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n--- Testing Consistency Logic (Simulation) ---")
    if frames:
        consistency_scores = []
        for i in range(len(frames) - 1):
            img1 = np.array(frames[i].convert('L'))
            img2 = np.array(frames[i+1].convert('L'))
            err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
            err /= float(img1.shape[0] * img1.shape[1])
            consistency_scores.append(err)
        
        avg_diff = np.mean(consistency_scores)
        print(f"Computed Temporal Diff: {avg_diff:.2f}")
        
        if avg_diff > 0:
            print("SUCCESS: Consistency logic detected breakdown/movement.")
        else:
            print("FAILURE: consistency score is 0 (static video?)")
            
    # Cleanup
    try:
        os.remove(video_path)
    except:
        pass

if __name__ == "__main__":
    test_video_pipeline()
