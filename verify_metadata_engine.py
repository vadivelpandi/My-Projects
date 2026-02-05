
import os
from PIL import Image
import io
try:
    from backend.metadata_engine import MetadataEngine
except ImportError:
    # Handle running from root or backend dir
    import sys
    sys.path.append(os.path.join(os.getcwd(), 'backend'))
    from metadata_engine import MetadataEngine

def create_dummy_image(filename="test_meta.jpg"):
    img = Image.new('RGB', (100, 100), color = (73, 109, 137))
    # Add minimal info
    info = {
        "Software": "Adobe Photoshop 2023",
        "Make": "Canon",
        "Model": "EOS 5D Mark IV"
    }
    # Saving with exif is tricky purely with PIL save, but we can try generic save and see what we get.
    # PIL's save doesn't writing arbitrary dicts to EXIF easily without piexif or similar.
    # We will just save standard.
    img.save(filename, quality=90)
    return filename

def test_metadata_extraction():
    print("--- Starting Metadata Engine Verification ---")
    
    # 1. Create Test Image
    filename = create_dummy_image()
    
    # 2. Run Engine
    engine = MetadataEngine()
    print(f"Analyzing {filename}...")
    
    with open(filename, 'rb') as f:
        file_bytes = f.read()
        
    try:
        report = engine.analyze(file_bytes, is_video=False)
        
        # 3. Validation
        print("\n[Report Keys Check]")
        expected_keys = [
            "file_overview", "camera_info", "software_trace", 
            "metadata_completeness", "metadata_consistency", 
            "forensic_anomalies", "ai_indicators", "metadata_based_conclusion"
        ]
        
        all_present = True
        for key in expected_keys:
            if key in report:
                print(f"✅ {key} present")
            else:
                print(f"❌ {key} MISSING")
                all_present = False
                
        # Check specific values
        print("\n[Content Check]")
        print(f"Media Type: {report['file_overview'].get('media_type')}")
        print(f"Resolution: {report['file_overview'].get('resolution')}")
        
        # Since our dummy image didn't have real Exif injected properly via standard PIL save (which needs `exif` bytes arg),
        # we expect "Unknown" or "Missing" for camera info, but file basics should be there.
        
        if report['file_overview']['media_type'] == "Image":
            print("✅ Media Type Correct")
        else:
            print("❌ Media Type Incorrect")
            
        print("\n--- Full Report Preview ---")
        import json
        print(json.dumps(report, indent=2))
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_metadata_extraction()
