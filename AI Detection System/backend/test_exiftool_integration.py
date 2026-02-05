
import os
import sys
from metadata_engine import MetadataEngine

def test_exiftool_integration():
    print("--- Starting ExifTool Integration Test ---")
    
    # Use existing test image if available
    test_file = "test_v2.jpg"
    if not os.path.exists(test_file):
        print(f"WARNING: {test_file} not found. Creating dummy.")
        with open("dummy_test.jpg", 'wb') as f:
            f.write(b'\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00\x48\x00\x48\x00\x00\xFF\xDB') # Tiny partial JPG
        test_file = "dummy_test.jpg"

    engine = MetadataEngine()
    if not engine.exiftool_path:
        print("FAILURE: ExifTool path not found in engine.")
        return

    print(f"Using ExifTool at: {engine.exiftool_path}")
    
    try:
        report = engine.analyze(test_file)
        
        if "error" in report:
            print(f"FAILURE: Engine returned error: {report['error']}")
        else:
            print("SUCCESS: Engine ran successfully.")
            print("Report Preview:")
            print(f"  Camera Make: {report['camera_info']['device_make']}")
            print(f"  Software: {report['software_trace']['capturing_software']}")
            print(f"  AI Detected: {report['software_trace']['ai_tool_detected']}")
            
            # Check debug keys
            debug_keys = list(report['debug_raw_tags'].keys())
            print(f"  Raw Tags Found: {len(debug_keys)}")
            if len(debug_keys) > 0:
                 print(f"  Sample Tag 1: {debug_keys[0]} = {report['debug_raw_tags'][debug_keys[0]]}")
                 
    except Exception as e:
        print(f"CRITICAL FAILURE: Script crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_exiftool_integration()
