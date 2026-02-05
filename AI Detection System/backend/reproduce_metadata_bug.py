
import os
import sys
from metadata_engine import MetadataEngine

# Create a dummy file that definitely triggers the bug
# It just needs to contain "MJ" somewhere and be large enough to look like a file
def create_test_file():
    filename = "dummy_test.bin"
    with open(filename, "wb") as f:
        # random junk
        f.write(b"Some random binary data... ")
        # The trigger
        f.write(b"This is a test file with MJ inside it... ")
        # More junk
        f.write(b"End of file")
    return filename

def test_mj_detection():
    print("--- Starting Reproduction Test ---")
    filename = create_test_file()
    
    try:
        engine = MetadataEngine()
        print(f"Analyzing {filename}...")
        report = engine.analyze(filename)
        
        ai_detected = report['software_trace']['ai_tool_detected']
        ai_tool_name = report['software_trace']['ai_tool_name']
        
        print(f"AI Tool Detected: {ai_detected}")
        print(f"AI Tool Name: {ai_tool_name}")
        
        if "MJ" in ai_tool_name:
            print("FAILURE: 'MJ' was falsely detected as an AI tool!")
        else:
            print("SUCCESS: 'MJ' was NOT detected.")
            
    finally:
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_mj_detection()
