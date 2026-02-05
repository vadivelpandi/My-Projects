
from metadata_engine import MetadataEngine

def test_picasa_detection():
    engine = MetadataEngine()
    
    # Simulate Picasa in metadata
    raw_data = {
        "Software": "Picasa",
        "Make": "Canon",
        "Model": "Canon EOS 5D"
    }
    
    print("Testing Picasa Detection...")
    result = engine._process_exiftool_data(raw_data, "test_file.jpg")
    
    ai_found = result['ai_indicators']['ai_software_signature'] == "Yes"
    tool_name = result['software_trace']['ai_tool_detected']
    
    print(f"AI Detected: {ai_found}")
    print(f"Tool Name: {result['software_trace']['ai_tool_name']}")
    
    if ai_found and "Picasa" in result['software_trace']['ai_tool_name']:
        print("SUCCESS: Picasa correctly flagged as AI.")
    else:
        print("FAIL: Picasa NOT flagged.")

if __name__ == "__main__":
    test_picasa_detection()
