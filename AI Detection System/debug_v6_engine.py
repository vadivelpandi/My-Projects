
import os
import sys
# Add current dir to path
sys.path.append(os.getcwd())
from backend.metadata_engine import MetadataEngine
from PIL import Image
import io

def test_engine():
    print("--- 1. Creating Fake Image with Metadata ---")
    img = Image.new('RGB', (100, 100), color='red')
    # Save as JPEG with some info
    info_dict = {"Software": "FakeGen 1.0", "Artist": "TestUser"}
    
    # Save to bytes
    b = io.BytesIO()
    img.save(b, format='JPEG', quality=80) 
    # Note: PIL simple save might not write extensive EXIF without piexif, 
    # but it writes JFIF/Software often.
    # Let's write a file with specific magic strings to test "String Soup"
    
    print("--- 2. Creating Fake Binary with Strings ---")
    # We simulate a "corrupt" file that just has binary + the words "Adobe Photoshop CS6" and "Canon EOS"
    fake_content = b"\xFF\xD8\xFF\xE0" + b"\x00" * 50 + b"Adobe Photoshop CS6" + b"\x00" * 20 + b"Canon EOS 5D" + b"\x00" * 50
    
    engine = MetadataEngine()
    
    print("--- 3. Running Analysis on Fake Binary ---")
    result = engine.analyze(fake_content)
    
    print("\n--- RESULT ---")
    print(f"File Type: {result['file_overview']['file_format']}")
    print(f"Software Trace: {result['software_trace']['capturing_software']}")
    print(f"Edit Software: {result['software_trace']['editing_software']}")
    print(f"Strings Soup Count: {len(result.get('debug_raw_tags', {}).get('Strings_Soup_Sample', []))}")
    
    soup = result.get('debug_raw_tags', {}).get('Strings_Soup_Sample', [])
    print(f"Soup Sample: {soup}")

    if "Adobe Photoshop CS6" in str(soup) or "Canon EOS" in str(soup):
        print("\nSUCCESS: The engine successfully blindly extracted strings!")
    else:
        print("\nFAILURE: The engine missed the obvious ASCII strings.")

if __name__ == "__main__":
    test_engine()
