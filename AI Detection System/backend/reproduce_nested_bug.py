
def simulate_analysis():
    # Simulating ExifTool output with -g1 (Nested)
    raw_data = {
        "SourceFile": "test.jpg",
        "IFD0": {
            "Make": "Canon",
            "Model": "Canon EOS 5D Mark IV",
            "ModifyDate": "2023:01:01 12:00:00"
        },
        "ExifIFD": {
            "DateTimeOriginal": "2023:01:01 12:00:00"
        }
    }
    
    print("Testing Aggressive Scan on Nested Data...")
    has_make = False
    has_model = False
    
    # The Current Logic in metadata_engine.py
    for key, val in raw_data.items():
        k_lower = key.lower()
        val_str = str(val).strip()
        
        # This loop iterates keys: "SourceFile", "IFD0", "ExifIFD"
        # key="IFD0", val=Dict
        
        if not val_str or val_str.lower() == "unknown":
            continue
            
        if "make" in k_lower and "lens" not in k_lower:
            # "make" is NOT in "ifd0" -> Fails
            has_make = True
            print(f"DEBUG: Found Make in key={key}")

        if "model" in k_lower:
            # "model" is NOT in "ifd0" -> Fails
            has_model = True
            print(f"DEBUG: Found Model in key={key}")

    print(f"Result: HasMake={has_make}, HasModel={has_model}")
    
    if not has_make or not has_model:
        print("FAIL: Failed to detect Make/Model in nested IFD0 block.")
    else:
        print("SUCCESS: Detected Make/Model.")

if __name__ == "__main__":
    simulate_analysis()
