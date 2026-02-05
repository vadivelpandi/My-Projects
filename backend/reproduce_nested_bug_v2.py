
def simulate_analysis_with_fix():
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
    
    print("Testing With Flatten Logic...")
    
    # helper from metadata_engine.py
    def flatten_dict(d, parent_key='', sep=':'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Flatten First
    if any(isinstance(v, dict) for v in raw_data.values()):
        print("Flattening data...")
        raw_data = flatten_dict(raw_data)
        
    print(f"Flattened Keys: {list(raw_data.keys())}")
    
    has_make = False
    has_model = False
    
    # The Fixed Logic
    for key, val in raw_data.items():
        k_lower = key.lower()
        val_str = str(val).strip()
        
        if not val_str or val_str.lower() == "unknown":
            continue
            
        if "make" in k_lower and "lens" not in k_lower:
            has_make = True
            print(f"DEBUG: Found Make in key={key}")

        if "model" in k_lower:
            has_model = True
            print(f"DEBUG: Found Model in key={key}")

    print(f"Result: HasMake={has_make}, HasModel={has_model}")
    
    if not has_make or not has_model:
        print("FAIL: Still failed.")
    else:
        print("SUCCESS: Detected Make/Model in nested structure!")

if __name__ == "__main__":
    simulate_analysis_with_fix()
