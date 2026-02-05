
import os
import subprocess
import json
import datetime
import re
import shutil

class MetadataEngine:
    """
    v7 ExifTool Engine
    - Uses external ExifTool executable for robust metadata extraction.
    - Maps ExifTool output to unified report structure.
    - Scans metadata values for AI signatures.
    """
    def __init__(self):
        self.ai_signatures = [
            "Stable Diffusion", "Midjourney", "DALL-E", "Imagine", "Leonard.ai",
            "Adobe Firefly", "Bing Image Creator", "Gencraft", "DreamStudio",
            "Photoshop AI", "Generative Fill", "AI-Generated",
            "sd-webui", "ComfyUI", "Automatic1111", "SwarmUI", "Fooocus",
            "NovelAI", "TrinArt", "Waifu Diffusion", "Wonder", "Starryai",
            "civitai", "huggingface", "SDXL",
            "c2pa", "content credentials", "provenance", "synthesized",
            "diffusion", "latent", "neural network", "generative ai",
            # Specific Models including Text Models often in metadata
            "gpt", "openai", "chatgpt", "gemini", "claude", "llama", "mistral",
            # User Requested (Legacy/Other)
            "picasa"
        ]
        
        self.editing_signatures = [
            "Photoshop", "GIMP", "Lightroom", "Canva", "PicsArt", "Snapseed",
            "Affinity Photo", "Paint.NET", "Krita", "After Effects", "Premiere",
            "Ezgif", "ImageMagick", "Lavf", "GoPro"
        ]

        # Locate ExifTool
        self.exiftool_path = self._find_exiftool()

    def _find_exiftool(self):
        # Look for the user-provided folder in project root (assuming we are in backend/)
        # Project structure:
        # /
        #   backend/
        #   exiftool-13.45_64/exiftool.exe
        
        potential_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exiftool-13.45_64", "exiftool.exe"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exiftool-13.45_64", "exiftool(-k).exe"),
            "exiftool" # System path fallback
        ]

        for p in potential_paths:
            if p == "exiftool":
                if shutil.which("exiftool"): return "exiftool"
            elif os.path.exists(p):
                return p
        
        print("WARNING: ExifTool not found in expected paths.")
        return None

    def analyze(self, file_path_or_bytes, is_video=False):
        temp_file = None
        file_path = None
        
        try:
            # Handle bytes/paths
            if isinstance(file_path_or_bytes, bytes):
                import tempfile
                suffix = ".bin" # ExifTool handles binary, but extension helps if known
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
                    t.write(file_path_or_bytes)
                    temp_file = t.name
                file_path = temp_file
            else:
                file_path = file_path_or_bytes
                if not os.path.exists(file_path):
                     return {"error": "File not found"}

            if not self.exiftool_path:
                return {"error": "ExifTool executable not found on server."}

            # RUN EXIFTOOL
            # -json: JSON output
            # -G: Group names (e.g. EXIF:Model)
            # -a: Duplicate tags allowed
            # -u: Unknown tags allowed
            # -g1: Group by specific family 1 (e.g. IFD0, ExifIFD)
            cmd = [self.exiftool_path, "-json", "-G", "-a", "-u", "-g1", file_path]
            
            # Using shell=True for windows .exe sometimes helps, but list is safer usually. 
            # For "exiftool(-k).exe", the parenthesis might need care if shell=True. 
            # subprocess.run with list args handles spaces/chars well on Windows.
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if result.returncode != 0 and not result.stdout:
                return {"error": f"ExifTool failed: {result.stderr}"}
            
            try:
                metadata_list = json.loads(result.stdout)
                if not metadata_list: return {"error": "Empty metadata extraction"}
                metadata = metadata_list[0] # ExifTool returns a list of objects (one per file)
            except json.JSONDecodeError:
                 return {"error": "Failed to parse ExifTool output"}

            return self._process_exiftool_data(metadata, file_path, is_video=is_video)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e), "metadata_reliability": "Low"}
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

    def _process_exiftool_data(self, raw_data, file_path, is_video=False):
        # [NEW] Flatten Metadata if Nested (due to -g/g1 flags)
        # ExifTool -g1 returns {"IFD0": {"Make": "Canon"}, ...}
        # We flatten this to {"IFD0:Make": "Canon", ...} for consistent processing.
        def flatten_dict(d, parent_key='', sep=':'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        if any(isinstance(v, dict) for v in raw_data.values()):
            print(f"--- [DEBUG] Flattening Nested Metadata for {file_path} ---")
            raw_data = flatten_dict(raw_data)
        
        print(f"--- [DEBUG] Raw ExifTool Data for {file_path} ---")
        # print(json.dumps(raw_data, indent=2, default=str)) 
        for k, v in raw_data.items():
             if "Thumbnail" not in k and "DataDump" not in k:
                 print(f"{k}: {str(v)[:100]}")
        print("-------------------------------------------------")
        
        report = self._init_report_structure()
        
        # Helper to find keys case-insensitively or via group
        # Helper to find keys case-insensitively or via group
        def get_val(keys, default="Unknown"):
            if isinstance(keys, str): keys = [keys]
            for k in keys:
                # 1. Direct Exact Match
                if k in raw_data: return str(raw_data[k])
                
                # 2. Case-Insensitive Match
                for clean_key in raw_data.keys():
                    if clean_key.lower() == k.lower():
                        return str(raw_data[clean_key])

                # 3. Suffix Match (e.g. 'Model' finds 'EXIF:Model')
                for rk in raw_data.keys():
                    if rk.endswith(f":{k}") or rk.split(":")[-1] == k:
                         return str(raw_data[rk])
            
            return default

        # 1. File Overview
        report['file_overview']['media_type'] = get_val(["MIMEType", "FileType"], "Unknown")
        report['file_overview']['file_format'] = get_val("FileType", "Unknown")
        report['file_overview']['file_size'] = get_val("FileSize", "0")
        
        w = get_val(["ImageWidth", "SourceImageWidth"])
        h = get_val(["ImageHeight", "SourceImageHeight"])
        if w != "Unknown" and h != "Unknown":
            report['file_overview']['resolution'] = f"{w} x {h}"
            
        report['file_overview']['color_space'] = get_val(["ColorSpace", "ProfileDescription"], "Unknown")
        report['file_overview']['bit_depth'] = get_val(["BitDepth", "BitsPerSample"], "Unknown")

        # 2. Camera Info
        report['camera_info']['device_make'] = get_val(["Make", "Android:Make", "Apple:Make"])
        report['camera_info']['device_model'] = get_val(["Model", "Android:Model", "Apple:Model"])
        report['camera_info']['lens_info'] = get_val(["LensModel", "LensInfo", "LensID"])
        report['camera_info']['capture_timestamp'] = get_val(["DateTimeOriginal", "CreateDate", "CreationDate"])
        
        gps = get_val(["GPSPosition", "GPSLatitude"])
        report['camera_info']['gps_location'] = "Present" if gps != "Unknown" else "Absent"

        # 3. Software Trace & AI Scanning
        software_candidates = []
        for key in ["Software", "CreatorTool", "HistorySoftwareAgent", "ProcessingSoftware", "ApplicationRecordVersion"]:
             val = get_val(key, None)
             if val and val != "Unknown": software_candidates.append(val)
        
        report['software_trace']['capturing_software'] = software_candidates[0] if software_candidates else "Unknown"
        
        # Scan EVERYTHING (Keys and Values) for signatures
        detected_ai = []
        detected_edit = []
        
        # Convert entire raw data to string to catch keys (e.g. 'C2PA') and values
        # logic: if "c2pa" matches a key like "XMP-c2pa:...", we want to catch it.
        # [FIX] Filter out SourceFile and Directory to avoid flagging file paths
        filtered_data = {k: v for k, v in raw_data.items() if k not in ["SourceFile", "Directory", "ExifToolVersion", "FilePermissions"]}
        
        try:
            all_text_dump = json.dumps(filtered_data, default=str).lower()
        except:
            # Fallback if json dump fails (rare)
            all_text_dump = str(filtered_data).lower()
            
        print(f"[DEBUG] AI Scan Dump Length: {len(all_text_dump)}")
        
        for sig in self.ai_signatures:
            if sig.lower() in all_text_dump:
                detected_ai.append(sig)
                
        for sig in self.editing_signatures:
             if sig.lower() in all_text_dump:
                 detected_edit.append(sig)

        detected_ai = list(set(detected_ai))
        detected_edit = list(set(detected_edit))

        if detected_ai:
             print(f"[DEBUG] DETECTED AI SIGNATURES: {detected_ai}")
             report['software_trace']['ai_tool_detected'] = "Yes"
             report['software_trace']['ai_tool_name'] = ", ".join(detected_ai)
             report['ai_indicators']['ai_software_signature'] = "Yes"
        else:
             report['software_trace']['ai_tool_detected'] = "No"

        if detected_edit:
             report['software_trace']['editing_software'] = ", ".join(detected_edit)

        # Encoder (Video)
        if "Video" in report['file_overview']['media_type'] or "MP4" in report['file_overview']['file_format'] or is_video:
             encoder = get_val(["CompressorName", "Encoder", "HandlerDescription", "MajorBrand"], "N/A")
             report['software_trace']['encoder_info'] = encoder  

        # 4. Completeness & Authenticity Scoring
        # Check for "Golden Tags" of real cameras
        golden_tags = ["Make", "Model", "ExposureTime", "ISO", "FNumber", "FocalLength", "DateTimeOriginal"]
        found_golden = []
        for tag in golden_tags:
            if get_val(tag, "Unknown") != "Unknown":
                found_golden.append(tag)
        
        # Reliability Logic
        # User Request: If any 2 golden tags are present (e.g. Make + Model), trust it as Real.
        if len(found_golden) >= 2:
             report['metadata_based_conclusion']['metadata_reliability'] = "Very High"
             report['metadata_based_conclusion']['explanation_summary'] = f"Rich Camera Metadata found ({len(found_golden)}/7 golden tags present)."
        elif report['metadata_completeness']['exif_metadata'] == "Present":
             report['metadata_based_conclusion']['metadata_reliability'] = "High"
             report['metadata_based_conclusion']['explanation_summary'] = "Standard EXIF data present."
        else:
             report['metadata_based_conclusion']['metadata_reliability'] = "Low"
             report['metadata_based_conclusion']['explanation_summary'] = "Minimal or missing metadata."
        
        # [NEW] Explicit Make/Model Check for Forced Real Verdict
        # User Rule: If Make + Model are present in raw metadata -> 99% Real
        # We search RAW keys directly to ensure we don't miss anything due to different naming conventions.
        
        has_make = False
        has_model = False
        
        # Aggressively scan all raw keys
        for key, val in raw_data.items():
            k_lower = key.lower()
            val_str = str(val).strip()
            
            # Skip empty values or "Unknown"
            if not val_str or val_str.lower() == "unknown":
                continue
                
            # Check for Make
            if "make" in k_lower and "lens" not in k_lower and "note" not in k_lower:
                has_make = True
                
            # Check for Model (exclude LensModel, ColorModel, etc.)
            if "model" in k_lower and "lens" not in k_lower and "color" not in k_lower and "release" not in k_lower:
                has_model = True
        
        # Check extraction fallback if raw scan missed (double safety)
        if not has_make:
             if report['camera_info']['device_make'] != "Unknown": has_make = True
        if not has_model:
             if report['camera_info']['device_model'] != "Unknown": has_model = True

        if has_make and has_model:
             report['metadata_based_conclusion']['has_make_model'] = "Yes"
             # Upgrade reliability
             report['metadata_based_conclusion']['metadata_reliability'] = "Very High"
             report['metadata_based_conclusion']['explanation_summary'] = f"Valid Camera Make & Model found in raw metadata."
        else:
             report['metadata_based_conclusion']['has_make_model'] = "No"
        
        report['metadata_completeness']['exif_metadata'] = "Present" if any("EXIF:" in k for k in raw_data.keys()) else "Missing"
        report['metadata_completeness']['xmp_metadata'] = "Present" if any("XMP:" in k for k in raw_data.keys()) else "Missing"
        report['metadata_completeness']['icc_profile'] = "Present" if any("ICC_Profile:" in k for k in raw_data.keys()) else "Missing"

        # 5. Consistency Check (Simple)
        make = report['camera_info']['device_make']
        soft = report['software_trace']['capturing_software']
        if make != "Unknown" and soft != "Unknown" and "Adobe" in soft:
             report['metadata_consistency']['software_overwrite_detected'] = "Yes"
        
        # 6. Conclusion Logic (Preliminary - Main logic in main.py)
        if report['ai_indicators']['ai_software_signature'] == "Yes":
             report['metadata_based_conclusion']['ai_generated_likelihood'] = "High"
             report['metadata_based_conclusion']['explanation_summary'] = f"AI signature found in metadata: {report['software_trace']['ai_tool_name']}"
        elif report['metadata_based_conclusion']['metadata_reliability'] == "Very High":
            report['metadata_based_conclusion']['ai_generated_likelihood'] = "Low"
             # Explanation already set above
        else:
            # Fallback
             pass

        # Debug Dump
        # Just Dump relevant tag keys for debug view
        report['debug_raw_tags'] = {k: str(v)[:200] for k, v in raw_data.items() if "Thumbnail" not in k and "DataDump" not in k}

        # Full Raw Tags for Frontend Display (Sorted)
        # Only filter out large binary blobs
        ignored_keys = ["ThumbnailImage", "DataDump"]
        
        all_raw_tags = []
        for k, v in raw_data.items():
            # Skip binary blobs if key name suggests it
            if any(ign in k for ign in ignored_keys): continue
            
            val_str = str(v)
            # Only skip wildly long binary strings (1000+ chars)
            if len(val_str) > 1000: continue
            
            all_raw_tags.append({"label": k, "value": val_str})
            
        # Sort alphabetically
        all_raw_tags.sort(key=lambda x: x['label'])

        report['raw_tags_display'] = all_raw_tags
        # Keep empty for legacy compatibility if needed, or remove
        report['top_8_tags'] = [] 

        return report

    def _init_report_structure(self):
        return {
            "file_overview": {
                "media_type": "Unknown", "file_format": "Unknown", "file_size": "0", "resolution": "Unknown", "color_space": "Unknown", "bit_depth": "Unknown"
            },
            "camera_info": {
                "device_make": "Unknown", "device_model": "Unknown", "lens_info": "Unknown", "capture_timestamp": "Unknown", "gps_location": "Unknown"
            },
            "software_trace": {
                "capturing_software": "Unknown", "editing_software": "None", "ai_tool_detected": "No", "ai_tool_name": "None", "encoder_info": "N/A"
            },
            "metadata_completeness": {
                "exif_metadata": "Unknown", "xmp_metadata": "Unknown", "iptc_metadata": "Unknown", "icc_profile": "Unknown"
            },
            "metadata_consistency": {
                "camera_vs_software": "Unknown", "timestamp_consistency": "Unknown", "resolution_vs_device": "Unknown", "software_overwrite_detected": "No"
            },
            "forensic_anomalies": {
                "missing_mandatory_exif": "No", "metadata_stripping": "No", "non_standard_tags": "No", "conflicting_entries": "No"
            },
            "ai_indicators": {
                "camera_sensor_data_present": "Unknown", "ai_software_signature": "No", "synthetic_metadata": "No", "screenshot_flag": "No"
            },
            "metadata_based_conclusion": {
                "metadata_reliability": "Medium", "ai_generated_likelihood": "Low", "deepfake_indicator": "No", "explanation_summary": ""
            },
            "debug_raw_tags": {}
        }
