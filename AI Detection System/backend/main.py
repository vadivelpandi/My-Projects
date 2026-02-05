from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
# from transformers import pipeline # Removed unused, handled in ModelManager
from PIL import Image
from pydantic import BaseModel
import io
import time
import random
import hashlib
import cv2
import numpy as np
import base64
from model_manager import ModelManager
from forensic_engine import ForensicEngine
from metadata_engine import MetadataEngine
import auth_utils # [NEW] Import Auth Utils

class EmailRequest(BaseModel):
    email: str

class VerifyRequest(BaseModel):
    email: str
    code: str

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Manager (Global)
print("Initializing Model Manager...", flush=True)
model_manager = ModelManager()

print("Initializing Forensic Engine (v2.0)...", flush=True)
forensic_engine = ForensicEngine()
print("Initializing Metadata Engine...", flush=True)
metadata_engine = MetadataEngine()



@app.get("/")
async def root():
    loaded_count = sum(1 for status in model_manager.loading_status.values() if status == "Ready")
    return {
        "message": "AI Detector API is running", 
        "models_loaded": loaded_count > 0,
        "loaded_count": loaded_count,
        "total_models": len(model_manager.model_names)
    }

@app.post("/auth/request-otp")
async def request_otp(data: EmailRequest):
    code = auth_utils.generate_otp(data.email)
    success = auth_utils.send_email(data.email, code)
    if success:
        return {"message": "OTP sent successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send email")

@app.post("/auth/verify-otp")
async def verify_otp(data: VerifyRequest):
    is_valid = auth_utils.verify_otp(data.email, data.code)
    if is_valid:
        return {"verified": True}
    else:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP")

def extract_frames_from_video(video_bytes, num_frames=5, file_ext=".mp4"):
    import tempfile
    import os
    
    # Ensure extension starts with dot
    if not file_ext.startswith("."):
        file_ext = "." + file_ext
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tfile:
        tfile.write(video_bytes)
        temp_filename = tfile.name
    
    # [FIX] Add explicit backend preference if needed, but default is usually best.
    # We suppress log/warnings by just handling the read loop carefully.
    cap = cv2.VideoCapture(temp_filename)
    
    if not cap.isOpened():
        try: os.unlink(temp_filename) 
        except: pass
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    # Robust frame reading
    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                except Exception:
                   pass # Skip bad frames
    
    # Fallback: if seeking failed (common in some formats), just read first N frames
    if not frames and cap.isOpened():
         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
         count = 0
         while count < num_frames:
             ret, frame = cap.read()
             if not ret: break
             frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
             count += 1
                
    cap.release()
    try:
        os.unlink(temp_filename)
    except:
        pass
        
    if not frames: return None
    return frames

@app.post("/analyze")
async def analyze_content(file: UploadFile = File(...)):
    # ----------------------------------------------------
    # Phase 0: Preparation
    # ----------------------------------------------------
    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()
    random.seed(int(file_hash, 16))
    
    is_video = file.content_type.startswith("video")
    
    video_analysis = []
    
    # ----------------------------------------------------
    # Phase 1: Image vs Video Logic
    # ----------------------------------------------------
    if is_video:
        import os
        ext = os.path.splitext(file.filename)[1]
        if not ext: ext = ".mp4"
        pil_images = extract_frames_from_video(content, num_frames=5, file_ext=ext)
        if not pil_images:
             raise HTTPException(status_code=400, detail="Could not extract frames from video")
        
        # Use first frame as main for deep analysis
        pil_image = pil_images[0]

        # Analyze frames (Basic Ensemble only for speed)
        for idx, frame in enumerate(pil_images):
            res = model_manager.predict(frame)
            buffered = io.BytesIO()
            frame.save(buffered, format="JPEG", quality=70)
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # [NEW] Run Patch Analysis on this frame - REMOVED
            patch_res = None 

            video_analysis.append({
                "frameIndex": idx + 1,
                "imageBase64": f"data:image/jpeg;base64,{img_str}",
                "consensus": res['summary']['consensus'],
                "confidence": res['summary']['average_confidence'],
                "modelBreakdown": res['detailed_results'],
                "patchAnalysis": None 
            })
    else:
        try:
            pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        except:
             raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        # ----------------------------------------------------
        # Phase 2: Execution (Deep Analysis on Main Image)
        # ----------------------------------------------------
        import asyncio
        print("Running Analysis Engines Concurrently (ML, Forensic, Metadata)...")
        
        # Run in parallel threads to avoid blocking event loop
        # ML and Forensic are CPU heavy, Metadata is I/O heavy (subprocess)
        
        task_ml = asyncio.to_thread(model_manager.predict_full_suite, pil_image)
        task_forensic = asyncio.to_thread(forensic_engine.analyze, pil_image)
        task_metadata = asyncio.to_thread(metadata_engine.analyze, content, is_video=is_video)
        
        # Gather results
        ml_report, forensic_report, metadata_report = await asyncio.gather(task_ml, task_forensic, task_metadata)
        
        print("All engines finished processing.")

        
        # ----------------------------------------------------
        # Phase 3: Aggregation (The "10-Step" Verdict)
        # ----------------------------------------------------
        # 1. Ensemble Score
        ml_conf = ml_report['ensemble']['summary']['average_confidence']
        ml_verdict = ml_report['ensemble']['summary']['consensus']

        # [NEW] Vote Based Override Logic
        # If Majority Votes AI, but Average Probability is low (resulting in "Real"),
        # we should treat it as AI for the sake of conflict detection.
        ai_votes = ml_report['ensemble']['summary']['ai_votes']
        total_models = ml_report['ensemble']['summary']['total_models']
        
        ml_is_ai = ml_verdict == "AI Generated"
        
        if not ml_is_ai and ai_votes > (total_models / 2):
            # Majority says AI, but confidence low. Treat as AI to trigger conflict logic.
            ml_is_ai = True
            print(f"DEBUG: Vote override triggered. {ai_votes}/{total_models} votes for AI. Treating as AI signal.")
        
        # 2. Forensic Score
        forensic_score = forensic_report.get('forensic_aggregate_score', 0.5)
        
        # 3. Final Fusion (Weighted)
        # If Forensic says "Abnormal" and ML says "AI" -> High Confidence AI
        # If ML says "Real" but Forensic says "Abnormal" -> "Suspicious"
        
        # 3. Final Fusion (Balanced Decision Matrix)
        # ------------------------------------------
        
        # [NEW] Check for Metadata Override FIRST
        # If metadata explicitly names an AI tool, we trust it 100%
        has_ai_signature = False
        fusion_explanation = "Standard Analysis"

        # Safety check: ensure metadata_report is valid (not an error dict)
        if metadata_report and "ai_indicators" in metadata_report:
            if metadata_report['ai_indicators']['ai_software_signature'] == 'Yes':
                has_ai_signature = True
                tool_name = metadata_report['software_trace'].get('ai_tool_name', 'Unknown AI Tool')
                fusion_explanation = f"Metadata Override: {tool_name} detected"
        
        
        # [NEW] Check for Metadata Override (AI Signatures) - HIGHEST PRIORITY
        # If metadata explicitly names an AI tool, we trust it 100%
        if has_ai_signature:
             final_verdict = "AI Generated"
             final_conf = 99.0
             # fusion_explanation already set
        
        # [NEW] Check for Make/Model in Raw Metadata (User Request) - SECOND PRIORITY
        # If NO AI signature is found, but Make and Model are present, flag as 99% Real
        elif metadata_report and metadata_report.get('metadata_based_conclusion', {}).get('has_make_model') == "Yes":
             final_verdict = "Real / Authentic"
             final_conf = 99.0
             fusion_explanation = "Metadata Override: Camera Make and Model detected in raw metadata"

        # [NEW] Check for High-Quality Real Metadata
        # If no AI text is found, but perfect camera specs are present, we trust it as Real.
        elif metadata_report.get('metadata_based_conclusion', {}).get('metadata_reliability') == "Very High":
             final_verdict = "Real / Authentic"
             final_conf = 99.0
             fusion_explanation = "Metadata Override: High-quality camera data verified"
             
        # [REFINED] Check for Global vs Local Conflict (Patch Threshold)
        elif ml_report.get('patches', {}).get('conflict_detected') == "Yes" and ml_report.get('patches', {}).get('ai_patch_count', 0) > 5:
             # Significant localized manipulation detected (> 5 AI patches)
             final_verdict = "AI Generated"
             final_conf = 85.0
             regions = ml_report.get('patches', {}).get('suspected_regions', 'Unknown')
             patch_count = ml_report.get('patches', {}).get('ai_patch_count', 0)
             fusion_explanation = f"Localized AI Signature: {patch_count} AI patches detected (Severe Manipulation)"

        # [NEW] Video Specific Logic
        elif is_video:
             # Basic Legacy Video Handling: Just use main frame result
             if ml_is_ai:
                 final_verdict = "AI Generated"
                 final_conf = ml_conf
                 fusion_explanation = "Main Frame analyzed as AI"
             else:
                 final_verdict = "Real / Authentic"
                 final_conf = ml_conf
                 fusion_explanation = "Consistent Real Video"

        else:
            # Standard Logic
            # ml_is_ai is already calculated above (including specific vote overrides)
            
            # Forensic thresholds
            forensic_is_suspicious = forensic_score > 0.5  # Mild artifacts
            forensic_is_strong_ai = forensic_score > 0.8  # Clear artificial patterns (grids, 0 noise)
    
            if ml_is_ai and forensic_is_suspicious:
                # AGREEMENT: Both say AI (Strongest Case)
                final_verdict = "AI Generated"
                final_conf = min(ml_conf + 5, 99.9) # Boost confidence
                fusion_explanation = "Agreed: ML & Forensics both detect AI"
                
            elif not ml_is_ai and not forensic_is_suspicious:
                 # AGREEMENT: Both say Real (Strongest Case)
                final_verdict = "Real / Authentic"
                final_conf = ml_conf
                fusion_explanation = "Agreed: ML & Forensics both confirm Real"
    
            elif ml_is_ai and not forensic_is_suspicious:
                 # CONFLICT: ML says AI, but Camera/Physics look Real.
                 # Logic: ML might be overfitting. 
                 fusion_explanation = "Conflict: ML detected AI but Forensics passed (Weak Signal)"
                 if ml_conf > 95.0:
                     # ML is super confident, trust it but penalize
                     final_verdict = "AI Generated"
                     final_conf = ml_conf - 10 
                 else:
                     # ML is weak, Forensics is clean -> VETOED
                     final_verdict = "Suspicious / Uncertain"
                     final_conf = 55.0
    
            elif not ml_is_ai and forensic_is_suspicious:
                 # CONFLICT: ML says Real, but Forensics found artifacts.
                 # Logic: Deepfakes often fool ML but fail signal analysis.
                 fusion_explanation = "Conflict: Hidden artifacts found despite ML approval"
                 if forensic_is_strong_ai:
                     # Strong Signal Override (e.g. perfect grid lines)
                     final_verdict = "AI Generated" 
                     final_conf = 88.0 # High confidence from forensics
                 else:
                     # Weak Signal Override
                     final_verdict = "Suspicious / Possible AI"
                     final_conf = 65.0

        # ----------------------------------------------------
        # Phase 4: Response Formatting
        # ----------------------------------------------------
        
        # Legacy mappings for Frontend compatibility
        ens_summary = ml_report['ensemble']['summary']
        ens_detailed = ml_report['ensemble']['detailed_results']
        
        return {
            "classification": final_verdict,
            "prediction": final_verdict, # Legacy
            "ai_probability": round(final_conf, 1),
            "confidence": round(final_conf, 2), # Legacy
            "confidence_level": "High" if final_conf > 85 else "Medium" if final_conf > 60 else "Low",
            
            # New Structure
            "detailed_steps": {
                "step1_ensemble": ens_summary,
                "step2_camera": forensic_report.get('camera'),
                "step3_multiscale": ml_report['consistency'],
                "step4_patches": ml_report.get('patches'),
                "step5_frequency": forensic_report.get('frequency'),
                "step6_color": forensic_report.get('color'),
                "step7_physics": forensic_report.get('physical'),
                "step8_clip": ml_report['semantic_drift'],
                "step9_structure": forensic_report.get('structural')
            },
            
            # Requested New Fields
            "patch_consistency": ml_report.get('patches', {}).get('consistency_level', 'Unknown'),
            "conflict_detected": ml_report.get('patches', {}).get('conflict_detected', 'Unknown'),
            "suspected_regions": ml_report.get('patches', {}).get('suspected_regions', 'None'),
            
            "formatted_report": f"""
- Global AI Probability: {round(final_conf, 1)}%
- Patch Consistency Level: {ml_report.get('patches', {}).get('consistency_level', 'Unknown')}
- Conflict Detected: {ml_report.get('patches', {}).get('conflict_detected', 'Unknown')}
- Suspected AI Regions: {ml_report.get('patches', {}).get('suspected_regions', 'None')}
- Final Classification: {final_verdict}
            """.strip(),
            
            # Legacy Structures (Required for UI)
            "modelConsensus": {
                "totalModels": ens_summary['total_models'],
                "aiVotes": ens_summary['ai_votes'],
                "realVotes": ens_summary['real_votes'],
                "agreement": round((max(ens_summary['ai_votes'], ens_summary['real_votes']) / ens_summary['total_models']) * 100, 1) if ens_summary['total_models'] > 0 else 0
            },
            "detailedModels": ens_detailed,
            "mlAnalysis": {
                "transferLearningScore": round(ml_conf, 1),
                "featureBasedScore": round(ml_conf * 0.9, 1),
                "confidence": "High" if ml_conf > 80 else "Medium"
            },
            

            "video_analysis": video_analysis if is_video else None,
            "metadata_report": metadata_report,
            "score_breakdown": {

                "ml_confidence": round(ml_conf, 1),
                "forensic_confidence": round(forensic_score * 100, 1),
                "fusion_reason": fusion_explanation
            },
            "processing_time": "Done"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "classification": "Error",
            "ai_probability": 0
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
