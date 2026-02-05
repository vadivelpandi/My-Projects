

# from transformers import pipeline, CLIPProcessor, CLIPModel # Moved to lazy load
# import torch # Moved to lazy load
from PIL import Image
import numpy as np
import math
import logging
import time


class ModelManager:

    def __init__(self):
        self.models = {}

        self.model_names = [
            "umm-maybe/AI-image-detector",
            "prithivMLmods/Deep-Fake-Detector-v2-Model",
            "prithivMLmods/deepfake-detector-model-v1",
            "Ateeqq/ai-vs-human-image-detector",
            "jacoballessio/ai-image-detect-distilled",
            "Hemg/AI-VS-REAL-IMAGE-DETECTION",
            "dima806/ai_vs_real_image_detection",
            "Organika/sdxl-detector",
            "Wvolf/ViT_Deepfake_Detection",
            "Nahrawy/AIorNot",
            "XenArcAI/AIRealNet"
        ]
        
        # Initialize dictionary
        for name in self.model_names:
            self.models[name] = None
        
        self.loading_status = {name: "Pending" for name in self.models} # Use self.models keys
        
        # CLIP for Semantic Drift
        self.clip_model = None
        self.clip_processor = None
        self.clip_status = "Pending"
        
        self.start_background_loading()

    def start_background_loading(self):
        import threading
        thread = threading.Thread(target=self._load_models_worker, daemon=True)
        thread.start()

    def _load_models_worker(self):
        print("Starting background model loading...", flush=True)
        try:
            from transformers import pipeline, CLIPProcessor, CLIPModel
            import torch
        except Exception as e:
            print(f"CRITICAL ERROR: Could not import transformers/torch: {e}")
            return

        
        # Load Detection Models
        for name in self.model_names:
            try:
                print(f"Loading {name}...", flush=True)
                self.loading_status[name] = "Loading"
                self.models[name] = pipeline("image-classification", model=name)
                self.loading_status[name] = "Ready"
                print(f"Loaded {name} successfully.", flush=True)
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                self.loading_status[name] = "Failed"
        
        # Load CLIP
        try:
            print("Loading CLIP (openai/clip-vit-base-patch32)...", flush=True)
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_status = "Ready"
            print("CLIP Loaded.", flush=True)
        except Exception as e:
            print(f"Failed to load CLIP: {e}")
            self.clip_status = "Failed"
            
        print("Background model loading complete.", flush=True)

    def predict_full_suite(self, image):
        """
        Executes Steps 1, 3, 4, 8 of the pipeline.
        Optimized with Batch Processing.
        """
        # Prepare all execution targets
        # 0: Original
        # 1: 50% Resize
        # 2: 25% Resize
        # 3-18: 16 Patches (4x4)
        
        # 1. Prepare Images
        w, h = image.size
        img_50 = image.resize((int(w*0.5), int(h*0.5)))
        img_25 = image.resize((int(w*0.25), int(h*0.25)))
        
        # Grid Split (3-18)
        grid_size = 4
        pw, ph = w // grid_size, h // grid_size
        patches_imgs = []
        patch_meta = [] # Store row/col info
        
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * pw
                upper = i * ph
                right = left + pw
                lower = upper + ph
                p_img = image.crop((left, upper, right, lower))
                patches_imgs.append(p_img)
                patch_meta.append({"row": i, "col": j})
                
        # Combine all for batch inference
        all_images = [image, img_50, img_25] + patches_imgs
        
        # 2. Batch Inference
        # Returns list of result dicts properly formatted
        print(f"Running batch inference on {len(all_images)} image views...", flush=True)
        batch_results = self._run_ensemble_batch(all_images)
        
        base_results = batch_results[0]
        res_50 = batch_results[1]
        res_25 = batch_results[2]
        patch_results_list = batch_results[3:]
        
        # 3. Post-Process Consistency
        conf_org = base_results['summary']['average_confidence']
        conf_50 = res_50['summary']['average_confidence']
        conf_25 = res_25['summary']['average_confidence']
        variance = np.var([conf_org, conf_50, conf_25])
        
        # 4. Post-Process Patches
        # Verify patch count matches
        if len(patch_results_list) != len(patch_meta):
            print("Error: Patch result count mismatch")
            
        patch_conflicts = self._analyze_patches_from_results(patch_results_list, patch_meta, base_results['summary']['consensus'])

        # 8. CLIP Semantic Drift (Single image for now, can be batched but usually fast)
        clip_drift = self._analyze_semantic_drift(image)

        return {
            "ensemble": base_results,
            "consistency": {
                "conf_original": conf_org,
                "conf_50": conf_50,
                "conf_25": conf_25,
                "variance": variance,
                "status": "Stable" if variance < 20 else "Unstable (AI Sign)"
            },
            "patches": patch_conflicts,
            "semantic_drift": clip_drift
        }

    def _analyze_patches_from_results(self, results_list, meta_list, global_consensus):
        """
        Analyzes pre-computed patch results.
        """
        grid_size = 4
        patch_detailed = []
        
        ai_patch_count = 0
        real_patch_count = 0
        patch_confidences = []
        
        for idx, res in enumerate(results_list):
            meta = meta_list[idx]
            conf = res['summary']['average_confidence']
            verdict = res['summary']['consensus']
            
            patch_confidences.append(conf)
            
            if verdict == "AI Generated":
                ai_patch_count += 1
            else:
                real_patch_count += 1
                
            patch_detailed.append({
                "id": idx,
                "row": meta['row'], 
                "col": meta['col'],
                "confidence": conf,
                "verdict": verdict
            })

        # 2. Statistics
        total_patches = len(results_list)
        patch_variance = np.var(patch_confidences) if patch_confidences else 0
        
        # 3. Conflict Logic
        is_global_ai = (global_consensus == "AI Generated")
        
        conflict_detected = False
        consistency_level = "High"
        suspected_regions = []
        
        if is_global_ai:
            if real_patch_count > (total_patches * 0.5):
                conflict_detected = True
                consistency_level = "Low"
                suspected_regions.append("Global-Local Disagreement")
            elif patch_variance > 300:
                 consistency_level = "Medium"
        else:
            hotspots = [p for p in patch_detailed if p['verdict'] == "AI Generated" and p['confidence'] > 80]
            if len(hotspots) >= 2:
                conflict_detected = True
                consistency_level = "Low"
                suspected_regions.append(f"Localized AI anomalies detected in {len(hotspots)} sectors")
        
        return {
            "grid_size": f"{grid_size}x{grid_size}",
            "patch_scores": patch_detailed,
            "consistency_level": consistency_level,
            "conflict_detected": "Yes" if conflict_detected else "No",
            "variance": float(round(patch_variance, 2)),
            "ai_patch_count": ai_patch_count,
            "ai_patch_ratio": f"{ai_patch_count}/{total_patches}",
            "suspected_regions": ", ".join(suspected_regions) if suspected_regions else "None"
        }

    def _run_ensemble_batch(self, images):
        """
        Runs all models on a list of images.
        Returns a list of summary dictionaries (one per image).
        """
        num_images = len(images)
        results_per_image = [ [] for _ in range(num_images) ]
        
        # Enhanced weighting
        model_weights = {name: 1.0 for name in self.model_names} 
        
        # Iterate MODELS (Outer Loop)
        for name in self.model_names:
            pipe = self.models.get(name)
            status = self.loading_status.get(name, "Inactive")
            
            if pipe is None:
                # Fill error/inactive for all images for this model
                for i in range(num_images):
                    results_per_image[i].append({
                        "model": name, "verdict": status, "confidence": 0.0, "status": "Inactive"
                    })
                continue
                
            try:
                # Batch Predict
                preds_batch = pipe(images, batch_size=4)
                
                # Process each image's prediction
                for i, preds in enumerate(preds_batch):
                    # Handle if preds is list (top_k) or dict
                    top = preds[0] if isinstance(preds, list) else preds
                    
                    label = top['label'].lower()
                    score = top['score']
                    
                    ai_labels = ['ai', 'fake', 'artificial', 'generated', 'deepfake', 'synthetic', '0']
                    is_ai = any(x in label for x in ai_labels)
                    ai_prob = score if is_ai else (1.0 - score)
                    verdict = "AI" if ai_prob > 0.5 else "Real"
                    
                    results_per_image[i].append({
                        "model": name,
                        "verdict": verdict,
                        "confidence": float(round(ai_prob * 100, 2)),
                        "status": "Active",
                        "raw_score": ai_prob, # for weighted calc
                        "weight": model_weights.get(name, 1.0)
                    })
                    
            except Exception as e:
                print(f"Error running {name} batch: {e}")
                for i in range(num_images):
                    results_per_image[i].append({
                        "model": name, "verdict": "Error", "confidence": 0.0, "status": "Error"
                    })

        # Summarize for each image
        final_summaries = []
        for img_results in results_per_image:
            ai_votes = 0
            total_valid = 0
            weighted_ai_score = 0
            total_weight = 0
            
            clean_results = []
            
            for res in img_results:
                if res['status'] == "Active":
                    total_valid += 1
                    if res['verdict'] == "AI": ai_votes += 1
                    
                    w = res.get('weight', 1.0)
                    prob = res.get('raw_score', 0)
                    weighted_ai_score += (prob * w)
                    total_weight += w
                
                # Clean up temp keys
                r_clean = {k:v for k,v in res.items() if k not in ['raw_score', 'weight']}
                clean_results.append(r_clean)

            if total_weight > 0:
                final_prob = weighted_ai_score / total_weight
            else:
                final_prob = 0
                
            consensus = "AI Generated" if final_prob > 0.5 else "Real"
            
            final_summaries.append({
                "detailed_results": clean_results,
                "summary": {
                    "total_models": total_valid,
                    "ai_votes": ai_votes,
                    "real_votes": total_valid - ai_votes,
                    "average_confidence": float(round(final_prob * 100, 2)),
                    "consensus": consensus
                }
            })
            
        return final_summaries

    def _run_ensemble(self, image):
        """
        Legacy Single Image Wrapper.
        """
        res = self._run_ensemble_batch([image])[0]
        return res


    def _analyze_semantic_drift(self, image):
        """CLIP Semantic Drift."""
        if self.clip_status != "Ready":
            return {"score": 0, "verdict": "CLIP Not Loaded"}
            
        try:
            # We construct a generic prompt vs specific prompts
            # Actually, standard drift is: Image Embedding vs "A photo of a [class]"
            # But we don't know the class.
            # Alternative: Image vs "AI generated image" vs "Real photo"
            
            inputs = self.clip_processor(text=["a real photo", "an ai generated image"], images=image, return_tensors="pt", padding=True)
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1) 
            
            ai_score = float(probs[0][1]) # Index 1 is "ai generated"
            
            return {
                "clip_ai_score": float(round(ai_score * 100, 2)),
                "verdict": "Semantically Artificial" if ai_score > 0.6 else "Natural Semantics"
            }
        except Exception as e:
            return {"score": 0, "verdict": f"Error: {str(e)}"}

    # Backwards compatibility proxy
    def predict(self, image):
        return self._run_ensemble(image)
