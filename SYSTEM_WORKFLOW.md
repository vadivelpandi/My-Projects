# System Workflow: AI Content Detector (System 2.0)

This document outlines the end-to-end workflow of the Advanced AI Content Detection system, featuring a **10-Step Multi-Layer Forensic Code Architecture**.

## 1. User Interaction (Frontend)

### 1.1. Initiation
- **Access**: User navigates to the web application (React/Vite).
- **Interface**: Presented with a clean, dark-themed dashboard featuring a prominent "Upload Content" area.

### 1.2. File Selection
- **Action**: User drags and drops a file or clicks to select one from the file system.
- **Support**:
  - **Images**: JPG, PNG.
  - **Videos**: MP4, MOV.
- **Immediate Feedback**:
  - **Preview**: The application generates a visual preview (thumbnail or video player) of the selected file.
  - **Validation**: Frontend checks file type and prepares it for transmission.

### 1.3. Analysis Request
- **Trigger**: User clicks the "Analyze Content" button.
- **State Change**:
  - UI enters a "Loading" state.
  - Progress indicators appear for "Initializing Model Ensemble", "Running Predictions", and "Aggregating Votes".

---

## 2. API Communication

- **Endpoint**: `POST /analyze`
- **Payload**: `FormData` containing the raw file object.
- **Protocol**: HTTP/1.1 (or 2.0 depending on server config), Asynchronous.

---

## 3. Backend Processing (FastAPI) & The 10-Step Pipeline

The core analysis has been upgraded to a **10-Step Multi-Layer Forensic Pipeline** combining Deep Learning and Signal Processing.

### 3.1. Initialization & Pre-processing
- **Background Loading**: The `ModelManager` loads 11+ AI models and the `CLIP` model in the background upon server start.
- **Input Handling**: The file is hashed (MD5) to seed the random generator for deterministic results.
- **Frame Extraction (Video)**: If video, 5 frames are extracted for deep analysis.

### 3.2. Step 1: Ensemble Deep Learning Analysis
- **Execution**: The image is passed through an ensemble of 11 distinct AI detection models (CNNs and ViTs).
- **Fusion**: A **Confidence-Weighted Fusion** algorithm aggregates the votes. Models with higher historical accuracy (simulated concept) carry more weight.
- **Output**: A preliminary "AI Probability Score" and Confidence Level.

### 3.3. Step 2: Camera Pipeline Forensics
- **PRNU (Photo Response Non-Uniformity)**: Simulates sensor noise analysis to detect if the image lacks the natural noise fingerprint of a physical camera.
- **CFA (Color Filter Array)**: Checks for resampling artifacts inconsistent with Bayer pattern demosaicing.

### 3.4. Step 3: Multi-Resolution Consistency
- **Method**: The image is analyzed at **100%, 50%, and 25%** resolution using a subset of the ensemble.
- **Logic**: AI generators often leave artifacts that persist or amplify differently at lower resolutions compared to real optical downscaling.
- **Verdict**: "Stable", "Unstable", or "Inconclusive".

### 3.5. Step 4: Patch-Level Conflict Detection
- **Method**: The image is split into a **3x3 Grid**.
- **Analysis**: Each patch is analyzed independently.
- **Goal**: Detect "Local Artifacts" (e.g., a perfect face but garbled hands) or "Inpainting" where part of the image is real and part is AI.

### 3.6. Step 5: Frequency-Domain Analysis
- **FFT (Fast Fourier Transform)**: Converts image to frequency domain to spot "Checkerboard Artifacts" common in GANs and Diffusion models.
- **DWT (Discrete Wavelet Transform)**: Analyzes high-frequency sub-bands to detect unnatural smoothness (lack of high-frequency noise).

### 3.7. Step 6: Color Space & Compression
- **Color**: Checks for HSV saturation distribution. AI images often have "Hyper-realistic" or "Perfectly Balanced" histograms that defy natural lighting statistics.
- **Compression**: Analyzes Error Level Analysis (ELA) to find if the compression error is uniform (AI/original) or patchy (manipulated).

### 3.8. Step 7: Semantic–Physical Rules
- **Physics**: Validates shadow consistency and lighting direction (currently rule-based heuristics).
- **Anatomy**: (Planned) Checks for anatomical structural flaws.

### 3.9. Step 8: CLIP Semantic Drift
- **Technique**: Uses OpenAI's **CLIP** model to compare the image embedding against text prompts "Real Photo" vs "AI Generated Image".
- **Logic**: Measures the "Semantic Distance" from the concept of reality. High drift indicates the image shares more semantic features with synthetic data.

### 3.10. Step 9: Structural Analysis
- **Edge Density**: Analyzes the gradient magnitude. AI images often have "soft" edges or lack the micro-contrast of real optical lenses.

### 3.11. Step 10: Explainability & Evidence
- **Reporting**: The system aggregates all previous steps.
- **Logic**:
    - If **Ensemble** says AI -> High Confidence AI.
    - If **Ensemble** says Real BUT **Forensics** (FFT/PRNU) are abnormal -> Reclassifies as "Suspicious/Possible AI".
- **Output**: A comprehensive JSON report.

---

## 4. Result Presentation (Frontend)

### 4.1. The 10-Step Dashboard
- **Legacy Compatibility**: The "Verdict Card" and "Model Consensus" graphs remain for quick reading.
- **New Multi-Layer Card**: A detailed dashboard displaying:
    - **Multi-Scale Stability**: (Stable/Unstable)
    - **Patch Conflicts**: (Number of zones with artifacts)
    - **FFT & Camera Status**: (Natural/Artificial)
    - **Semantic Drift**: (Percentage score from CLIP)
- **Physics & Color**: Metrics for Saturation and Lighting logic.

### 4.2. Video Analysis
- Displays the Consensus and Confidence for the sampled frames, allowing users to spot single-frame injections.
