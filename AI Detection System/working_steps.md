# System Working Steps

This document outlines the end-to-end workflow of the AI Content Detector system, from user interaction to the final forensic verdict.

## 1. User Interaction (Frontend)

The frontend is a React-based web application (`App.jsx`) that serves as the entry point for users.

### 1.1 File Upload & Type Detection
- **Action**: User uploads a file via drag-and-drop or file selector.
- **Support**: Images (JPG, PNG) and Videos (MP4, MOV).
- **Preview**: Immediate client-side preview is rendered.
- **Type Logic**: The app detects MIME types to toggle between Image Mode and Video Mode.

### 1.2 Analysis Trigger
- **Action**: User clicks "Analyze Content".
- **Process**: The file is sent to the backend `/analyze` endpoint via a `POST` request.
- **UI State**: The interface enters an "Analyzing..." state, displaying a 4-step progress indicator (Initializing, Deep Learning, Signal Analysis, Verdict).

### 1.3 Result Visualization
Upon receiving the JSON response, the UI displays:
1.  **Main Verdict**: "AI Generated" or "Real / Authentic" with a confidence score.
2.  **Scoring Breakdown**: Separate scores for ML Consensus, Forensic Likelihood, and the specific "Fusion Logic" used.
3.  **Conflict Region Map**: A 4x4 interactive grid overlaying the image, highlighting regions where local analysis disagrees with the global verdict.
4.  **Model Consensus**: Vote breakdown (AI vs. Real) from the ensemble models.
5.  **Forensic Report**: A multi-step breakdown including Multi-Scale Consistency, Semantic Drift, Physics/Lighting, and Camera Noise.

### 1.4 Secure Raw Metadata Access (OTP)
- **Default State**: Sensitive raw metadata is hidden behind a lock screen.
- **Request Flow**:
    1.  User enters an email address.
    2.  Backend generates a 4-digit OTP and sends it via email (`/auth/request-otp`).
    3.  User enters the code.
    4.  Backend verifies the code (`/auth/verify-otp`).
- **Unlocked State**: Upon success, the full Raw Metadata Dump is revealed, grouped by tag families (e.g., Exif, File, Composite).

---

## 2. Backend Analysis Pipeline

The backend (`main.py`) processes requests in four distinct phases.

### Phase 0: Initialization & Preparation
- **System Startup**: On server start, the system initializes the AI models. This process involves loading heavy `transformers` and `torch` libraries and can take ~30 seconds. During this time, the API may return a "System Initializing" status.
- **Hashing**: An MD5 hash of the file content is generated to seed random number generators for reproducibility.
- **Video Pre-processing**: If the input is a video, 5 key frames are extracted. The first frame is treated as the "Main Image" for deep forensic analysis, while all 5 are run through the ML ensemble.

### Phase 1: Execution (Parallel Engines)
The system runs three specialized engines in parallel:

#### A. Model Manager (`model_manager.py`)
- **Ensemble Analysis**: Runs a suite of 7 active AI detection models using a weighted voting system.
- **Majority Vote Override**: If >50% of models vote "AI", the system treats the signal as AI even if the average confidence is low, preventing false negatives from high-confidence "Real" outliers.
- **Multi-Scale Consistency**: Analyzes the image at 100%, 50%, and 25% resolution to check for stability.
- **Patch Analysis (4x4)**: Splits the image into a 4x4 grid (16 patches) and runs the ensemble on each to find localized artifacts. 
- **Semantic Drift (CLIP)**: Uses OpenAI's CLIP model to compare the image embedding against prompts like "a real photo" vs "an ai generated image".

#### B. Forensic Engine (`forensic_engine.py`)
- **Signal Analysis**: Checks for Camera Noise (PRNU), Error Level Analysis (ELA), and Frequency artifacts (FFT/DWT).
- **Physical Rules**: Validates lighting consistency and color saturation levels.

#### C. Metadata Engine (`metadata_engine.py`)
- **Extraction**: Parses standard Exif/IPTC data and proprietary tags.
- **AI Signature Search**: Scans for known AI software traces (e.g., "Adobe Firefly", "Stable Diffusion").

### Phase 2: Analysis Aggregation
The results from all engines are gathered into a unified report structure.

---

## 3. Verdict Logic (The Decision Matrix)

The final conclusion is determined by a strict priority-based logic in `backend/main.py`.

### Priority 1: Metadata Overrides (Highest)
Before looking at pixels, the system checks for definitive metadata proofs:
1.  **AI Signature**: If an AI tool name is found (e.g., "Midjourney"), Verdict = **AI Generated (99%)**.
2.  **Verified Camera Data**: If `Make` and `Model` tags are present AND no AI signatures exist, Verdict = **Real / Authentic (99%)**.
3.  **High Reliability**: If metadata quality is scored "Very High" by the engine, Verdict = **Real / Authentic (99%)**.

### Priority 2: Patch Conflict (Medium-High)
If metadata is inconclusive, the system checks for localized manipulation in the 4x4 grid:
- **Severe Manipulation (> 5 AI Patches)**: If more than 5 patches detected as AI, Verdict = **AI Generated**. This flags significant localized AI modifications.
- **Minor Conflict (≤ 5 AI Patches)**: If localized AI is present but below the threshold, the system **falls through** to the Hybrid Logic step to weigh this against Global ML and Signal Forensics (avoiding false positives from minor artifacts).

### Priority 3: Standard Hybrid Logic
If no metadata or local conflicts are found, the system weighs ML consensus against Forensic signals:

| ML Verdict | Forensic Signal | Final Verdict | Logic / Explanation |
| :--- | :--- | :--- | :--- |
| **AI** | **Suspicious** | **AI Generated** | **Agreement**: Both systems did detect AI artifacts. |
| **Real** | **Clean** | **Real / Authentic** | **Agreement**: No AI traces found by any system. |
| **AI** | Clean | **Suspicious** | **Conflict**: ML is unsure. If ML confidence is >95%, it stays AI; otherwise, downgraded to Suspicious. |
| **Real** | **Suspicious** | **Suspicious** | **Conflict**: Hidden artifacts found. If artifacts are strong (e.g., perfect grid lines), it overrides to **AI Generated**. |

---

## 4. Response & Reporting

The final JSON response includes:
- **`classification`**: The final text verdict.
- **`ai_probability`**: The unified confidence score.
- **`score_breakdown`**: Individual scores for ML and Forensics to explain the fusion.
- **`detailed_steps`**: Granular data for the 8-10 forensic steps (for the frontend report card).
- **`metadata_report`**: The full parsed metadata (protected by OTP on frontend).
- **`video_analysis`**: Frame-by-frame results (if applicable).
