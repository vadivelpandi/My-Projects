# System Details: AI Content Detector

## Architecture Overview
This project is a full-stack application designed to detect AI-generated content (images and videos) using a Multi-Model Ensemble approach.

- **Frontend**: Single Page Application (SPA) built with React and Vite.
- **Backend**: API server built with Python and FastAPI, utilizing Hugging Face Transformers.
- **Communication**: REST API (POST /analyze).

## Directory Structure
```
/
├── backend/                  # Python FastAPI Server
│   ├── main.py              # Application entry point & API logic
│   ├── model_manager.py     # ML Model management & Ensemble logic
│   ├── requirements.txt     # Python dependencies
│   └── __pycache__/         # Compiled Python files
├── frontend/                 # React Application
│   ├── src/
│   │   ├── App.jsx          # Main component & UI logic
│   │   ├── index.css        # Global styles (Tailwind v4)
│   │   └── main.jsx         # React entry point
│   ├── package.json         # Node.js dependencies
│   ├── postcss.config.js    # PostCSS config
│   ├── tailwind.config.js   # Tailwind config
│   └── vite.config.js       # Vite config
└── SYSTEM_DETAILS.md        # This documentation
```

## Detailed Components

### Frontend (User Interface)
- **Tech Stack**: React 19, Vite, Tailwind CSS v4, Lucide React (Icons).
- **Features**:
  - **Drag & Drop Upload**: Supports Images (JPG, PNG) and Videos (MP4, MOV).
  - **Real-time Feedback**: Loading states with step-by-step progress indicators.
  - **Result Visualization**:
    - **Global Verdict**: Clear AI vs Real classification.
    - **Confidence Score**: Overall probability percentage.
    - **Model Consensus**: Breakdown of how many models voted "AI" vs "Real".
    - **Detailed Breakdown**: Individual results from each of the 11 underlying models.
    - **Forensic Dashboard**: Visualizes (simulated) ELA, Noise, and Frequency metrics.

### Backend (Analysis Engine)
- **Tech Stack**: FastAPI, Uvicorn, Python 3.11+, PyTorch, Transformers.
- **Core Components**:
  - `main.py`: Handles HTTP requests, file processing, and response formatting.
  - `model_manager.py`: Manages the loading and execution of the 11-model ensemble.

- **Logic Breakdown**:
  1.  **Deterministic Analysis**:
      - Calculates MD5 hash of the uploaded file.
      - Seeds the random number generator (`random.seed(hash)`).
      - Ensures identical files always yield identical results.
  2.  **Multi-Model Ensemble**:
      - Aggregates predictions from 11 distinct pre-trained models (e.g., `umm-maybe`, `prithivMLmods`, `google/vit`, etc.).
      - **Consensus**: The final verdict is determined by the majority vote (>50%).
  3.  **Forensic Simulation**:
      - Currently, forensic metrics (ELA, Noise, Frequency) are **derived** from the main ensemble confidence to provide consistent UI feedback without heavy computational overhead of dedicated forensic algorithms.

## Working Process (Data Flow)

### 1. User Upload
- User selects a file.
- Frontend determines type (Image/Video) and previews it.

### 2. Analysis Request
- Frontend sends `POST /analyze` with the file.

### 3. Backend Processing
- **Image Processing**:
  - Loaded into PIL (Python Imaging Library).
  - Passed to `ModelManager.predict()`.
- **Video Processing**:
  - **First Frame Extraction**: The system extracts the first frame of the video.
  - This frame is processed exactly like an image.
  - *Note: Multi-frame temporal analysis is currently not active in strict mode.*

### 4. Ensemble Prediction (`model_manager.py`)
- The image is run through **11 concurrently loaded models**:
  1.  `umm-maybe/AI-image-detector`
  2.  `prithivMLmods/Deep-Fake-Detector-v2-Model`
  3.  `prithivMLmods/deepfake-detector-model-v1`
  4.  `Ateeqq/ai-vs-human-image-detector`
  5.  `jacoballessio/ai-image-detect-distilled`
  6.  `Hemg/AI-VS-REAL-IMAGE-DETECTION`
  7.  `dima806/ai_vs_real_image_detection`
  8.  `Organika/sdxl-detector`
  9.  `Wvolf/ViT_Deepfake_Detection`
  10. `Nahrawy/AIorNot`
  11. `XenArcAI/AIRealNet`
- **Scoring**: Each model provides a label and confidence score.
- **Voting**: 
  - If a model predicts "fake", "ai", "artificial", etc., it counts as an AI Vote.

### 5. Response Construction
- Backend compiles:
  - **Verdict**: AI Generated / Real.
  - **Confidence**: Average confidence of the voting models.
  - **Consensus Data**: Vote counts.
  - **Forensics**: Simulated scores based on deterministically seeded random values tied to the confidence score.

## Running the System

### Prerequisites
- Node.js & npm
- Python 3.10+ & pip
- CUDA compatible GPU (Recommended for faster inference)

### Start Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```
*Server runs at `http://localhost:8000`*

### Start Frontend
```bash
cd frontend
npm install
npm run dev
```
*Client runs at `http://localhost:5173`*