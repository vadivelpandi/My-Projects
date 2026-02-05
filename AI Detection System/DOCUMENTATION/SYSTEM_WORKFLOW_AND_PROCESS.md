# SYSTEM WORKFLOW AND PROCESS SPECIFICATION
**Project:** Advanced AI Content Detection System  
**Document Type:** Technical Specification  
**Version:** 1.0  
**Date:** 2026-01-19  

---

## TABLE OF CONTENTS

1. **INTRODUCTION**
    - 1.1 Purpose
    - 1.2 Scope
2. **SYSTEM ARCHITECTURE**
    - 2.1 High-Level Design
    - 2.2 Component Interaction
3. **DETAILED PROCESS FLOW**
    - 3.1 Initialization Phase
    - 3.2 Analysis Pipeline Execution
    - 3.3 Forensic & ML Fusion
4. **DATA STRUCTURES & API**
    - 4.1 Input Specification
    - 4.2 Output JSON Schema
5. **VISUALIZATION & FEEDBACK**
    - 5.1 Frontend Rendering Logic

---

## 1. INTRODUCTION

### 1.1 Purpose
The purpose of this document is to define the end-to-end workflow of the AI Content Detection System. It serves as a comprehensive guide for developers and stakeholders to understand how the system processes media from ingestion to final verdict.

### 1.2 Scope
This specification covers the backend processing logic (FastAPI), the machine learning ensemble orchestration, the digital forensic analysis engine, and the frontend data consumption.

## 2. SYSTEM ARCHITECTURE

The system is built on a **Service-Oriented Architecture (SOA)** with a clear separation of concerns between the presentation layer and the analysis core.

### 2.1 High-Level Design

```mermaid
graph TD
    %% Nodes
    Client[User Client / Browser]
    FE[React Frontend]
    API[FastAPI Backend]
    MM[Model Manager (Singleton)]
    FE_Engine[Forensic Engine]
    
    %% Relationships
    Client -->|1. Upload File| FE
    FE -->|2. POST /analyze| API
    
    subgraph "Backend Core"
        API -->|3. Route Request| MM
        API -->|3b. Trigger Forensics| FE_Engine
        
        MM -->|4. Parallel Inference| Ensemble[ML Model Ensemble]
        FE_Engine -->|5. Compute Metrics| Forensics[ELA / Noise / FFT]
        
        Ensemble -->|6. Local Verdicts| Aggregator[Voting Logic]
        Forensics -->|7. Forensic Metrics| Response[Response Builder]
        
        Aggregator -->|8. Final Verdict| Response
    end
    
    Response -->|9. JSON Payload| FE
    FE -->|10. Render Dashboard| Client
```

## 3. DETAILED PROCESS FLOW

### 3.1 Initialization Phase
*   **Startup**: The `ModelManager` is instantiated as a global singleton in `main.py`.
*   **Lazy Loading**: To optimize server boot time, actual model weights (stored in `~/.cache/huggingface`) are loaded in a background daemon thread.
*   **Health Check**: The `/` root endpoint returns the status of loaded models (`Ready`, `Loading`, or `Failed`).

### 3.2 Analysis Pipeline Execution
When a request hits the `/analyze` endpoint:
1.  **Input Ingestion**: The `UploadFile` stream is read into memory.
2.  **Hashing**: An MD5 hash is calculated to serve as a seed for deterministic operations.
3.  **Media Handling**:
    *   **Images**: Converted to RGB via Pillow.
    *   **Videos**: `OpenCV` extracts 5 keyframes at equal intervals. The first frame is treated as the primary sample for legacy compatibility.

### 3.3 Forensic Analysis & Verdict Generation
The system performs two parallel analyses but derives the final verdict solely from the ML Ensemble:

1.  **ML Ensemble (Verdict Driver)**:
    *   Iterates through 11 classifiers.
    *   Normalizes logits to a 0-100% confidence scale.
    *   Computes `ML_Confidence` and `Vote_Split` (e.g., 9 AI vs 2 Real).
    *   **Verdict**: Determined by simple majority vote of the active models.

2.  **Forensic Analysis (Supplementary Insights)**:
    *   **ELA (Error Level Analysis)**: Identifies compression discrepancies.
    *   **Noise Analysis**: Detects synthetic smoothness.
    *   **Frequency Analysis**: Spots high-frequency anomalies.
    *   Result: Reported separately as forensic metrics for user review.

3.  **Final Decision Logic**:
    ```python
    Final_Confidence = ML_Confidence
    Verdict = "AI Generated" if Consensus_Vote == "AI" else "Real"
    ```

## 4. DATA STRUCTURES & API

### 4.1 Output JSON Schema
The API returns a strictly typed JSON object:

```json
{
  "prediction": "AI Generated",
  "confidence": 98.5,
  "modelConsensus": {
    "totalModels": 11,
    "aiVotes": 10,
    "realVotes": 1,
    "agreement": 90.9
  },
  "forensicAnalysis": {
    "elaScore": 85.2,
    "noisePattern": "Smooth/Synthetic",
    "frequencyAnalysis": "Abnormal",
    "forensicScore": 0.88
  },
  "detailedModels": [
    { "model": "Model_A", "verdict": "AI", "confidence": 99.0 },
    { "model": "Model_B", "verdict": "Real", "confidence": 45.0 }
  ]
}
```

## 5. VISUALIZATION
The Frontend maps these data points to UI components:
*   **`modelConsensus`** → Donut Chart (AI vs Real).
*   **`detailedModels`** → Expandable Accordion List.
*   **`forensicAnalysis`** → Stat Cards with color-coded badges (Red for Suspicious, Green for Natural).
