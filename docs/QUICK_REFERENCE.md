# Quick Reference Guide

## 🖥️ Using the GUI

Launch the application:
```bash
python3 run_gui.py
```

### 1. Data Collection Tab
- **Collect Human Content**: Scrapes latest articles from RSS feeds defined in `config.yaml`.
- **Generate AI Content**: Uses Ollama/HuggingFace to write articles based on your configured topics.
- **Progress Bars**: Monitor real-time progress.
- **Stop**: Use the yellow "Stop Current Action" button on the right sidebar to cancel.

### 2. Training Tab
- **Model Type**: Choose between `neural_network` (Recommended) or `ensemble`.
- **Hyperparameters**:
  - **Epochs**: How many times to cycle through the dataset (e.g., 50-100).
  - **Learning Rate**: Step size for the optimizer (e.g., 0.001).
  - **Batch Size**: Samples per update step (e.g., 32).
- **Train Model**: Starts the training worker. Results (AUC, Accuracy) will appear automatically.
- **Interactive Params**: Changing these values saves them to `config.yaml` automatically when training starts.

### 3. Detection Tab
- **Text Analysis**: Paste text into the box and click "Analyze".
- **PDF Analysis**: Click "Import PDF" -> "Select File" to analyze entire documents.
- **Results**:
  - **AI Probability**: 0-100% scale.
  - **Highlighted Segments**: Red segments are likely AI-generated.

### 4. Config Tab
- **Edit Config**: Modify `config.yaml` directly with syntax highlighting.
- **Save**: Persists changes to disk immediately.
- **Reload**: Reverts to the file on disk.

---

## 🔌 API Usage

You can run the detection engine as a headless API service.

### Start the Server
```bash
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Endpoints

#### `POST /v1/detect/text`
Analyze a raw text string.

**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/v1/detect/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "The rapid advancement of artificial intelligence..."}'
```

**Response:**
```json
{
  "prediction": "AI",
  "probability_ai": 0.98,
  "confidence": 0.99,
  "explanations": ["Low perplexity", "Repetitive structure"]
}
```

---

## 🛠️ Troubleshooting

### "Files not saving"
- Check the **System Monitor** log in the GUI sidebar. It prints the exact `training_data/ai` path.
- Remember: AI files now use **timestamps** (`ai_2025...txt`) to prevent overwriting.

### "Validation Loss not dropping"
- Try **lowering the Learning Rate** (e.g., from 0.001 to 0.0001).
- Increase **Batch Size** (e.g., to 64).

### "Import Error: module not found"
- Ensure you activated your venv:
  ```bash
  source .venv/bin/activate
  ```
