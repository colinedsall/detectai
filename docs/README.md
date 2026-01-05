# DetectAI: AI Text Detection System

A powerful, GUI-based system for detecting AI-generated text using machine learning and neural networks. Features automated data collection, diverse AI content generation, and interactive model training.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Launch the GUI**
   ```bash
   python3 run_gui.py
   ```

3. **Set up API Keys**
   Create a `.api_keys` file in the project root:
   ```bash
   HUGGINGFACE_API_KEY=your_token_here
   ```

## ✨ Key Features

### 🖥️ Interactive GUI
- **Dashboard**: Monitor system resources.
- **Config Editor**: Syntax-highlighted YAML editor for full system control.
- **Data Collection**:
  - Scrape verified human articles from 40+ news sites (BBC, Reuters, etc.).
  - Generate AI samples using LLMs (Ollama/HuggingFace) with configurable topics.
- **Training**:
  - Train Neural Networks (MLP) or Ensemble models.
  - Tune hyperparameters (Epochs, Learning Rate, Batch Size) directly in the UI.
  - Visual analytics (ROC curves, Confusion Matrices).
- **Detection**:
  - Analyze text or PDFs.
  - Color-coded highlighting of AI-suspected segments.

### ⚙️ Customizable
- **AI Topics**: Define what the AI writes about (e.g., "Politics", "Science") in `config.yaml`.
- **Training Params**: Adjust epochs, batch sizes, and model types.
- **Data Sources**: Add your own RSS feeds for human data.

### 🧠 Advanced Models
- **Neural Network**: PyTorch-based MLP with TF-IDF features.
- **Ensemble**: Voting classifier combining Random Forest and Logistic Regression.
- **Robust Features**: Uses perplexity approximation, entropy, and linguistic patterns.

## 📂 File Structure

```
detectai/
├── app/                     # Application source code
│   ├── gui/                 # PyQt6 GUI components
│   ├── services/            # Core logic (Detector, Scraper, Generator)
│   └── main.py              # FastAPI backend
├── scripts/                 # Utility scripts (PDF import, etc.)
├── training_data/           # Collected Human/AI datasets
├── docs/                    # Documentation
├── config.yaml              # Main configuration
└── run_gui.py               # Application entry point
```

## 📚 Documentation

- **[Quick Reference](QUICK_REFERENCE.md)**: Essential commands and GUI guide.
- **[Configuration Guide](CONFIG_USAGE.md)**: Details on `config.yaml` parameters.
- **[Training Workflow](TRAINING_WORKFLOW.md)**: Deep dive into the methodology. (Note: Workflow concepts apply, but use GUI instead of scripts).
