# Quick Reference Guide

## Essential Commands

### Setup
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
python3 -m pip install -r requirements.txt
```

### Data Collection
```bash
# Collect training data (interactive)
python3 collect_training_data.py

# Collect data using config.yaml
python3 collect_data_with_config.py

# Quick collection (500 human + 500 AI samples)
python3 -c "
from collect_training_data import TrainingDataCollector
from web_scraper import get_target_sites

c = TrainingDataCollector()
sites = [s for s in get_target_sites() if s.get('rss')]
for s in sites: s['max_pages'] = 50
sites = sites[:10]

human = c.scraper.scrape_multiple_sites(sites)
ai = c.collect_ai_content(num_samples=500, target_words=700)
ds = c.create_training_dataset(human, ai)
c.create_auto_labels(ds)
"
```

### Model Training
```bash
# Train with collected data
python3 train_with_collected_data.py

# Train using config.yaml (complete pipeline)
python3 train_with_config.py

# Quick training
python3 -c "
from train_with_collected_data import train_detector
train_detector()
"
```

### Testing
```bash
# Analyze your files
python3 analyze_my_files.py

# Start API server
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# Test API endpoint
curl -X POST "http://127.0.0.1:8000/v1/detect/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

## Configuration

### config.yaml
```yaml
# Files to analyze
files_to_analyze:
  - "example.txt"
  - "example1.txt"
  - "example2.txt"

# Training configuration
training:
  human_articles_per_site: 50
  ai_samples: 500
  target_words_per_sample: 700
  min_human_words: 400
  max_sites: 10

# Model configuration
model:
  type: "ensemble"
  test_size: 0.2
  cv_folds: 5
```

### Using config.yaml for Training
```bash
# Complete pipeline (data collection + training + testing)
python3 train_with_config.py

# Data collection only using config
python3 collect_data_with_config.py

# Edit config.yaml to customize parameters
nano config.yaml  # or your preferred editor
```

### Configuration Parameters

#### Training Data Collection
- `human_articles_per_site`: Articles to scrape per website (default: 50)
- `ai_samples`: Number of AI samples to generate (default: 500)
- `target_words_per_sample`: Target words per AI sample (default: 700)
- `min_human_words`: Minimum words for human articles (default: 400)
- `max_sites`: Maximum websites to scrape (default: 10)

#### Model Training
- `model.type`: Model type - "ensemble", "random_forest", "logistic_regression" (default: "ensemble")
- `model.test_size`: Test split ratio (default: 0.2)
- `model.cv_folds`: Cross-validation folds (default: 5)

## File Structure

```
detectai/
├── docs/                    # Documentation
├── app/                     # API and services
├── training_data/           # Processed training data
├── scraped_data/            # Raw scraped content
├── web_scraper.py           # Web scraping
├── ai_content_generator.py  # AI content generation
├── collect_training_data.py # Data collection
├── train_with_collected_data.py # Training
├── analyze_my_files.py      # Testing interface
├── config.yaml              # Configuration
└── ai_detector_model.pkl    # Trained model
```

## Common Workflows

### 1. First-Time Setup (Config-Based)
```bash
# 1. Setup environment
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# 2. Edit configuration (optional)
nano config.yaml

# 3. Run complete pipeline
python3 train_with_config.py

# 4. Test on your files
python3 analyze_my_files.py
```

### 2. First-Time Setup (Manual)
```bash
# 1. Setup environment
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# 2. Collect data
python3 collect_training_data.py

# 3. Train model
python3 train_with_collected_data.py

# 4. Test on your files
python3 analyze_my_files.py
```

### 3. Retrain Model (Config-Based)
```bash
# 1. Edit configuration
nano config.yaml

# 2. Run complete pipeline
python3 train_with_config.py

# 3. Test
python3 analyze_my_files.py
```

### 4. Retrain Model (Manual)
```bash
# 1. Collect new data
python3 collect_training_data.py

# 2. Retrain
python3 train_with_collected_data.py

# 3. Test
python3 analyze_my_files.py
```

### 5. Data Collection Only
```bash
# Using config.yaml
python3 collect_data_with_config.py

# Or manual collection
python3 collect_training_data.py
```

### 6. API Usage
```bash
# 1. Start server
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000

# 2. Test endpoint
curl -X POST "http://127.0.0.1:8000/v1/detect/text" \
  -H "Content-Type: application/json" \
  -d '{"text": "Sample text to analyze"}'
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Solution: Activate virtual environment
source .venv/bin/activate
```

#### Model Not Found
```bash
# Solution: Train model first
python3 train_with_collected_data.py
```

#### API Connection Refused
```bash
# Solution: Start API server
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

#### Insufficient Data
```bash
# Solution: Increase collection parameters
# Edit config.yaml or use command line options
```

## Performance Tips

### Data Collection
- Use RSS feeds for reliable article URLs
- Set appropriate `max_pages` per site
- Filter by minimum word count (400+)

### Training
- Ensure balanced human/AI dataset
- Use cross-validation for reliable estimates
- Monitor feature importance

### Testing
- Test on diverse content types
- Monitor confidence distributions
- Validate on known human/AI samples

## Model Output Format

```python
{
    'probability_ai': 0.74,        # AI probability (0.0-1.0)
    'confidence': 0.95,            # Confidence (0.3-0.95)
    'prediction': 'AI',            # 'AI' or 'Human'
    'explanations': [              # Feature-based explanations
        'Low vocabulary diversity',
        'High word repetition'
    ],
    'features': {                  # Extracted features
        'word_count': 236,
        'type_token_ratio': 0.585,
        # ... more features
    },
    'method': 'ml_classifier'      # Detection method
}
```

## API Endpoints

### POST /v1/detect/text
**Input**:
```json
{
    "text": "Text to analyze",
    "file_path": "optional/path/to/file.txt"
}
```

**Output**:
```json
{
    "probability_ai": 0.74,
    "confidence": 0.95,
    "prediction": "AI",
    "explanations": ["Low vocabulary diversity"],
    "method": "ml_classifier"
}
```

## Monitoring

### Key Metrics
- **Accuracy**: Overall classification performance
- **Confidence Distribution**: Spread of confidence scores
- **Response Time**: API endpoint performance
- **Feature Importance**: Which features drive predictions

### Logs
- Check console output for training progress
- Monitor API server logs for errors
- Review data collection reports

## Support

### Documentation
- `docs/TRAINING_WORKFLOW.md`: Complete workflow guide
- `docs/MODEL_ARCHITECTURE.md`: Technical model details
- `README.md`: Project overview

### Configuration
- `config.yaml`: Main configuration file
- `requirements.txt`: Python dependencies

### Testing
- `analyze_my_files.py`: User testing interface
- `test_ai_detection.py`: Unit tests
- `test_diverse_content.py`: Performance tests
