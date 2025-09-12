# Using config.yaml for Training

## Overview

The AI text detection system now supports configuration-based training using `config.yaml`. This allows you to customize all training parameters without modifying code.

## Quick Start

### 1. Edit Configuration
```bash
# Edit the configuration file
nano config.yaml
```

### 2. Run Complete Pipeline
```bash
# Complete pipeline (data collection + training + testing)
python3 train_with_config.py
```

### 3. Or Run Individual Steps
```bash
# Data collection only
python3 collect_data_with_config.py

# Training only (if data already collected)
python3 train_with_collected_data.py
```

## Configuration Parameters

### Training Data Collection
```yaml
training:
  # Human content collection
  human_articles_per_site: 50      # Articles to scrape per website
  max_sites: 10                    # Maximum websites to scrape
  min_human_words: 400             # Minimum words for human articles
  
  # AI content generation
  ai_samples: 500                  # Number of AI samples to generate
  target_words_per_sample: 700     # Target words per AI sample
  
  # Quality control
  filter_duplicates: true          # Remove duplicate content
  validate_content: true           # Validate content quality
```

### Model Configuration
```yaml
model:
  # Model type: "ensemble", "random_forest", or "logistic_regression"
  type: "ensemble"
  
  # Training parameters
  test_size: 0.2                   # Test split ratio (20% for testing)
  cv_folds: 5                      # Cross-validation folds
  
  # Feature engineering
  max_features: 3000               # Maximum TF-IDF features
  ngram_range: [1, 2]             # N-gram range for TF-IDF
  
  # Ensemble parameters (if using ensemble)
  random_forest_estimators: 200    # Number of trees in Random Forest
  random_forest_max_depth: 10      # Maximum depth of trees
  logistic_regression_c: 1.0       # Regularization strength
```

### Files to Analyze
```yaml
files_to_analyze:
  - "example.txt"
  - "example1.txt"
  - "example2.txt"
  - "sample_ai_text.txt"
```

## Example Configurations

### Quick Training (Small Dataset)
```yaml
training:
  human_articles_per_site: 20
  ai_samples: 100
  target_words_per_sample: 500
  max_sites: 5

model:
  type: "random_forest"
  test_size: 0.2
```

### Comprehensive Training (Large Dataset)
```yaml
training:
  human_articles_per_site: 100
  ai_samples: 1000
  target_words_per_sample: 1000
  max_sites: 20

model:
  type: "ensemble"
  test_size: 0.2
  cv_folds: 10
```

### Production Training
```yaml
training:
  human_articles_per_site: 200
  ai_samples: 2000
  target_words_per_sample: 800
  max_sites: 30
  min_human_words: 600

model:
  type: "ensemble"
  test_size: 0.15
  cv_folds: 5
  max_features: 5000
```

## Workflow Examples

### First-Time Setup
```bash
# 1. Edit configuration
nano config.yaml

# 2. Run complete pipeline
python3 train_with_config.py

# 3. Test the model
python3 analyze_my_files.py
```

### Retrain with New Parameters
```bash
# 1. Modify configuration
nano config.yaml

# 2. Run complete pipeline
python3 train_with_config.py
```

### Data Collection Only
```bash
# 1. Edit training parameters
nano config.yaml

# 2. Collect data only
python3 collect_data_with_config.py

# 3. Train later
python3 train_with_collected_data.py
```

### Custom Training
```bash
# 1. Edit model parameters
nano config.yaml

# 2. Train with existing data
python3 train_with_collected_data.py
```

## Configuration Tips

### Data Collection Optimization
- **Increase `human_articles_per_site`** for more diverse human content
- **Increase `ai_samples`** for better AI representation
- **Adjust `target_words_per_sample`** based on your use case
- **Use `max_sites`** to control scraping time

### Model Performance
- **Use `ensemble`** for best overall performance
- **Use `random_forest`** for faster training
- **Use `logistic_regression`** for interpretability
- **Adjust `test_size`** based on dataset size

### Quality Control
- **Set `min_human_words`** to filter out short articles
- **Enable `filter_duplicates`** to remove redundant content
- **Enable `validate_content`** for quality checks

## Troubleshooting

### Common Issues

#### Insufficient Data
```yaml
# Increase collection parameters
training:
  human_articles_per_site: 100  # More articles per site
  ai_samples: 1000              # More AI samples
  max_sites: 20                 # More websites
```

#### Training Too Slow
```yaml
# Use simpler model
model:
  type: "random_forest"         # Faster than ensemble
  max_features: 1000           # Fewer features
```

#### Low Accuracy
```yaml
# Improve data quality
training:
  min_human_words: 600         # Longer articles
  target_words_per_sample: 800 # Longer AI samples

model:
  type: "ensemble"             # Better performance
  cv_folds: 10                # More validation
```

#### Memory Issues
```yaml
# Reduce memory usage
model:
  max_features: 1000          # Fewer features
  random_forest_estimators: 100  # Fewer trees
```

## Advanced Configuration

### Custom Model Parameters
```yaml
model:
  type: "ensemble"
  random_forest_estimators: 300
  random_forest_max_depth: 15
  logistic_regression_c: 0.5
  max_features: 5000
  ngram_range: [1, 3]
```

### Quality Control
```yaml
training:
  filter_duplicates: true
  validate_content: true
  min_human_words: 500
  max_human_words: 5000
  target_words_per_sample: 700
  word_variance: 0.2
```

### Output Configuration
```yaml
output:
  model_file: "my_custom_model.pkl"
  training_data_dir: "my_training_data"
  save_reports: true
  save_feature_importance: true
```

## Best Practices

1. **Start Small**: Begin with smaller datasets to test configuration
2. **Iterate**: Gradually increase parameters based on results
3. **Monitor**: Check training logs for issues
4. **Validate**: Test on diverse content types
5. **Document**: Keep track of successful configurations

## Example Complete Workflow

```bash
# 1. Setup environment
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

# 2. Create custom configuration
cat > config.yaml << EOF
training:
  human_articles_per_site: 50
  ai_samples: 500
  target_words_per_sample: 700
  max_sites: 10
  min_human_words: 400

model:
  type: "ensemble"
  test_size: 0.2
  cv_folds: 5

files_to_analyze:
  - "example.txt"
  - "example1.txt"
  - "sample_ai_text.txt"
EOF

# 3. Run complete training
python3 train_with_config.py

# 4. Test results
python3 analyze_my_files.py
```
