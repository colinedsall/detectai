# Configuration Guide (config.yaml)

The `config.yaml` file controls every aspect of the DetectAI system. You can edit this file manually or use the built-in **Config Tab** in the GUI.

## Structure Overview

### 1. AI Generation (`ai_generation`)
Controls how synthetic data is created.

```yaml
ai_generation:
  target_words_per_sample: 700   # Approximate length of generated articles
  ai_topics:                     # List of topics for the LLM to write about
    - "machine learning algorithms"
    - "climate change policy"
    - "modern art history"
    # ... add as many as you like
```

### 2. Training Settings (`training`)
General settings for data collection and model architecture.

```yaml
training:
  human_articles_per_site: 50    # How many articles to scrape per human site
  max_sites: 100                 # Limit for scraping session
  min_human_words: 400           # Ignore short snippets
  
  # Neural Network Hyperparameters (Tunable in GUI)
  neural_network:
    epochs: 100                  # Training iterations
    learning_rate: 0.001         # Optimizer step size
    batch_size: 32               # Samples per batch
    hidden_size: 64              # Neurons in hidden layers
```

### 3. Human Data Sources (`sites`)
A list of verified news/blog sources to scrape.

```yaml
sites:
  - name: bbc_news
    url: https://www.bbc.com/news
    rss: https://feeds.bbci.co.uk/news/rss.xml
    description: "BBC News"
    
  - name: techcrunch
    url: https://techcrunch.com
    rss: https://techcrunch.com/feed/
```

### 4. API & System (`api`)
Settings for the backend server.

```yaml
api:
  host: 127.0.0.1
  port: 8000
  reload: true
```

## Best Practices

1. **Balance Data**: Try to have roughly equal numbers of Human and AI samples.
   - Example: If you scrape 10 sites x 50 articles (500 Human), generate ~500 AI samples.
2. **Diverse Topics**: Add specific, niche topics to `ai_topics` to prevent the model from learning "AI writes about X, Humans write about Y".
3. **Hyperparameters**:
   - If validation loss isn't decreasing: Lower `learning_rate` (e.g., 0.0001).
   - If training is unstable: Increase `batch_size`.
