# AI Text Detection Documentation

This folder contains comprehensive documentation for the AI text detection system.

## Documentation Structure

### 📋 [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)
**Complete training workflow guide**
- Detailed workflow diagram (text-based)
- Step-by-step process from data collection to deployment
- Configuration parameters and best practices
- Troubleshooting guide and performance monitoring
- File structure and organization

### 🏗️ [MODEL_ARCHITECTURE.md](../MODEL_ARCHITECTURE.md)
**Technical model architecture details**
- Ensemble classifier design (Random Forest + Logistic Regression)
- Feature engineering (TF-IDF + 12 linguistic features)
- Dynamic confidence estimation algorithm
- Probability calibration and cross-validation
- Performance metrics and limitations

### ⚡ [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
**Essential commands and workflows**
- Setup and installation commands
- Common workflows (first-time setup, retraining, API usage)
- Configuration examples
- Troubleshooting common issues
- API endpoint documentation

### ⚙️ [CONFIG_USAGE.md](CONFIG_USAGE.md)
**Configuration-based training guide**
- Complete guide to using config.yaml
- Parameter explanations and examples
- Workflow templates for different use cases
- Troubleshooting configuration issues
- Best practices for parameter tuning

## Quick Start

1. **Setup**: Follow the [Quick Reference Guide](QUICK_REFERENCE.md#setup)
2. **Workflow**: Understand the complete process in [Training Workflow](TRAINING_WORKFLOW.md)
3. **Architecture**: Learn about the model design in [Model Architecture](../MODEL_ARCHITECTURE.md)

## Key Concepts

### Data Collection
- **Human Content**: Web scraping from RSS feeds (BBC, NPR, Reuters, etc.)
- **AI Content**: Generated samples with diverse topics and styles
- **Quality Control**: Minimum word count, duplicate removal, content filtering

### Model Training
- **Ensemble Approach**: Voting classifier with Random Forest and Logistic Regression
- **Feature Engineering**: TF-IDF + linguistic features for comprehensive analysis
- **Calibration**: Isotonic calibration for reliable probability estimates
- **Cross-Validation**: 5-fold CV for robust performance estimation

### Confidence Estimation
- **Dynamic Confidence**: Multi-factor calculation (not fixed at 80%)
- **Factors**: Probability distance, text length, linguistic features
- **Range**: 30-95% based on prediction certainty

### Deployment
- **API Integration**: FastAPI endpoint for easy integration
- **Model Serialization**: Pickle format with all components
- **Testing Framework**: Comprehensive testing and validation

## File Organization

```
detectai/
├── docs/                          # This documentation folder
│   ├── README.md                  # This file
│   ├── TRAINING_WORKFLOW.md       # Complete workflow guide
│   └── QUICK_REFERENCE.md         # Essential commands
├── MODEL_ARCHITECTURE.md          # Technical model details
├── README.md                      # Project overview
└── QUICKSTART.md                  # 5-minute setup guide
```

## Getting Help

### For New Users
1. Start with [QUICKSTART.md](../QUICKSTART.md) for 5-minute setup
2. Follow [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for essential commands
3. Understand the process in [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)

### For Developers
1. Review [MODEL_ARCHITECTURE.md](../MODEL_ARCHITECTURE.md) for technical details
2. Study the workflow in [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)
3. Use [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common tasks

### For Troubleshooting
- Check [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md#troubleshooting)
- Review [QUICK_REFERENCE.md](QUICK_REFERENCE.md#troubleshooting)
- Examine error messages and logs

## Contributing

When adding new features or making changes:

1. **Update Documentation**: Modify relevant documentation files
2. **Add Examples**: Include usage examples in quick reference
3. **Update Workflow**: Reflect any process changes in workflow guide
4. **Test Instructions**: Ensure all commands work as documented

## Documentation Standards

- **Code Blocks**: Use syntax highlighting for all code examples
- **File Paths**: Use relative paths from project root
- **Commands**: Include expected output where helpful
- **Diagrams**: Use text-based diagrams for portability
- **Cross-References**: Link between related documentation sections
