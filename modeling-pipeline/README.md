# Modeling Pipeline - Fertility Reconstruction

This pipeline handles machine learning model development, training, and evaluation for fertility reconstruction.

## Overview

The modeling pipeline provides tools for:
1. **Data Analysis** and exploration
2. **Feature Engineering** and preprocessing  
3. **Model Development** and training
4. **Evaluation** and comparison
5. **Prediction** and inference

## Directory Structure

```
modeling-pipeline/
├── notebooks/         # Jupyter notebooks for analysis and experimentation
├── scripts/           # Production modeling scripts
├── models/           # Trained model artifacts
├── experiments/      # Experiment tracking and results
└── README.md        # This file
```

## Notebooks

### Current Notebooks
- `notebooks/dataviz.ipynb` - Data visualization and exploration
- `notebooks/ml_modeling.ipynb` - Machine learning model development

### Recommended Workflow
1. **Data Exploration** (`dataviz.ipynb`)
   - Load and examine the comprehensive dataset
   - Visualize spatial and temporal patterns
   - Identify data quality issues

2. **Feature Engineering** (to be created)
   - Transform raw environmental data
   - Create derived features
   - Handle missing values and outliers

3. **Model Development** (`ml_modeling.ipynb`)
   - Implement various ML algorithms
   - Hyperparameter tuning
   - Cross-validation

4. **Evaluation** (to be created)
   - Model performance metrics
   - Spatial and temporal validation
   - Error analysis

## Data Input

The modeling pipeline expects data from the data pipeline:
- **Main dataset**: `../data/processed/comprehensive_pixels_dataset.csv`
- **Format**: Each row = one pixel × one year
- **Columns**: pixel_id, year, + 496 environmental bands

## Planned Development

### Scripts (to be created)
- `scripts/feature_engineering.py` - Automated feature creation
- `scripts/train_models.py` - Model training pipeline
- `scripts/evaluate_models.py` - Model evaluation and comparison
- `scripts/predict.py` - Inference on new data

### Models Directory
Will contain:
- Trained model files (.pkl, .joblib, .h5)
- Model metadata and configuration
- Feature importance and analysis

### Experiments Directory
Will track:
- Experiment configurations
- Training logs and metrics
- Model comparison results
- Hyperparameter optimization results

## Getting Started

### 1. Environment Setup
```bash
# Use the same environment as data pipeline
conda activate ee

# Or create modeling-specific environment
# conda env create -f modeling_env.yml  # to be created
```

### 2. Data Exploration
```bash
jupyter notebook notebooks/dataviz.ipynb
```

### 3. Model Development
```bash
jupyter notebook notebooks/ml_modeling.ipynb
```

## Expected Workflow

1. **Load Data**: Import the comprehensive pixels dataset
2. **Explore**: Understand data patterns and relationships
3. **Engineer Features**: Create meaningful predictors
4. **Split Data**: Temporal/spatial train-test splits
5. **Train Models**: Implement various algorithms
6. **Evaluate**: Compare model performance
7. **Deploy**: Create inference pipeline

## Dependencies

Will include:
- scikit-learn for machine learning
- pandas for data manipulation
- matplotlib/seaborn for visualization
- jupyter for interactive development
- optionally: pytorch/tensorflow for deep learning

## Notes

- Models should account for spatial and temporal dependencies
- Consider both pixel-level and regional-level predictions
- Validate models on holdout time periods
- Document all experiments for reproducibility
