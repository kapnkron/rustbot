# Machine Learning Components Usage Guide

## Overview
This guide explains how to use the machine learning components of the trading bot. The ML system consists of several modules that work together to provide market analysis and trading signals.

## Components

### 1. Feature Extraction
The `FeatureExtractor` class handles data preprocessing and feature engineering.

```rust
use trading_system::ml::features::FeatureExtractor;

// Initialize feature extractor with a window size
let mut extractor = FeatureExtractor::new(30);

// Update with market data
extractor.update(&market_data);

// Extract features
let features = extractor.extract_features(&market_data)?;
```

### 2. Model Training
The `ModelTrainer` class handles model training and optimization.

```rust
use trading_system::ml::training::ModelTrainer;

// Initialize model trainer with configuration
let mut trainer = ModelTrainer::new(config)?;

// Train model
trainer.train(&training_data, &validation_data, Some(5)).await?;

// Save trained model
trainer.save(Path::new("model.pt"))?;
```

### 3. Model Inference
The `ModelInference` class handles real-time predictions.

```rust
use trading_system::ml::inference::ModelInference;

// Initialize model inference
let mut inference = ModelInference::new(config).await?;

// Load trained model
inference.load(Path::new("model.pt")).await?;

// Make prediction
let prediction = inference.predict(&market_data).await?;
println!("Prediction: {}, Confidence: {}", prediction.value, prediction.confidence);

// Make batch predictions
let predictions = inference.predict_batch(&market_data_batch).await?;
```

### 4. Model Evaluation
The `ModelEvaluator` class handles model performance evaluation.

```rust
use trading_system::ml::evaluation::{ModelEvaluator, ModelMetrics};

// Initialize model evaluator
let mut evaluator = ModelEvaluator::new(100);

// Record predictions and actual values
evaluator.record_prediction(timestamp, buy_prob, sell_prob);
evaluator.record_actual_move(timestamp, price_change);

// Get metrics
let metrics = evaluator.get_metrics();
println!("Accuracy: {}", metrics.accuracy);
println!("Precision: {}", metrics.precision);
println!("Recall: {}", metrics.recall);
println!("F1 Score: {}", metrics.f1_score);
```

## Configuration

The ML system is configured through the `MLConfig` struct:

```rust
let config = MLConfig {
    input_size: 9,
    hidden_size: 20,
    output_size: 2,
    learning_rate: 0.001,
    model_path: "model.pt".to_string(),
    confidence_threshold: 0.7,
    training_batch_size: 32,
    training_epochs: 100,
    window_size: 10,
    min_data_points: 100,
    validation_split: 0.2,
    early_stopping_patience: 5,
    save_best_model: true,
};
```

## Best Practices

1. **Data Preparation**
   - Ensure sufficient historical data
   - Normalize features appropriately
   - Handle missing data
   - Split data into training and validation sets

2. **Model Training**
   - Use appropriate batch size
   - Monitor training metrics
   - Implement early stopping
   - Save model checkpoints
   - Use cross-validation

3. **Model Evaluation**
   - Use multiple metrics
   - Track performance over time
   - Set up alerts for degradation
   - Regular model validation

4. **Model Deployment**
   - Test thoroughly before deployment
   - Monitor prediction latency
   - Implement fallback strategies
   - Version control models
   - Document model architecture

## Error Handling

The ML system uses the `Result` type for error handling. Common errors include:
- Insufficient data
- Invalid input features
- Model loading failures
- Training convergence issues
- Prediction errors

## Testing

Run the ML system tests with:
```bash
cargo test --package trading_system --lib ml
```

## Monitoring

The ML system includes monitoring capabilities:
- Prediction latency
- Model confidence
- Training metrics
- Validation metrics
- Error rates

## Next Steps

1. Implement automated model retraining
2. Add feature importance tracking
3. Implement model explainability
4. Add A/B testing framework
5. Implement model versioning 