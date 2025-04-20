use crate::api::MarketData;
use crate::trading::TradingSignal;
use crate::error::Result;
use log::{info, error};
use tch::{Device, Tensor};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    pub true_positives: usize,
    pub true_negatives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

#[derive(Debug, Clone)]
pub struct ModelMetrics {
    pub mse: f64,
    pub mae: f64,
    pub rmse: f64,
}

impl ModelMetrics {
    pub fn new() -> Self {
        Self {
            mse: 0.0,
            mae: 0.0,
            rmse: 0.0,
        }
    }
}

pub struct ModelEvaluator {
    window_size: usize,
    predictions: VecDeque<(DateTime<Utc>, f64, f64)>, // (timestamp, buy_prob, sell_prob)
    actual_moves: VecDeque<(DateTime<Utc>, f64)>,     // (timestamp, price_change)
    metrics: ModelMetrics,
}

impl ModelEvaluator {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            predictions: VecDeque::new(),
            actual_moves: VecDeque::new(),
            metrics: ModelMetrics::new(),
        }
    }

    pub fn record_prediction(&mut self, timestamp: DateTime<Utc>, buy_prob: f64, sell_prob: f64) {
        self.predictions.push_back((timestamp, buy_prob, sell_prob));
        if self.predictions.len() > self.window_size {
            self.predictions.pop_front();
        }
    }

    pub fn record_actual_move(&mut self, timestamp: DateTime<Utc>, price_change: f64) {
        self.actual_moves.push_back((timestamp, price_change));
        if self.actual_moves.len() > self.window_size {
            self.actual_moves.pop_front();
        }
    }

    pub fn update_metrics(&mut self) -> Result<()> {
        if self.predictions.is_empty() || self.actual_moves.is_empty() {
            return Ok(());
        }

        let mut squared_errors = Vec::new();
        let mut absolute_errors = Vec::new();

        for ((_, buy_prob, sell_prob), (_, price_change)) in 
            self.predictions.iter().zip(self.actual_moves.iter()) 
        {
            let predicted_move = if buy_prob > sell_prob { 1.0 } else { -1.0 };
            let actual_move = if *price_change > 0.0 { 1.0 } else { -1.0 };

            let diff: f64 = predicted_move - actual_move;
            squared_errors.push(diff.powi(2));
            absolute_errors.push(diff.abs());
        }

        let mse = squared_errors.iter().sum::<f64>() / squared_errors.len() as f64;
        let mae = absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64;
        let rmse = mse.sqrt();

        self.metrics.mse = mse;
        self.metrics.mae = mae;
        self.metrics.rmse = rmse;

        Ok(())
    }

    pub fn get_metrics(&self) -> ModelMetrics {
        self.metrics.clone()
    }

    pub fn calculate_metrics(&mut self) -> ModelMetrics {
        let mut squared_errors = Vec::new();
        let mut absolute_errors = Vec::new();

        for (predicted_move, actual_move) in self.predictions.iter() {
            let diff: f64 = predicted_move - actual_move;
            squared_errors.push(diff.powi(2));
            absolute_errors.push(diff.abs());
        }

        let mse = squared_errors.iter().sum::<f64>() / squared_errors.len() as f64;
        let mae = absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64;
        let rmse = mse.sqrt();

        ModelMetrics {
            mse,
            mae,
            rmse,
        }
    }

    fn calculate_accuracy(&self, predicted: &[f64], actual: &[f64]) -> f64 {
        let mut correct = 0;
        let total = predicted.len();

        for (pred, act) in predicted.iter().zip(actual.iter()) {
            if (pred > &0.5 && act > &0.5) || (pred <= &0.5 && act <= &0.5) {
                correct += 1;
            }
        }

        correct as f64 / total as f64
    }

    fn calculate_precision(&self, predicted: &[f64], actual: &[f64]) -> f64 {
        let mut true_positives = 0;
        let mut false_positives = 0;

        for (pred, act) in predicted.iter().zip(actual.iter()) {
            if pred > &0.5 {
                if act > &0.5 {
                    true_positives += 1;
                } else {
                    false_positives += 1;
                }
            }
        }

        if true_positives + false_positives == 0 {
            0.0
        } else {
            true_positives as f64 / (true_positives + false_positives) as f64
        }
    }

    fn calculate_recall(&self, predicted: &[f64], actual: &[f64]) -> f64 {
        let mut true_positives = 0;
        let mut false_negatives = 0;

        for (pred, act) in predicted.iter().zip(actual.iter()) {
            if act > &0.5 {
                if pred > &0.5 {
                    true_positives += 1;
                } else {
                    false_negatives += 1;
                }
            }
        }

        if true_positives + false_negatives == 0 {
            0.0
        } else {
            true_positives as f64 / (true_positives + false_negatives) as f64
        }
    }

    fn calculate_f1_score(&self, precision: f64, recall: f64) -> f64 {
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * (precision * recall) / (precision + recall)
        }
    }

    fn calculate_confusion_matrix(&self, predicted: &[f64], actual: &[f64]) -> ConfusionMatrix {
        let mut matrix = ConfusionMatrix {
            true_positives: 0,
            true_negatives: 0,
            false_positives: 0,
            false_negatives: 0,
        };

        for (pred, act) in predicted.iter().zip(actual.iter()) {
            if pred > &0.5 {
                if act > &0.5 {
                    matrix.true_positives += 1;
                } else {
                    matrix.false_positives += 1;
                }
            } else {
                if act > &0.5 {
                    matrix.false_negatives += 1;
                } else {
                    matrix.true_negatives += 1;
                }
            }
        }

        matrix
    }

    fn calculate_r2_score(&self, predicted: &[f64], actual: &[f64]) -> f64 {
        let mean = actual.iter().sum::<f64>() / actual.len() as f64;
        let total_sum_squares: f64 = actual.iter().map(|&x| (x - mean).powi(2)).sum();
        let residual_sum_squares: f64 = predicted.iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (a - p).powi(2))
            .sum();

        1.0 - (residual_sum_squares / total_sum_squares)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    #[test]
    fn test_model_evaluation() {
        let mut evaluator = ModelEvaluator::new(100);
        
        // Record some predictions and actual moves
        let timestamp = Utc::now();
        evaluator.record_prediction(timestamp, 0.8, 0.2); // Strong buy signal
        evaluator.record_actual_move(timestamp, 0.1);     // Price went up
        
        evaluator.record_prediction(timestamp, 0.3, 0.7); // Strong sell signal
        evaluator.record_actual_move(timestamp, -0.1);    // Price went down
        
        evaluator.record_prediction(timestamp, 0.6, 0.4); // Weak buy signal
        evaluator.record_actual_move(timestamp, -0.05);   // Price went down (false positive)
        
        evaluator.record_prediction(timestamp, 0.4, 0.6); // Weak sell signal
        evaluator.record_actual_move(timestamp, 0.05);    // Price went up (false negative)

        assert!(evaluator.update_metrics().is_ok());
        
        let metrics = evaluator.get_metrics();
        assert!(metrics.mse > 0.0);
        assert!(metrics.mae > 0.0);
        assert!(metrics.rmse > 0.0);
    }
} 