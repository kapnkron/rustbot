use crate::utils::error::Result;
use crate::trading::{MarketData, TradingSignal};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confusion_matrix: ConfusionMatrix,
    pub roc_auc: f64,
    pub mse: f64,
    pub mae: f64,
    pub r2_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    pub true_positives: usize,
    pub true_negatives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
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
            metrics: ModelMetrics {
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                confusion_matrix: ConfusionMatrix {
                    true_positives: 0,
                    true_negatives: 0,
                    false_positives: 0,
                    false_negatives: 0,
                },
                roc_auc: 0.0,
                mse: 0.0,
                mae: 0.0,
                r2_score: 0.0,
            },
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

        let mut confusion_matrix = ConfusionMatrix {
            true_positives: 0,
            true_negatives: 0,
            false_positives: 0,
            false_negatives: 0,
        };

        let mut predicted_values = Vec::new();
        let mut actual_values = Vec::new();
        let mut squared_errors = Vec::new();
        let mut absolute_errors = Vec::new();

        for ((_, buy_prob, sell_prob), (_, price_change)) in 
            self.predictions.iter().zip(self.actual_moves.iter()) 
        {
            let predicted_move = if buy_prob > sell_prob { 1.0 } else { -1.0 };
            let actual_move = if *price_change > 0.0 { 1.0 } else { -1.0 };

            // Update confusion matrix
            if predicted_move > 0.0 && actual_move > 0.0 {
                confusion_matrix.true_positives += 1;
            } else if predicted_move < 0.0 && actual_move < 0.0 {
                confusion_matrix.true_negatives += 1;
            } else if predicted_move > 0.0 && actual_move < 0.0 {
                confusion_matrix.false_positives += 1;
            } else {
                confusion_matrix.false_negatives += 1;
            }

            // Collect values for regression metrics
            predicted_values.push(predicted_move);
            actual_values.push(actual_move);
            squared_errors.push((predicted_move - actual_move).powi(2));
            absolute_errors.push((predicted_move - actual_move).abs());
        }

        // Calculate classification metrics
        let total = confusion_matrix.true_positives + confusion_matrix.true_negatives +
                   confusion_matrix.false_positives + confusion_matrix.false_negatives;
        
        self.metrics.accuracy = (confusion_matrix.true_positives + confusion_matrix.true_negatives) as f64 / total as f64;
        
        let precision_denominator = confusion_matrix.true_positives + confusion_matrix.false_positives;
        self.metrics.precision = if precision_denominator > 0 {
            confusion_matrix.true_positives as f64 / precision_denominator as f64
        } else {
            0.0
        };

        let recall_denominator = confusion_matrix.true_positives + confusion_matrix.false_negatives;
        self.metrics.recall = if recall_denominator > 0 {
            confusion_matrix.true_positives as f64 / recall_denominator as f64
        } else {
            0.0
        };

        self.metrics.f1_score = if self.metrics.precision + self.metrics.recall > 0.0 {
            2.0 * (self.metrics.precision * self.metrics.recall) / (self.metrics.precision + self.metrics.recall)
        } else {
            0.0
        };

        // Calculate regression metrics
        self.metrics.mse = squared_errors.iter().sum::<f64>() / squared_errors.len() as f64;
        self.metrics.mae = absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64;

        // Calculate R² score
        let mean_actual = actual_values.iter().sum::<f64>() / actual_values.len() as f64;
        let total_sum_squares = actual_values.iter()
            .map(|&x| (x - mean_actual).powi(2))
            .sum::<f64>();
        let residual_sum_squares = squared_errors.iter().sum::<f64>();
        
        self.metrics.r2_score = if total_sum_squares > 0.0 {
            1.0 - (residual_sum_squares / total_sum_squares)
        } else {
            0.0
        };

        // Calculate ROC AUC
        self.metrics.roc_auc = self.calculate_roc_auc(&predicted_values, &actual_values);

        self.metrics.confusion_matrix = confusion_matrix;

        Ok(())
    }

    fn calculate_roc_auc(&self, predicted: &[f64], actual: &[f64]) -> f64 {
        let mut points = predicted.iter()
            .zip(actual.iter())
            .map(|(p, a)| (*p, *a))
            .collect::<Vec<_>>();
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut auc = 0.0;
        let mut prev_tpr = 0.0;
        let mut prev_fpr = 0.0;

        let total_positives = actual.iter().filter(|&&x| x > 0.5).count() as f64;
        let total_negatives = actual.iter().filter(|&&x| x <= 0.5).count() as f64;

        for (_, actual) in points {
            let tpr = if actual > 0.5 { prev_tpr + 1.0 / total_positives } else { prev_tpr };
            let fpr = if actual <= 0.5 { prev_fpr + 1.0 / total_negatives } else { prev_fpr };

            auc += (tpr + prev_tpr) * (fpr - prev_fpr) / 2.0;

            prev_tpr = tpr;
            prev_fpr = fpr;
        }

        auc
    }

    pub fn get_metrics(&self) -> &ModelMetrics {
        &self.metrics
    }

    pub fn calculate_metrics(&self, predicted: &[f64], actual: &[f64]) -> ModelMetrics {
        let mut squared_errors = Vec::new();
        let mut absolute_errors = Vec::new();

        for (pred, act) in predicted.iter().zip(actual.iter()) {
            let error = pred - act;
            squared_errors.push(error * error);
            absolute_errors.push(error.abs());
        }

        let mse = squared_errors.iter().sum::<f64>() / squared_errors.len() as f64;
        let mae = absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64;

        let accuracy = self.calculate_accuracy(predicted, actual);
        let precision = self.calculate_precision(predicted, actual);
        let recall = self.calculate_recall(predicted, actual);
        let f1_score = self.calculate_f1_score(precision, recall);
        let confusion_matrix = self.calculate_confusion_matrix(predicted, actual);
        let roc_auc = self.calculate_roc_auc(predicted, actual);
        let r2_score = self.calculate_r2_score(predicted, actual);

        ModelMetrics {
            accuracy,
            precision,
            recall,
            f1_score,
            confusion_matrix,
            roc_auc,
            mse,
            mae,
            r2_score,
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
        assert!(metrics.accuracy > 0.0);
        assert!(metrics.precision > 0.0);
        assert!(metrics.recall > 0.0);
        assert!(metrics.f1_score > 0.0);
        assert!(metrics.roc_auc > 0.0);
        assert!(metrics.mse > 0.0);
        assert!(metrics.mae > 0.0);
        assert!(metrics.r2_score <= 1.0);
    }
} 