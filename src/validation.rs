//! Validation utilities for improving user experience with edge cases
//!
//! This module provides validation functions to help users understand
//! when they're hitting edge cases or limitations of the algorithm.

use crate::error::{AnomalyGridError, AnomalyGridResult};
use std::collections::HashMap;

/// Validates training data quality and provides warnings for edge cases
pub fn validate_training_data_quality(sequence: &[String]) -> Vec<String> {
    let mut warnings = Vec::new();

    if sequence.is_empty() {
        return warnings;
    }

    // Check for monotonous data (all same element)
    let unique_elements: std::collections::HashSet<&String> = sequence.iter().collect();
    if unique_elements.len() == 1 {
        warnings.push(format!(
            "Training data contains only one unique element ('{}') - this may severely limit anomaly detection capability",
            sequence[0]
        ));
    }

    // Check for very low diversity
    let diversity_ratio = unique_elements.len() as f64 / sequence.len() as f64;
    if diversity_ratio < 0.01 && unique_elements.len() > 1 {
        warnings.push(format!(
            "Training data has very low diversity ({:.1}% unique elements) - consider using more varied training data",
            diversity_ratio * 100.0
        ));
    }

    // Check for extremely short sequences
    if sequence.len() < 10 {
        warnings.push(format!(
            "Training data is very short ({} elements) - consider using more training data for better model quality",
            sequence.len()
        ));
    }

    warnings
}

/// Validates detection sequence length against max_order
pub fn validate_detection_sequence(sequence: &[String], max_order: usize) -> AnomalyGridResult<()> {
    if sequence.is_empty() {
        return Err(AnomalyGridError::invalid_configuration(
            "detection_sequence",
            "empty",
            "non-empty sequence",
        ));
    }

    if sequence.len() <= max_order {
        return Err(AnomalyGridError::sequence_too_short(
            max_order + 1,
            sequence.len(),
            &format!("anomaly detection with max_order {max_order}"),
        ));
    }

    Ok(())
}

/// Analyzes training data characteristics for diagnostic purposes
pub fn analyze_training_data_characteristics(sequence: &[String]) -> TrainingDataAnalysis {
    let mut element_counts = HashMap::new();
    for element in sequence {
        *element_counts.entry(element.clone()).or_insert(0) += 1;
    }

    let unique_elements = element_counts.len();
    let total_elements = sequence.len();
    let diversity_ratio = unique_elements as f64 / total_elements as f64;

    // Calculate entropy as a measure of diversity
    let entropy = if total_elements > 0 {
        element_counts
            .values()
            .map(|&count| {
                let p = count as f64 / total_elements as f64;
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum()
    } else {
        0.0
    };

    let max_possible_entropy = (unique_elements as f64).log2();
    let normalized_entropy = if max_possible_entropy > 0.0 {
        entropy / max_possible_entropy
    } else {
        0.0
    };

    TrainingDataAnalysis {
        total_elements,
        unique_elements,
        diversity_ratio,
        entropy,
        normalized_entropy,
        most_common_element: element_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(element, &count)| (element.clone(), count)),
    }
}

/// Analysis results for training data characteristics
#[derive(Debug, Clone)]
pub struct TrainingDataAnalysis {
    pub total_elements: usize,
    pub unique_elements: usize,
    pub diversity_ratio: f64,
    pub entropy: f64,
    pub normalized_entropy: f64,
    pub most_common_element: Option<(String, usize)>,
}

impl TrainingDataAnalysis {
    /// Provides a quality assessment of the training data
    pub fn quality_assessment(&self) -> &'static str {
        if self.unique_elements == 1 {
            "Poor - Monotonous data"
        } else if self.normalized_entropy > 0.8 {
            "Excellent - High diversity"
        } else if self.normalized_entropy > 0.6 {
            "Good - Adequate diversity"
        } else if self.normalized_entropy > 0.4 {
            "Fair - Limited diversity"
        } else {
            "Poor - Very low diversity"
        }
    }

    /// Provides recommendations for improving training data
    pub fn recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        if self.unique_elements == 1 {
            recommendations.push("Add more diverse elements to training data".to_string());
        } else if self.normalized_entropy < 0.5 {
            recommendations.push("Consider adding more variety to training sequences".to_string());
        }

        if self.total_elements < 100 {
            recommendations
                .push("Consider using more training data for better model quality".to_string());
        }

        if let Some((element, count)) = &self.most_common_element {
            let frequency = *count as f64 / self.total_elements as f64;
            if frequency > 0.8 {
                recommendations.push(format!(
                    "Element '{}' appears {:.1}% of the time - consider balancing the data",
                    element,
                    frequency * 100.0
                ));
            }
        }

        recommendations
    }
}
