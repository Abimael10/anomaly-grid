//! Crate-internal training-data quality checks.
//!
//! Returns warnings rather than errors: monotonous or extremely short
//! input can still be trained on, but the caller may want to surface a
//! warning. [`crate::AnomalyDetector::training_warnings`] exposes the
//! result on the public API.

use std::collections::HashSet;

/// Diagnose obvious quality issues in a training sequence.
#[allow(dead_code)] // wired into the public API in a later commit
pub fn validate_training_data_quality(sequence: &[String]) -> Vec<String> {
    let mut warnings = Vec::new();
    if sequence.is_empty() {
        return warnings;
    }

    let unique_elements: HashSet<&String> = sequence.iter().collect();
    if unique_elements.len() == 1 {
        warnings.push(format!(
            "training data contains only one unique element ('{}') — anomaly detection will be degenerate",
            sequence[0]
        ));
    }

    let diversity_ratio = unique_elements.len() as f64 / sequence.len() as f64;
    if diversity_ratio < 0.01 && unique_elements.len() > 1 {
        warnings.push(format!(
            "training data has very low diversity ({:.1}% unique elements) — model quality will suffer",
            diversity_ratio * 100.0
        ));
    }

    if sequence.len() < 10 {
        warnings.push(format!(
            "training data is very short ({} elements) — model quality will suffer",
            sequence.len()
        ));
    }

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| (*x).to_string()).collect()
    }

    #[test]
    fn empty_input_produces_no_warnings() {
        assert!(validate_training_data_quality(&[]).is_empty());
    }

    #[test]
    fn monotonous_input_warns() {
        let warnings = validate_training_data_quality(&s(&["A", "A", "A", "A"]));
        assert!(warnings.iter().any(|w| w.contains("only one unique element")));
    }

    #[test]
    fn short_input_warns() {
        let warnings = validate_training_data_quality(&s(&["A", "B"]));
        assert!(warnings.iter().any(|w| w.contains("very short")));
    }
}
