//! Quick Start Example
//!
//! This example demonstrates the basic usage of the anomaly-grid library.
//! It shows how to train a detector and detect anomalies in sequences.

#![allow(clippy::uninlined_format_args)]
#![allow(clippy::useless_vec)]

use anomaly_grid::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Anomaly Grid - Quick Start Example");
    println!("Demonstrating accurate anomaly detection with real patterns\n");

    // Create detector with optimal configuration for this example
    let config = AnomalyGridConfig::default()
        .with_max_order(3)? // Good balance of context and performance
        .with_smoothing_alpha(1.0)? // Standard Laplace smoothing
        .with_weights(0.7, 0.3)?; // Emphasize likelihood over information

    let mut detector = AnomalyDetector::with_config(config)?;
    println!("✅ Created detector with order 3 and optimized configuration");

    // Generate realistic training data with clear patterns
    let normal_patterns = generate_realistic_training_data();
    println!(
        "📚 Generated {} training sequences with realistic patterns",
        normal_patterns.len()
    );

    // Train the detector
    let train_start = Instant::now();
    detector.train(&normal_patterns)?;
    let train_time = train_start.elapsed();

    // Get training metrics
    let metrics = detector.performance_metrics();
    println!("🎯 Training completed in {:?}", train_time);
    println!("   - Contexts learned: {}", metrics.context_count);
    println!(
        "   - Memory usage: {:.1} KB",
        metrics.estimated_memory_bytes as f64 / 1024.0
    );

    // Test different scenarios with proper threshold tuning
    println!("\n🔍 Testing Anomaly Detection Scenarios");

    let test_scenarios = vec![
        ("Normal Pattern", generate_normal_sequence(), 0.1),
        ("Slight Deviation", generate_slight_deviation(), 0.05),
        ("Clear Anomaly", generate_clear_anomaly(), 0.01),
        ("Severe Anomaly", generate_severe_anomaly(), 0.001),
    ];

    for (scenario_name, test_sequence, threshold) in test_scenarios {
        println!("\n--- {} (threshold: {}) ---", scenario_name, threshold);

        let detect_start = Instant::now();
        let anomalies = detector.detect_anomalies(&test_sequence, threshold)?;
        let detect_time = detect_start.elapsed();

        println!("Sequence: {:?}", test_sequence);
        println!("Detection time: {:?}", detect_time);

        if anomalies.is_empty() {
            println!("✅ No anomalies detected (normal behavior)");
        } else {
            println!("⚠️  {} anomalies detected:", anomalies.len());

            for (i, anomaly) in anomalies.iter().enumerate() {
                println!("   Anomaly {}: {:?}", i + 1, anomaly.sequence);
                println!("     - Likelihood: {:.6}", anomaly.likelihood);
                println!("     - Anomaly Strength: {:.3}", anomaly.anomaly_strength);
                println!("     - Information Score: {:.3}", anomaly.information_score);

                // Classify anomaly severity
                let severity = if anomaly.anomaly_strength > 0.8 {
                    "🚨 CRITICAL"
                } else if anomaly.anomaly_strength > 0.6 {
                    "⚠️  HIGH"
                } else if anomaly.anomaly_strength > 0.4 {
                    "⚡ MEDIUM"
                } else {
                    "📝 LOW"
                };

                println!("     - Severity: {}", severity);
            }
        }
    }

    // Demonstrate threshold sensitivity analysis
    println!("\n📊 Threshold Sensitivity Analysis");
    demonstrate_threshold_sensitivity(&detector)?;

    // Show mathematical properties
    println!("\n🔬 Mathematical Properties Validation");
    validate_mathematical_properties(&detector)?;

    println!("\n🎉 Quick start example completed successfully!");
    println!("💡 Key takeaways:");
    println!("   - Higher thresholds detect fewer, stronger anomalies");
    println!("   - Lower thresholds detect more, including weaker anomalies");
    println!("   - Anomaly strength combines likelihood and information content");
    println!("   - Detection is fast and mathematically sound");

    Ok(())
}

/// Generate realistic training data with clear, learnable patterns
fn generate_realistic_training_data() -> Vec<String> {
    let mut training_data = Vec::new();

    // Pattern 1: Sequential workflow (A -> B -> C -> D)
    let workflow_pattern = vec!["START", "PROCESS", "VALIDATE", "COMPLETE"];
    for _ in 0..50 {
        training_data.extend(workflow_pattern.iter().map(|s| s.to_string()));
    }

    // Pattern 2: Alternating states (X -> Y -> X -> Y)
    let alternating_pattern = vec!["STATE_X", "STATE_Y"];
    for _ in 0..30 {
        training_data.extend(alternating_pattern.iter().map(|s| s.to_string()));
    }

    // Pattern 3: Cyclic pattern (P -> Q -> R -> P)
    let cyclic_pattern = vec!["PHASE_P", "PHASE_Q", "PHASE_R"];
    for _ in 0..25 {
        training_data.extend(cyclic_pattern.iter().map(|s| s.to_string()));
    }

    training_data
}

/// Generate a sequence that follows normal patterns
fn generate_normal_sequence() -> Vec<String> {
    vec![
        "START".to_string(),
        "PROCESS".to_string(),
        "VALIDATE".to_string(),
        "COMPLETE".to_string(),
    ]
}

/// Generate a sequence with slight deviation from normal
fn generate_slight_deviation() -> Vec<String> {
    vec![
        "START".to_string(),
        "PROCESS".to_string(),
        "PROCESS".to_string(), // Repeated step (slight anomaly)
        "VALIDATE".to_string(),
        "COMPLETE".to_string(),
    ]
}

/// Generate a clearly anomalous sequence
fn generate_clear_anomaly() -> Vec<String> {
    vec![
        "START".to_string(),
        "UNKNOWN_STEP".to_string(), // Completely unknown step
        "VALIDATE".to_string(),
        "COMPLETE".to_string(),
    ]
}

/// Generate a severely anomalous sequence
fn generate_severe_anomaly() -> Vec<String> {
    vec![
        "MALICIOUS_ACTION".to_string(),
        "SECURITY_BREACH".to_string(),
        "UNAUTHORIZED_ACCESS".to_string(),
        "DATA_THEFT".to_string(),
    ]
}

/// Demonstrate how different thresholds affect detection sensitivity
fn demonstrate_threshold_sensitivity(
    detector: &AnomalyDetector,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_sequence = vec![
        "START".to_string(),
        "ANOMALOUS_STEP".to_string(),
        "VALIDATE".to_string(),
    ];

    let thresholds = vec![0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9];

    println!("Testing sequence: {:?}", test_sequence);
    println!("Threshold | Anomalies Detected | Max Strength");
    println!("---------|--------------------|-------------");

    for threshold in thresholds {
        let anomalies = detector.detect_anomalies(&test_sequence, threshold)?;
        let max_strength = anomalies
            .iter()
            .map(|a| a.anomaly_strength)
            .fold(0.0f64, f64::max);

        println!(
            "{:8.3} | {:18} | {:11.3}",
            threshold,
            anomalies.len(),
            max_strength
        );
    }

    Ok(())
}

/// Validate that the mathematical properties are correct
fn validate_mathematical_properties(
    detector: &AnomalyDetector,
) -> Result<(), Box<dyn std::error::Error>> {
    let test_sequence = vec![
        "START".to_string(),
        "PROCESS".to_string(),
        "UNKNOWN".to_string(),
    ];

    let anomalies = detector.detect_anomalies(&test_sequence, 0.0)?; // Get all anomalies

    println!(
        "Validating mathematical properties for {} anomalies:",
        anomalies.len()
    );

    for (i, anomaly) in anomalies.iter().enumerate() {
        // Validate probability bounds
        assert!(
            anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0,
            "Likelihood out of bounds: {}",
            anomaly.likelihood
        );

        // Validate anomaly strength bounds
        assert!(
            anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
            "Anomaly strength out of bounds: {}",
            anomaly.anomaly_strength
        );

        // Validate information score is non-negative
        assert!(
            anomaly.information_score >= 0.0,
            "Information score negative: {}",
            anomaly.information_score
        );

        // Validate numerical stability
        assert!(anomaly.likelihood.is_finite(), "Likelihood not finite");
        assert!(
            anomaly.anomaly_strength.is_finite(),
            "Anomaly strength not finite"
        );
        assert!(
            anomaly.information_score.is_finite(),
            "Information score not finite"
        );

        // Validate log-likelihood consistency
        if anomaly.likelihood > 0.0 {
            let expected_log_likelihood = anomaly.likelihood.ln();
            let error = (anomaly.log_likelihood - expected_log_likelihood).abs();
            assert!(
                error < 1e-10,
                "Log-likelihood inconsistency: error = {:.2e}",
                error
            );
        }

        println!(
            "   ✅ Anomaly {} passes all mathematical validations",
            i + 1
        );
    }

    println!("✅ All mathematical properties validated successfully!");
    Ok(())
}
