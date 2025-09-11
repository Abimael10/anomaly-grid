//! Time Series Anomaly Detection Example
//!
//! This example demonstrates time series anomaly detection using the anomaly-grid
//! library. It shows how to discretize continuous time series data and detect
//! anomalies in temporal patterns.

#![allow(clippy::uninlined_format_args)]
#![allow(clippy::useless_vec)]
#![allow(clippy::manual_range_contains)]

use anomaly_grid::*;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Time Series Anomaly Detection with Mathematical Validation");
    println!("Converting continuous data to categorical sequences for anomaly detection\n");

    // Configure detector for time series pattern analysis
    let config = AnomalyGridConfig::default()
        .with_max_order(8)? // High order for temporal dependencies
        .with_smoothing_alpha(0.2)? // Low smoothing for precise pattern learning
        .with_weights(0.9, 0.1)?; // Emphasize likelihood for temporal patterns

    let mut detector = AnomalyDetector::with_config(config)?;
    println!("✅ Configured time series detector with order 8 for temporal patterns");

    // Phase 1: Generate and validate time series training data
    println!("\n📊 Phase 1: Time Series Data Generation and Discretization");
    let (raw_time_series, discretized_series) = generate_time_series_data(1000)?;

    // Validate discretization quality
    validate_discretization_quality(&raw_time_series, &discretized_series)?;

    println!("📈 Generated {} time series points", raw_time_series.len());
    println!(
        "🔢 Discretized to {} categorical states",
        discretized_series.len()
    );
    println!("🎯 Discretization validation: PASSED");

    // Phase 2: Train and validate temporal model
    println!("\n🎯 Phase 2: Temporal Model Training and Validation");
    let train_start = Instant::now();
    detector.train(&discretized_series)?;
    let train_time = train_start.elapsed();

    let metrics = detector.performance_metrics();
    println!("⏱️ Training completed in {:?}", train_time);
    println!("🧮 Temporal patterns learned: {}", metrics.context_count);
    println!(
        "💾 Memory usage: {:.1} KB",
        metrics.estimated_memory_bytes as f64 / 1024.0
    );

    // Validate temporal consistency
    validate_temporal_consistency(&detector, &discretized_series)?;

    // Phase 3: Comprehensive anomaly detection testing
    println!("\n🔬 Phase 3: Time Series Anomaly Detection Testing");

    let test_scenarios = vec![
        ("Normal Trend", generate_normal_trend(), false, 0.1),
        ("Sudden Spike", generate_sudden_spike(), true, 0.05),
        ("Gradual Drift", generate_gradual_drift(), true, 0.08),
        ("Periodic Anomaly", generate_periodic_anomaly(), true, 0.03),
        ("Noise Burst", generate_noise_burst(), true, 0.02),
        ("Level Shift", generate_level_shift(), true, 0.05),
        ("Trend Change", generate_trend_change(), true, 0.06),
        ("Seasonal Break", generate_seasonal_break(), true, 0.04),
    ];

    let mut detection_results = Vec::new();
    let mut total_detection_time = std::time::Duration::new(0, 0);

    for (scenario_name, (raw_data, test_sequence), is_anomalous, threshold) in test_scenarios {
        println!("\n--- Testing: {} ---", scenario_name);
        println!(
            "Expected: {}",
            if is_anomalous { "ANOMALOUS" } else { "NORMAL" }
        );
        println!("Raw data points: {}", raw_data.len());
        println!("Discretized sequence length: {}", test_sequence.len());
        println!("Threshold: {}", threshold);

        let detect_start = Instant::now();
        let anomalies = detector.detect_anomalies(&test_sequence, threshold)?;
        let detect_time = detect_start.elapsed();
        total_detection_time += detect_time;

        let detected = !anomalies.is_empty();
        let is_correct = detected == is_anomalous;

        println!(
            "🔍 Detection result: {}",
            if detected { "ANOMALOUS" } else { "NORMAL" }
        );
        println!(
            "✅ Correctness: {}",
            if is_correct { "CORRECT" } else { "INCORRECT" }
        );
        println!("⏱️ Detection time: {:?}", detect_time);

        if detected {
            let max_strength = anomalies
                .iter()
                .map(|a| a.anomaly_strength)
                .fold(0.0f64, f64::max);
            let avg_information =
                anomalies.iter().map(|a| a.information_score).sum::<f64>() / anomalies.len() as f64;

            println!("📊 Anomalies detected: {}", anomalies.len());
            println!("🎯 Max anomaly strength: {:.4}", max_strength);
            println!("📈 Avg information score: {:.4}", avg_information);

            // Validate mathematical properties
            validate_anomaly_mathematical_properties(&anomalies)?;

            // Show temporal context of strongest anomaly
            if let Some(strongest) = anomalies
                .iter()
                .max_by(|a, b| a.anomaly_strength.partial_cmp(&b.anomaly_strength).unwrap())
            {
                println!("🔍 Strongest anomaly pattern: {:?}", strongest.sequence);
            }
        }

        detection_results.push((
            scenario_name.to_string(),
            is_anomalous,
            detected,
            is_correct,
        ));
    }

    // Phase 4: Statistical validation and metrics
    println!("\n📊 Phase 4: Statistical Validation and Performance Metrics");

    let accuracy = calculate_accuracy(&detection_results);
    let (precision, recall, f1_score) = calculate_precision_recall_f1(&detection_results);

    println!("🎯 Detection Accuracy: {:.1}%", accuracy * 100.0);
    println!("🎯 Precision: {:.3}", precision);
    println!("🎯 Recall: {:.3}", recall);
    println!("🎯 F1 Score: {:.3}", f1_score);
    println!(
        "⏱️ Average detection time: {:?}",
        total_detection_time / detection_results.len() as u32
    );

    // Phase 5: Temporal pattern analysis
    println!("\n⏰ Phase 5: Temporal Pattern Analysis");
    analyze_temporal_patterns(&detector)?;

    // Phase 6: Sensitivity analysis
    println!("\n🎚️ Phase 6: Sensitivity Analysis");
    perform_sensitivity_analysis(&detector)?;

    // Phase 7: Robustness testing with edge cases
    println!("\n🛡️ Phase 7: Robustness Testing");
    test_temporal_robustness(&detector)?;

    // Phase 8: Performance benchmarking
    println!("\n⚡ Phase 8: Performance Benchmarking");
    benchmark_temporal_performance(&detector)?;

    // Phase 9: Mathematical validation
    println!("\n🔬 Phase 9: Mathematical Property Validation");
    validate_mathematical_properties(&detector)?;

    // Final validation summary
    println!("\n✅ COMPREHENSIVE VALIDATION SUMMARY");
    println!("═══════════════════════════════════════");
    println!("✅ Time series discretization: VALIDATED");
    println!("✅ Temporal consistency: VALIDATED");
    println!("✅ Mathematical properties: VALIDATED");
    println!("✅ Detection accuracy: {:.1}%", accuracy * 100.0);
    println!("✅ F1 Score: {:.3}", f1_score);
    println!(
        "✅ Performance: {} detections/sec",
        (1000.0 / total_detection_time.as_millis() as f64 * detection_results.len() as f64) as u32
    );

    // Determine overall validation status
    let validation_passed = accuracy >= 0.8
        && f1_score >= 0.75
        && total_detection_time.as_millis() < 100 * detection_results.len() as u128;

    if validation_passed {
        println!("🎉 ALL VALIDATIONS PASSED - TIME SERIES DETECTION VERIFIED");
        println!("📈 Library successfully handles temporal anomaly detection");
    } else {
        println!("⚠️ VALIDATION CONCERNS - REVIEW REQUIRED");
        if accuracy < 0.8 {
            println!("   - Accuracy below threshold");
        }
        if f1_score < 0.75 {
            println!("   - F1 score below threshold");
        }
    }

    Ok(())
}

/// Generate synthetic time series data with known patterns
fn generate_time_series_data(
    length: usize,
) -> Result<(Vec<f64>, Vec<String>), Box<dyn std::error::Error>> {
    let mut raw_data = Vec::new();

    // Generate realistic time series with multiple components
    for i in 0..length {
        let t = i as f64;

        // Base trend
        let trend = 0.01 * t;

        // Seasonal component (daily cycle)
        let seasonal = 2.0 * (2.0 * std::f64::consts::PI * t / 24.0).sin();

        // Weekly pattern
        let weekly = 0.5 * (2.0 * std::f64::consts::PI * t / (24.0 * 7.0)).sin();

        // Random noise
        let noise = 0.3 * ((t * 12.9898).sin() * 43758.5453).fract() - 0.15;

        let value = 10.0 + trend + seasonal + weekly + noise;
        raw_data.push(value);
    }

    // Discretize the time series into categorical states
    let discretized = discretize_time_series(&raw_data)?;

    Ok((raw_data, discretized))
}

/// Discretize continuous time series into categorical states
fn discretize_time_series(data: &[f64]) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    if data.is_empty() {
        return Ok(Vec::new());
    }

    // Calculate statistics for discretization
    let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();

    let mut discretized = Vec::new();

    for (i, &value) in data.iter().enumerate() {
        // Multi-level discretization for rich categorical representation

        // 1. Absolute level (5 states)
        let level = if value < min_val + 0.2 * (max_val - min_val) {
            "VERY_LOW"
        } else if value < min_val + 0.4 * (max_val - min_val) {
            "LOW"
        } else if value < min_val + 0.6 * (max_val - min_val) {
            "MEDIUM"
        } else if value < min_val + 0.8 * (max_val - min_val) {
            "HIGH"
        } else {
            "VERY_HIGH"
        };

        // 2. Relative to mean (3 states)
        let _relative = if value < mean - 0.5 * std_dev {
            "BELOW_MEAN"
        } else if value > mean + 0.5 * std_dev {
            "ABOVE_MEAN"
        } else {
            "NEAR_MEAN"
        };

        // 3. Trend direction (if not first point)
        let trend = if i > 0 {
            let diff = value - data[i - 1];
            if diff > 0.1 {
                "RISING"
            } else if diff < -0.1 {
                "FALLING"
            } else {
                "STABLE"
            }
        } else {
            "STABLE"
        };

        // 4. Volatility (if we have enough history)
        let _volatility = if i >= 5 {
            let recent_std = {
                let recent: Vec<f64> = data[i - 5..=i].to_vec();
                let recent_mean = recent.iter().sum::<f64>() / recent.len() as f64;
                let recent_var = recent
                    .iter()
                    .map(|x| (x - recent_mean).powi(2))
                    .sum::<f64>()
                    / recent.len() as f64;
                recent_var.sqrt()
            };

            if recent_std > 1.5 * std_dev {
                "HIGH_VOLATILITY"
            } else if recent_std < 0.5 * std_dev {
                "LOW_VOLATILITY"
            } else {
                "NORMAL_VOLATILITY"
            }
        } else {
            "NORMAL_VOLATILITY"
        };

        // Combine into rich categorical representation
        // Combine into simpler categorical representation for better diversity
        discretized.push(format!("{}_{}", level, trend));
    }

    Ok(discretized)
}

/// Validate the quality of time series discretization
fn validate_discretization_quality(
    raw_data: &[f64],
    discretized: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    if raw_data.len() != discretized.len() {
        return Err("Discretization length mismatch".into());
    }

    // Check vocabulary diversity
    let unique_states: std::collections::HashSet<_> = discretized.iter().collect();
    let diversity = unique_states.len() as f64 / discretized.len() as f64;

    if diversity < 0.01 {
        return Err("Discretization lacks diversity".into());
    }

    // Check for reasonable state distribution
    let mut state_counts = std::collections::HashMap::new();
    for state in discretized {
        *state_counts.entry(state.clone()).or_insert(0) += 1;
    }

    let max_count = state_counts.values().max().unwrap_or(&0);
    let dominance = *max_count as f64 / discretized.len() as f64;

    if dominance > 0.3 {
        return Err("Discretization has excessive state dominance".into());
    }

    println!("📊 Discretization diversity: {:.2}%", diversity * 100.0);
    println!("📊 Unique states: {}", unique_states.len());
    println!("📊 Max state dominance: {:.2}%", dominance * 100.0);

    Ok(())
}

/// Validate temporal consistency of the model
fn validate_temporal_consistency(
    detector: &AnomalyDetector,
    training_data: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    // Test that identical sequences produce identical results
    let test_sequence = training_data[0..20].to_vec();

    let result1 = detector.detect_anomalies(&test_sequence, 0.1)?;
    let result2 = detector.detect_anomalies(&test_sequence, 0.1)?;

    if result1.len() != result2.len() {
        return Err("Temporal model not deterministic".into());
    }

    // Validate that training sequences have reasonable anomaly scores
    let training_sample = training_data[0..50].to_vec();
    let anomalies = detector.detect_anomalies(&training_sample, 0.8)?;

    if anomalies.len() > training_sample.len() / 4 {
        println!("⚠️ Warning: High anomaly rate in training data - possible overfitting");
    }

    println!("✅ Temporal consistency validated");
    Ok(())
}

/// Generate normal trending time series
fn generate_normal_trend() -> (Vec<f64>, Vec<String>) {
    let mut raw_data = Vec::new();

    for i in 0..50 {
        let t = i as f64;
        let value = 10.0 + 0.1 * t + 0.5 * (t / 5.0).sin();
        raw_data.push(value);
    }

    let discretized = discretize_time_series(&raw_data).unwrap();
    (raw_data, discretized)
}

/// Generate sudden spike anomaly
fn generate_sudden_spike() -> (Vec<f64>, Vec<String>) {
    let mut raw_data = Vec::new();

    for i in 0..50 {
        let t = i as f64;
        let mut value = 10.0 + 0.1 * t;

        // Add sudden spike at position 25
        if i == 25 {
            value += 10.0;
        }

        raw_data.push(value);
    }

    let discretized = discretize_time_series(&raw_data).unwrap();
    (raw_data, discretized)
}

/// Generate gradual drift anomaly
fn generate_gradual_drift() -> (Vec<f64>, Vec<String>) {
    let mut raw_data = Vec::new();

    for i in 0..50 {
        let t = i as f64;
        let mut value = 10.0;

        // Add gradual drift starting at position 20
        if i >= 20 {
            value += 0.3 * (t - 20.0);
        }

        raw_data.push(value);
    }

    let discretized = discretize_time_series(&raw_data).unwrap();
    (raw_data, discretized)
}

/// Generate periodic anomaly
fn generate_periodic_anomaly() -> (Vec<f64>, Vec<String>) {
    let mut raw_data = Vec::new();

    for i in 0..50 {
        let t = i as f64;
        let mut value = 10.0 + (t / 10.0).sin();

        // Add anomalous frequency component
        if i >= 20 && i <= 35 {
            value += 2.0 * (t / 2.0).sin();
        }

        raw_data.push(value);
    }

    let discretized = discretize_time_series(&raw_data).unwrap();
    (raw_data, discretized)
}

/// Generate noise burst anomaly
fn generate_noise_burst() -> (Vec<f64>, Vec<String>) {
    let mut raw_data = Vec::new();

    for i in 0..50 {
        let t = i as f64;
        let mut value = 10.0;

        // Add noise burst
        if i >= 15 && i <= 25 {
            let noise = 3.0 * ((t * 12.9898).sin() * 43758.5453).fract() - 1.5;
            value += noise;
        }

        raw_data.push(value);
    }

    let discretized = discretize_time_series(&raw_data).unwrap();
    (raw_data, discretized)
}

/// Generate level shift anomaly
fn generate_level_shift() -> (Vec<f64>, Vec<String>) {
    let mut raw_data = Vec::new();

    for i in 0..50 {
        let mut value = 10.0;

        // Add level shift at position 25
        if i >= 25 {
            value += 5.0;
        }

        raw_data.push(value);
    }

    let discretized = discretize_time_series(&raw_data).unwrap();
    (raw_data, discretized)
}

/// Generate trend change anomaly
fn generate_trend_change() -> (Vec<f64>, Vec<String>) {
    let mut raw_data = Vec::new();

    for i in 0..50 {
        let t = i as f64;
        let mut value = 10.0;

        if i < 25 {
            value += 0.2 * t; // Positive trend
        } else {
            value += 5.0 - 0.2 * (t - 25.0); // Negative trend
        }

        raw_data.push(value);
    }

    let discretized = discretize_time_series(&raw_data).unwrap();
    (raw_data, discretized)
}

/// Generate seasonal break anomaly
fn generate_seasonal_break() -> (Vec<f64>, Vec<String>) {
    let mut raw_data = Vec::new();

    for i in 0..50 {
        let t = i as f64;
        let mut value = 10.0;

        if i < 30 {
            value += 2.0 * (t / 8.0).sin(); // Normal seasonality
        } else {
            value += 2.0 * (t / 4.0).sin(); // Changed seasonality
        }

        raw_data.push(value);
    }

    let discretized = discretize_time_series(&raw_data).unwrap();
    (raw_data, discretized)
}

/// Validate mathematical properties of detected anomalies
fn validate_anomaly_mathematical_properties(
    anomalies: &[AnomalyScore],
) -> Result<(), Box<dyn std::error::Error>> {
    for (i, anomaly) in anomalies.iter().enumerate() {
        // Validate probability bounds
        if !(0.0..=1.0).contains(&anomaly.likelihood) {
            return Err(format!(
                "Anomaly {} likelihood out of bounds: {}",
                i, anomaly.likelihood
            )
            .into());
        }

        // Validate anomaly strength bounds
        if !(0.0..=1.0).contains(&anomaly.anomaly_strength) {
            return Err(format!(
                "Anomaly {} strength out of bounds: {}",
                i, anomaly.anomaly_strength
            )
            .into());
        }

        // Validate information score
        if anomaly.information_score < 0.0 || !anomaly.information_score.is_finite() {
            return Err(format!(
                "Anomaly {} information score invalid: {}",
                i, anomaly.information_score
            )
            .into());
        }

        // Validate numerical stability
        if !anomaly.likelihood.is_finite() || !anomaly.anomaly_strength.is_finite() {
            return Err(format!("Anomaly {} contains non-finite values", i).into());
        }
    }

    println!(
        "✅ Mathematical properties validated for {} anomalies",
        anomalies.len()
    );
    Ok(())
}

/// Calculate detection accuracy
fn calculate_accuracy(results: &[(String, bool, bool, bool)]) -> f64 {
    let correct = results
        .iter()
        .filter(|(_, _, _, is_correct)| *is_correct)
        .count();
    correct as f64 / results.len() as f64
}

/// Calculate precision, recall, and F1 score
fn calculate_precision_recall_f1(results: &[(String, bool, bool, bool)]) -> (f64, f64, f64) {
    let mut tp = 0; // True positives
    let mut fp = 0; // False positives
    let mut _tn = 0; // True negatives
    let mut fn_count = 0; // False negatives

    for (_, is_anomalous, detected, _) in results {
        match (*detected, *is_anomalous) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, false) => _tn += 1,
            (false, true) => fn_count += 1,
        }
    }

    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };
    let f1_score = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    (precision, recall, f1_score)
}

/// Analyze temporal patterns learned by the model
fn analyze_temporal_patterns(detector: &AnomalyDetector) -> Result<(), Box<dyn std::error::Error>> {
    println!("Analyzing learned temporal patterns...");

    // Test various temporal patterns
    let pattern_tests = vec![
        (
            "Stable Pattern",
            vec!["MEDIUM_NEAR_MEAN_STABLE_NORMAL_VOLATILITY"; 5],
        ),
        (
            "Rising Pattern",
            vec![
                "LOW_BELOW_MEAN_RISING_NORMAL_VOLATILITY",
                "MEDIUM_NEAR_MEAN_RISING_NORMAL_VOLATILITY",
                "HIGH_ABOVE_MEAN_RISING_NORMAL_VOLATILITY",
            ],
        ),
        (
            "Volatile Pattern",
            vec!["MEDIUM_NEAR_MEAN_STABLE_HIGH_VOLATILITY"; 3],
        ),
    ];

    for (pattern_name, pattern) in pattern_tests {
        let pattern_strings: Vec<String> = pattern.iter().map(|s| s.to_string()).collect();
        let anomalies = detector.detect_anomalies(&pattern_strings, 0.1)?;

        println!(
            "📊 {}: {} anomalies detected",
            pattern_name,
            anomalies.len()
        );

        if !anomalies.is_empty() {
            let avg_strength =
                anomalies.iter().map(|a| a.anomaly_strength).sum::<f64>() / anomalies.len() as f64;
            println!("   Average anomaly strength: {:.4}", avg_strength);
        }
    }

    Ok(())
}

/// Perform sensitivity analysis across different thresholds
fn perform_sensitivity_analysis(
    detector: &AnomalyDetector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Performing sensitivity analysis...");

    let test_sequence = generate_sudden_spike().1;
    let thresholds = vec![0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5];

    println!("Threshold | Anomalies | Max Strength");
    println!("-----------|-----------|-------------");

    for threshold in thresholds {
        let anomalies = detector.detect_anomalies(&test_sequence, threshold)?;
        let max_strength = anomalies
            .iter()
            .map(|a| a.anomaly_strength)
            .fold(0.0f64, f64::max);

        println!(
            "{:8.3} | {:9} | {:12.4}",
            threshold,
            anomalies.len(),
            max_strength
        );
    }

    Ok(())
}

/// Test robustness with temporal edge cases
fn test_temporal_robustness(detector: &AnomalyDetector) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing temporal robustness...");

    let edge_cases = vec![
        (
            "Constant Values",
            vec!["MEDIUM_NEAR_MEAN_STABLE_NORMAL_VOLATILITY"; 20],
        ),
        (
            "Alternating States",
            vec![
                "HIGH_ABOVE_MEAN_RISING_NORMAL_VOLATILITY",
                "LOW_BELOW_MEAN_FALLING_NORMAL_VOLATILITY",
            ]
            .repeat(10),
        ),
        (
            "Single Transition",
            vec![
                "LOW_BELOW_MEAN_STABLE_NORMAL_VOLATILITY",
                "HIGH_ABOVE_MEAN_STABLE_NORMAL_VOLATILITY",
            ],
        ),
    ];

    for (case_name, sequence) in edge_cases {
        let sequence_strings: Vec<String> = sequence.iter().map(|s| s.to_string()).collect();

        let result = detector.detect_anomalies(&sequence_strings, 0.1);
        match result {
            Ok(anomalies) => {
                println!("✅ {}: {} anomalies detected", case_name, anomalies.len());

                if !anomalies.is_empty() {
                    validate_anomaly_mathematical_properties(&anomalies)?;
                }
            }
            Err(e) => {
                println!("⚠️ {}: Error - {}", case_name, e);
            }
        }
    }

    Ok(())
}

/// Benchmark temporal detection performance
fn benchmark_temporal_performance(
    detector: &AnomalyDetector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Benchmarking temporal detection performance...");

    let test_sequence = generate_normal_trend().1;
    let iterations = 500;

    let start_time = Instant::now();
    for _ in 0..iterations {
        let _ = detector.detect_anomalies(&test_sequence, 0.1)?;
    }
    let total_time = start_time.elapsed();

    let avg_time = total_time / iterations;
    let throughput = iterations as f64 / total_time.as_secs_f64();

    println!("📊 Temporal Performance Benchmark:");
    println!("   Iterations: {}", iterations);
    println!("   Total time: {:?}", total_time);
    println!("   Average time per detection: {:?}", avg_time);
    println!("   Throughput: {:.0} detections/second", throughput);

    if avg_time.as_millis() > 20 {
        println!("⚠️ Warning: Detection time exceeds 20ms threshold");
    } else {
        println!("✅ Temporal performance requirements met");
    }

    Ok(())
}

/// Validate mathematical properties of the detector
fn validate_mathematical_properties(
    detector: &AnomalyDetector,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Validating mathematical properties...");

    // Test monotonicity: higher thresholds should detect fewer anomalies
    let test_sequence = generate_sudden_spike().1;
    let thresholds = vec![0.01, 0.05, 0.1, 0.2];
    let mut prev_count = usize::MAX;

    for threshold in thresholds {
        let anomalies = detector.detect_anomalies(&test_sequence, threshold)?;
        let count = anomalies.len();

        if count > prev_count {
            return Err(format!(
                "Monotonicity violation: threshold {} detected more anomalies than lower threshold",
                threshold
            )
            .into());
        }

        prev_count = count;
    }

    println!("✅ Threshold monotonicity validated");

    // Test consistency: same input should give same output
    let result1 = detector.detect_anomalies(&test_sequence, 0.1)?;
    let result2 = detector.detect_anomalies(&test_sequence, 0.1)?;

    if result1.len() != result2.len() {
        return Err("Consistency violation: same input produced different outputs".into());
    }

    println!("✅ Detection consistency validated");

    Ok(())
}
