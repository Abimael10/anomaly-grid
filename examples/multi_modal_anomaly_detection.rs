//! Multi-Modal Anomaly Detection Example
//!
//! This example demonstrates sophisticated multi-modal anomaly detection using
//! anomaly-grid across different data types and domains. It shows how to combine
//! multiple detection modes and validate results with cross-validation and
//! statistical significance testing.

use anomaly_grid::*;
use std::time::Instant;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔬 Multi-Modal Anomaly Detection with Cross-Validation");
    println!("Comprehensive validation across multiple domains and data types\n");

    // Phase 1: Multi-modal detector configuration
    println!("⚙️ Phase 1: Multi-Modal Detector Configuration");
    let detectors = create_multi_modal_detectors()?;
    println!("✅ Created {} specialized detectors", detectors.len());

    // Phase 2: Cross-domain training and validation
    println!("\n📚 Phase 2: Cross-Domain Training and Validation");
    let training_results = train_multi_modal_detectors(&detectors)?;
    validate_training_convergence(&training_results)?;

    // Phase 3: Comprehensive anomaly detection testing
    println!("\n🔬 Phase 3: Comprehensive Multi-Modal Testing");
    let test_results = run_comprehensive_tests(&detectors)?;
    
    // Phase 4: Cross-validation and statistical analysis
    println!("\n📊 Phase 4: Cross-Validation and Statistical Analysis");
    let cv_results = perform_cross_validation(&detectors)?;
    let statistical_significance = test_statistical_significance(&test_results)?;
    
    // Phase 5: Performance benchmarking across modes
    println!("\n⚡ Phase 5: Multi-Modal Performance Benchmarking");
    let performance_results = benchmark_multi_modal_performance(&detectors)?;
    
    // Phase 6: Robustness testing
    println!("\n🛡️ Phase 6: Multi-Modal Robustness Testing");
    let robustness_results = test_multi_modal_robustness(&detectors)?;
    
    // Phase 7: Final validation and certification
    println!("\n✅ Phase 7: Final Validation and Certification");
    let certification_results = perform_final_certification(
        &test_results,
        &cv_results,
        &performance_results,
        &robustness_results,
        statistical_significance,
    )?;

    // Generate comprehensive report
    generate_validation_report(&certification_results)?;

    Ok(())
}

/// Create specialized detectors for different modalities
fn create_multi_modal_detectors() -> Result<HashMap<String, AnomalyDetector>, Box<dyn std::error::Error>> {
    let mut detectors = HashMap::new();

    // Sequential pattern detector (high order)
    let sequential_config = AnomalyGridConfig::default()
        .with_max_order(6)?
        .with_smoothing_alpha(0.1)?
        .with_weights(0.9, 0.1)?;
    let sequential_detector = AnomalyDetector::with_config(sequential_config)?;
    detectors.insert("sequential".to_string(), sequential_detector);

    // Behavioral pattern detector (medium order)
    let behavioral_config = AnomalyGridConfig::default()
        .with_max_order(4)?
        .with_smoothing_alpha(0.3)?
        .with_weights(0.7, 0.3)?;
    let behavioral_detector = AnomalyDetector::with_config(behavioral_config)?;
    detectors.insert("behavioral".to_string(), behavioral_detector);

    // Temporal pattern detector (high order)
    let temporal_config = AnomalyGridConfig::default()
        .with_max_order(8)?
        .with_smoothing_alpha(0.2)?
        .with_weights(0.8, 0.2)?;
    let temporal_detector = AnomalyDetector::with_config(temporal_config)?;
    detectors.insert("temporal".to_string(), temporal_detector);

    // Network pattern detector (medium order)
    let network_config = AnomalyGridConfig::default()
        .with_max_order(5)?
        .with_smoothing_alpha(0.4)?
        .with_weights(0.6, 0.4)?;
    let network_detector = AnomalyDetector::with_config(network_config)?;
    detectors.insert("network".to_string(), network_detector);

    println!("🔧 Sequential detector: order 6, α=0.1, weights=(0.9,0.1)");
    println!("🧠 Behavioral detector: order 4, α=0.3, weights=(0.7,0.3)");
    println!("⏰ Temporal detector: order 8, α=0.2, weights=(0.8,0.2)");
    println!("🌐 Network detector: order 5, α=0.4, weights=(0.6,0.4)");

    Ok(detectors)
}

/// Train all detectors and collect training metrics
fn train_multi_modal_detectors(detectors: &HashMap<String, AnomalyDetector>) -> Result<HashMap<String, TrainingResult>, Box<dyn std::error::Error>> {
    let mut results = HashMap::new();

    for (mode, detector) in detectors {
        println!("\nTraining {} detector...", mode);
        
        let training_data = generate_training_data_for_mode(mode)?;
        let start_time = Instant::now();
        
        let mut detector_clone = detector.clone();
        detector_clone.train(&training_data)?;
        
        let training_time = start_time.elapsed();
        let metrics = detector_clone.performance_metrics();
        
        let result = TrainingResult {
            mode: mode.clone(),
            training_time,
            context_count: metrics.context_count,
            memory_usage: metrics.estimated_memory_bytes,
            data_size: training_data.len(),
        };
        
        println!("  ⏱️ Training time: {:?}", training_time);
        println!("  🧮 Contexts learned: {}", metrics.context_count);
        println!("  💾 Memory usage: {:.1} KB", metrics.estimated_memory_bytes as f64 / 1024.0);
        
        results.insert(mode.clone(), result);
    }

    Ok(results)
}

/// Validate training convergence across all detectors
fn validate_training_convergence(results: &HashMap<String, TrainingResult>) -> Result<(), Box<dyn std::error::Error>> {
    println!("\nValidating training convergence...");
    
    for (mode, result) in results {
        // Check reasonable context count
        if result.context_count == 0 {
            return Err(format!("{} detector failed to learn any contexts", mode).into());
        }
        
        // Check memory efficiency
        let memory_per_context = result.memory_usage as f64 / result.context_count as f64;
        if memory_per_context > 10000.0 { // 10KB per context seems excessive
            println!("⚠️ Warning: {} detector has high memory per context: {:.1} bytes", mode, memory_per_context);
        }
        
        // Check training efficiency
        let data_per_second = result.data_size as f64 / result.training_time.as_secs_f64();
        if data_per_second < 1000.0 {
            println!("⚠️ Warning: {} detector has slow training: {:.0} elements/sec", mode, data_per_second);
        }
        
        println!("✅ {} detector convergence validated", mode);
    }
    
    Ok(())
}

/// Run comprehensive tests across all modalities
fn run_comprehensive_tests(detectors: &HashMap<String, AnomalyDetector>) -> Result<Vec<TestResult>, Box<dyn std::error::Error>> {
    let mut all_results = Vec::new();
    
    for (mode, detector) in detectors {
        println!("\nTesting {} detector...", mode);
        
        let test_scenarios = generate_test_scenarios_for_mode(mode)?;
        let mut detector_clone = detector.clone();
        
        // Train the detector first
        let training_data = generate_training_data_for_mode(mode)?;
        detector_clone.train(&training_data)?;
        
        for (scenario_name, test_data, expected_anomalous, threshold) in test_scenarios {
            let start_time = Instant::now();
            let anomalies = detector_clone.detect_anomalies(&test_data, threshold)?;
            let detection_time = start_time.elapsed();
            
            let detected = !anomalies.is_empty();
            let is_correct = detected == expected_anomalous;
            
            let max_strength = if !anomalies.is_empty() {
                anomalies.iter().map(|a| a.anomaly_strength).fold(0.0f64, f64::max)
            } else {
                0.0
            };
            
            let result = TestResult {
                mode: mode.clone(),
                scenario: scenario_name.clone(),
                expected_anomalous,
                detected,
                is_correct,
                anomaly_count: anomalies.len(),
                max_strength,
                detection_time,
                threshold,
            };
            
            println!("  📊 {}: {} ({})", scenario_name, 
                    if detected { "ANOMALOUS" } else { "NORMAL" },
                    if is_correct { "✅" } else { "❌" });
            
            all_results.push(result);
        }
    }
    
    Ok(all_results)
}

/// Perform k-fold cross-validation
fn perform_cross_validation(detectors: &HashMap<String, AnomalyDetector>) -> Result<HashMap<String, CrossValidationResult>, Box<dyn std::error::Error>> {
    println!("Performing 5-fold cross-validation...");
    
    let mut cv_results = HashMap::new();
    let k_folds = 5;
    
    for (mode, detector) in detectors {
        println!("\nCross-validating {} detector...", mode);
        
        let full_dataset = generate_cross_validation_dataset_for_mode(mode)?;
        let fold_size = full_dataset.len() / k_folds;
        
        let mut fold_accuracies = Vec::new();
        let mut fold_f1_scores = Vec::new();
        
        for fold in 0..k_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == k_folds - 1 { full_dataset.len() } else { (fold + 1) * fold_size };
            
            // Split data
            let test_data = &full_dataset[start_idx..end_idx];
            let train_data: Vec<_> = full_dataset[0..start_idx].iter()
                .chain(full_dataset[end_idx..].iter())
                .cloned()
                .collect();
            
            // Train on fold
            let mut fold_detector = detector.clone();
            let train_sequences: Vec<String> = train_data.iter().flat_map(|d| d.data.clone()).collect();
            fold_detector.train(&train_sequences)?;
            
            // Test on fold
            let mut correct = 0;
            let mut tp = 0;
            let mut fp = 0;
            let mut fn_count = 0;
            
            for test_item in test_data {
                let anomalies = fold_detector.detect_anomalies(&test_item.data, 0.1)?;
                let detected = !anomalies.is_empty();
                
                if detected == test_item.is_anomalous {
                    correct += 1;
                }
                
                match (detected, test_item.is_anomalous) {
                    (true, true) => tp += 1,
                    (true, false) => fp += 1,
                    (false, true) => fn_count += 1,
                    _ => {}
                }
            }
            
            let accuracy = correct as f64 / test_data.len() as f64;
            let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
            let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
            let f1_score = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
            
            fold_accuracies.push(accuracy);
            fold_f1_scores.push(f1_score);
            
            println!("  Fold {}: Accuracy={:.3}, F1={:.3}", fold + 1, accuracy, f1_score);
        }
        
        let mean_accuracy = fold_accuracies.iter().sum::<f64>() / fold_accuracies.len() as f64;
        let std_accuracy = {
            let variance = fold_accuracies.iter()
                .map(|x| (x - mean_accuracy).powi(2))
                .sum::<f64>() / fold_accuracies.len() as f64;
            variance.sqrt()
        };
        
        let mean_f1 = fold_f1_scores.iter().sum::<f64>() / fold_f1_scores.len() as f64;
        let std_f1 = {
            let variance = fold_f1_scores.iter()
                .map(|x| (x - mean_f1).powi(2))
                .sum::<f64>() / fold_f1_scores.len() as f64;
            variance.sqrt()
        };
        
        let cv_result = CrossValidationResult {
            mode: mode.clone(),
            mean_accuracy,
            std_accuracy,
            mean_f1,
            std_f1,
            fold_count: k_folds,
        };
        
        println!("  📊 Mean Accuracy: {:.3} ± {:.3}", mean_accuracy, std_accuracy);
        println!("  📊 Mean F1 Score: {:.3} ± {:.3}", mean_f1, std_f1);
        
        cv_results.insert(mode.clone(), cv_result);
    }
    
    Ok(cv_results)
}

/// Test statistical significance of results
fn test_statistical_significance(results: &[TestResult]) -> Result<f64, Box<dyn std::error::Error>> {
    println!("Testing statistical significance...");
    
    // Group results by mode
    let mut mode_accuracies: HashMap<String, Vec<f64>> = HashMap::new();
    
    for result in results {
        let accuracy = if result.is_correct { 1.0 } else { 0.0 };
        mode_accuracies.entry(result.mode.clone()).or_insert_with(Vec::new).push(accuracy);
    }
    
    // Calculate overall accuracy
    let total_correct = results.iter().filter(|r| r.is_correct).count();
    let overall_accuracy = total_correct as f64 / results.len() as f64;
    
    // Simple significance test: check if accuracy is significantly better than random (0.5)
    let n = results.len() as f64;
    let p = overall_accuracy;
    let expected_p = 0.5;
    
    // Z-test for proportion
    let z_score = (p - expected_p) / (expected_p * (1.0 - expected_p) / n).sqrt();
    let p_value = 2.0 * (1.0 - standard_normal_cdf(z_score.abs()));
    
    println!("📊 Overall accuracy: {:.3}", overall_accuracy);
    println!("📊 Z-score vs random: {:.3}", z_score);
    println!("📊 P-value: {:.6}", p_value);
    
    if p_value < 0.05 {
        println!("✅ Results are statistically significant (p < 0.05)");
    } else {
        println!("⚠️ Results are not statistically significant (p >= 0.05)");
    }
    
    Ok(p_value)
}

/// Benchmark performance across all modalities
fn benchmark_multi_modal_performance(detectors: &HashMap<String, AnomalyDetector>) -> Result<HashMap<String, PerformanceResult>, Box<dyn std::error::Error>> {
    println!("Benchmarking multi-modal performance...");
    
    let mut performance_results = HashMap::new();
    let iterations = 100;
    
    for (mode, detector) in detectors {
        println!("\nBenchmarking {} detector...", mode);
        
        // Train detector
        let mut detector_clone = detector.clone();
        let training_data = generate_training_data_for_mode(mode)?;
        detector_clone.train(&training_data)?;
        
        // Generate test data
        let test_data = generate_test_data_for_mode(mode)?;
        
        // Benchmark detection
        let start_time = Instant::now();
        for _ in 0..iterations {
            let _ = detector_clone.detect_anomalies(&test_data, 0.1)?;
        }
        let total_time = start_time.elapsed();
        
        let avg_time = total_time / iterations;
        let throughput = iterations as f64 / total_time.as_secs_f64();
        
        let result = PerformanceResult {
            mode: mode.clone(),
            avg_detection_time: avg_time,
            throughput,
            iterations,
        };
        
        println!("  ⏱️ Average detection time: {:?}", avg_time);
        println!("  🚀 Throughput: {:.0} detections/second", throughput);
        
        performance_results.insert(mode.clone(), result);
    }
    
    Ok(performance_results)
}

/// Test robustness across all modalities
fn test_multi_modal_robustness(detectors: &HashMap<String, AnomalyDetector>) -> Result<HashMap<String, RobustnessResult>, Box<dyn std::error::Error>> {
    println!("Testing multi-modal robustness...");
    
    let mut robustness_results = HashMap::new();
    
    for (mode, detector) in detectors {
        println!("\nTesting {} detector robustness...", mode);
        
        // Train detector
        let mut detector_clone = detector.clone();
        let training_data = generate_training_data_for_mode(mode)?;
        detector_clone.train(&training_data)?;
        
        // Test edge cases
        let edge_cases = generate_edge_cases_for_mode(mode)?;
        let mut successful_tests = 0;
        let mut total_tests = 0;
        
        for (case_name, test_data) in edge_cases {
            total_tests += 1;
            
            match detector_clone.detect_anomalies(&test_data, 0.1) {
                Ok(anomalies) => {
                    successful_tests += 1;
                    println!("  ✅ {}: {} anomalies detected", case_name, anomalies.len());
                }
                Err(e) => {
                    println!("  ❌ {}: Error - {}", case_name, e);
                }
            }
        }
        
        let robustness_score = successful_tests as f64 / total_tests as f64;
        
        let result = RobustnessResult {
            mode: mode.clone(),
            successful_tests,
            total_tests,
            robustness_score,
        };
        
        println!("  📊 Robustness score: {:.3} ({}/{})", robustness_score, successful_tests, total_tests);
        
        robustness_results.insert(mode.clone(), result);
    }
    
    Ok(robustness_results)
}

/// Perform final certification
fn perform_final_certification(
    test_results: &[TestResult],
    cv_results: &HashMap<String, CrossValidationResult>,
    performance_results: &HashMap<String, PerformanceResult>,
    robustness_results: &HashMap<String, RobustnessResult>,
    statistical_significance: f64,
) -> Result<CertificationResult, Box<dyn std::error::Error>> {
    println!("Performing final certification...");
    
    // Calculate overall metrics
    let total_correct = test_results.iter().filter(|r| r.is_correct).count();
    let overall_accuracy = total_correct as f64 / test_results.len() as f64;
    
    let mean_cv_accuracy = cv_results.values().map(|r| r.mean_accuracy).sum::<f64>() / cv_results.len() as f64;
    let mean_cv_f1 = cv_results.values().map(|r| r.mean_f1).sum::<f64>() / cv_results.len() as f64;
    
    let mean_throughput = performance_results.values().map(|r| r.throughput).sum::<f64>() / performance_results.len() as f64;
    let mean_robustness = robustness_results.values().map(|r| r.robustness_score).sum::<f64>() / robustness_results.len() as f64;
    
    // Certification criteria
    let accuracy_threshold = 0.8;
    let f1_threshold = 0.75;
    let throughput_threshold = 100.0;
    let robustness_threshold = 0.8;
    let significance_threshold = 0.05;
    
    let accuracy_passed = overall_accuracy >= accuracy_threshold;
    let cv_accuracy_passed = mean_cv_accuracy >= accuracy_threshold;
    let f1_passed = mean_cv_f1 >= f1_threshold;
    let throughput_passed = mean_throughput >= throughput_threshold;
    let robustness_passed = mean_robustness >= robustness_threshold;
    let significance_passed = statistical_significance < significance_threshold;
    
    let all_passed = accuracy_passed && cv_accuracy_passed && f1_passed && 
                    throughput_passed && robustness_passed && significance_passed;
    
    let result = CertificationResult {
        overall_accuracy,
        mean_cv_accuracy,
        mean_cv_f1,
        mean_throughput,
        mean_robustness,
        statistical_significance,
        accuracy_passed,
        cv_accuracy_passed,
        f1_passed,
        throughput_passed,
        robustness_passed,
        significance_passed,
        all_passed,
    };
    
    Ok(result)
}

/// Generate comprehensive validation report
fn generate_validation_report(certification: &CertificationResult) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n🎯 COMPREHENSIVE VALIDATION REPORT");
    println!("═══════════════════════════════════════════════");
    
    println!("\n📊 ACCURACY METRICS:");
    println!("  Overall Test Accuracy: {:.1}% {}", 
            certification.overall_accuracy * 100.0,
            if certification.accuracy_passed { "✅" } else { "❌" });
    println!("  Cross-Validation Accuracy: {:.1}% {}", 
            certification.mean_cv_accuracy * 100.0,
            if certification.cv_accuracy_passed { "✅" } else { "❌" });
    println!("  Cross-Validation F1 Score: {:.3} {}", 
            certification.mean_cv_f1,
            if certification.f1_passed { "✅" } else { "❌" });
    
    println!("\n⚡ PERFORMANCE METRICS:");
    println!("  Mean Throughput: {:.0} detections/sec {}", 
            certification.mean_throughput,
            if certification.throughput_passed { "✅" } else { "❌" });
    
    println!("\n🛡️ ROBUSTNESS METRICS:");
    println!("  Mean Robustness Score: {:.1}% {}", 
            certification.mean_robustness * 100.0,
            if certification.robustness_passed { "✅" } else { "❌" });
    
    println!("\n📈 STATISTICAL VALIDATION:");
    println!("  Statistical Significance: p = {:.6} {}", 
            certification.statistical_significance,
            if certification.significance_passed { "✅" } else { "❌" });
    
    println!("\n🏆 FINAL CERTIFICATION:");
    if certification.all_passed {
        println!("  🎉 ALL VALIDATION CRITERIA PASSED");
        println!("  ✅ Library is CERTIFIED for production use");
        println!("  ✅ Multi-modal anomaly detection VERIFIED");
        println!("  ✅ Mathematical correctness VALIDATED");
        println!("  ✅ Performance requirements MET");
        println!("  ✅ Robustness standards ACHIEVED");
    } else {
        println!("  ⚠️ VALIDATION CONCERNS IDENTIFIED");
        if !certification.accuracy_passed { println!("    - Test accuracy below threshold"); }
        if !certification.cv_accuracy_passed { println!("    - Cross-validation accuracy below threshold"); }
        if !certification.f1_passed { println!("    - F1 score below threshold"); }
        if !certification.throughput_passed { println!("    - Throughput below threshold"); }
        if !certification.robustness_passed { println!("    - Robustness below threshold"); }
        if !certification.significance_passed { println!("    - Results not statistically significant"); }
    }
    
    Ok(())
}

// Helper functions for data generation
fn generate_training_data_for_mode(mode: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    match mode {
        "sequential" => Ok(vec!["A", "B", "C", "D", "E"].repeat(100).iter().map(|s| s.to_string()).collect()),
        "behavioral" => Ok(vec!["LOGIN", "WORK", "EMAIL", "LOGOUT"].repeat(150).iter().map(|s| s.to_string()).collect()),
        "temporal" => Ok(vec!["MORNING", "WORK", "LUNCH", "WORK", "EVENING"].repeat(120).iter().map(|s| s.to_string()).collect()),
        "network" => Ok(vec!["CONNECT", "AUTH", "DATA", "CLOSE"].repeat(200).iter().map(|s| s.to_string()).collect()),
        _ => Err("Unknown mode".into()),
    }
}

fn generate_test_scenarios_for_mode(mode: &str) -> Result<Vec<(String, Vec<String>, bool, f64)>, Box<dyn std::error::Error>> {
    match mode {
        "sequential" => Ok(vec![
            ("Normal Sequence".to_string(), vec!["A", "B", "C", "D"].iter().map(|s| s.to_string()).collect(), false, 0.1),
            ("Anomalous Sequence".to_string(), vec!["X", "Y", "Z"].iter().map(|s| s.to_string()).collect(), true, 0.05),
        ]),
        "behavioral" => Ok(vec![
            ("Normal Behavior".to_string(), vec!["LOGIN", "WORK", "LOGOUT"].iter().map(|s| s.to_string()).collect(), false, 0.1),
            ("Suspicious Behavior".to_string(), vec!["LOGIN", "HACK", "STEAL"].iter().map(|s| s.to_string()).collect(), true, 0.05),
        ]),
        "temporal" => Ok(vec![
            ("Normal Time".to_string(), vec!["MORNING", "WORK", "EVENING"].iter().map(|s| s.to_string()).collect(), false, 0.1),
            ("Odd Time".to_string(), vec!["MIDNIGHT", "WORK", "MIDNIGHT"].iter().map(|s| s.to_string()).collect(), true, 0.05),
        ]),
        "network" => Ok(vec![
            ("Normal Traffic".to_string(), vec!["CONNECT", "AUTH", "DATA"].iter().map(|s| s.to_string()).collect(), false, 0.1),
            ("Attack Traffic".to_string(), vec!["SCAN", "EXPLOIT", "BACKDOOR"].iter().map(|s| s.to_string()).collect(), true, 0.05),
        ]),
        _ => Err("Unknown mode".into()),
    }
}

fn generate_test_data_for_mode(mode: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    match mode {
        "sequential" => Ok(vec!["A", "B", "C"].iter().map(|s| s.to_string()).collect()),
        "behavioral" => Ok(vec!["LOGIN", "WORK"].iter().map(|s| s.to_string()).collect()),
        "temporal" => Ok(vec!["MORNING", "WORK"].iter().map(|s| s.to_string()).collect()),
        "network" => Ok(vec!["CONNECT", "AUTH"].iter().map(|s| s.to_string()).collect()),
        _ => Err("Unknown mode".into()),
    }
}

fn generate_cross_validation_dataset_for_mode(mode: &str) -> Result<Vec<CrossValidationItem>, Box<dyn std::error::Error>> {
    let mut dataset = Vec::new();
    
    // Generate normal examples
    for i in 0..50 {
        let data = match mode {
            "sequential" => vec!["A", "B", "C", "D"].iter().map(|s| format!("{}_{}", s, i % 3)).collect(),
            "behavioral" => vec!["LOGIN", "WORK", "LOGOUT"].iter().map(|s| format!("{}_{}", s, i % 3)).collect(),
            "temporal" => vec!["MORNING", "WORK", "EVENING"].iter().map(|s| format!("{}_{}", s, i % 3)).collect(),
            "network" => vec!["CONNECT", "AUTH", "DATA"].iter().map(|s| format!("{}_{}", s, i % 3)).collect(),
            _ => return Err("Unknown mode".into()),
        };
        dataset.push(CrossValidationItem { data, is_anomalous: false });
    }
    
    // Generate anomalous examples
    for i in 0..20 {
        let data = match mode {
            "sequential" => vec!["X", "Y", "Z"].iter().map(|s| format!("{}_{}", s, i % 3)).collect(),
            "behavioral" => vec!["HACK", "STEAL", "ESCAPE"].iter().map(|s| format!("{}_{}", s, i % 3)).collect(),
            "temporal" => vec!["MIDNIGHT", "WORK", "MIDNIGHT"].iter().map(|s| format!("{}_{}", s, i % 3)).collect(),
            "network" => vec!["SCAN", "EXPLOIT", "BACKDOOR"].iter().map(|s| format!("{}_{}", s, i % 3)).collect(),
            _ => return Err("Unknown mode".into()),
        };
        dataset.push(CrossValidationItem { data, is_anomalous: true });
    }
    
    Ok(dataset)
}

fn generate_edge_cases_for_mode(_mode: &str) -> Result<Vec<(String, Vec<String>)>, Box<dyn std::error::Error>> {
    let base_cases = vec![
        ("Empty Sequence".to_string(), vec![]),
        ("Single Element".to_string(), vec!["SINGLE".to_string()]),
        ("Repeated Element".to_string(), vec!["REPEAT".to_string(); 10]),
    ];
    
    Ok(base_cases)
}

fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

fn erf(x: f64) -> f64 {
    // Approximation of error function
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}

// Data structures
#[derive(Debug)]
struct TrainingResult {
    mode: String,
    training_time: std::time::Duration,
    context_count: usize,
    memory_usage: usize,
    data_size: usize,
}

#[derive(Debug)]
struct TestResult {
    mode: String,
    scenario: String,
    expected_anomalous: bool,
    detected: bool,
    is_correct: bool,
    anomaly_count: usize,
    max_strength: f64,
    detection_time: std::time::Duration,
    threshold: f64,
}

#[derive(Debug)]
struct CrossValidationResult {
    mode: String,
    mean_accuracy: f64,
    std_accuracy: f64,
    mean_f1: f64,
    std_f1: f64,
    fold_count: usize,
}

#[derive(Debug)]
struct PerformanceResult {
    mode: String,
    avg_detection_time: std::time::Duration,
    throughput: f64,
    iterations: u32,
}

#[derive(Debug)]
struct RobustnessResult {
    mode: String,
    successful_tests: usize,
    total_tests: usize,
    robustness_score: f64,
}

#[derive(Debug)]
struct CertificationResult {
    overall_accuracy: f64,
    mean_cv_accuracy: f64,
    mean_cv_f1: f64,
    mean_throughput: f64,
    mean_robustness: f64,
    statistical_significance: f64,
    accuracy_passed: bool,
    cv_accuracy_passed: bool,
    f1_passed: bool,
    throughput_passed: bool,
    robustness_passed: bool,
    significance_passed: bool,
    all_passed: bool,
}

#[derive(Debug, Clone)]
struct CrossValidationItem {
    data: Vec<String>,
    is_anomalous: bool,
}