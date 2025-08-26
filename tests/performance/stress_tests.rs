//! Stress Testing and Performance Validation
//! 
//! This module tests the library under extreme conditions and validates
//! performance characteristics with large datasets.

use anomaly_grid::*;
use std::time::Instant;
use std::collections::HashSet;

#[test]
fn test_large_alphabet_stress() {
    println!("🔤 Testing Large Alphabet Stress (100 unique states)");
    
    let start_time = Instant::now();
    
    // Generate sequence with 100 unique states
    let alphabet_size = 100;
    let sequence_length = 10000;
    
    let sequence: Vec<String> = (0..sequence_length)
        .map(|i| format!("STATE_{:03}", i % alphabet_size))
        .collect();
    
    let mut detector = AnomalyDetector::new(3);
    
    let train_start = Instant::now();
    detector.train(&sequence).unwrap();
    let train_time = train_start.elapsed();
    
    let detect_start = Instant::now();
    let anomalies = detector.detect_anomalies(&sequence, 0.01);
    let detect_time = detect_start.elapsed();
    
    let total_time = start_time.elapsed();
    
    println!("Large Alphabet Results:");
    println!("  Alphabet Size: {}", alphabet_size);
    println!("  Sequence Length: {}", sequence_length);
    println!("  Training Time: {:?}", train_time);
    println!("  Detection Time: {:?}", detect_time);
    println!("  Total Time: {:?}", total_time);
    println!("  Anomalies Detected: {}", anomalies.len());
    
    // Performance assertions
    assert!(
        train_time.as_secs() < 30,
        "Training should complete within 30 seconds: {:?}",
        train_time
    );
    
    assert!(
        detect_time.as_secs() < 10,
        "Detection should complete within 10 seconds: {:?}",
        detect_time
    );
    
    // Validate results
    for anomaly in &anomalies {
        assert!(anomaly.likelihood.is_finite());
        assert!(anomaly.information_score.is_finite());
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
    }
    
    println!("✅ Large alphabet stress test passed");
}

#[test]
fn test_high_order_markov_stress() {
    println!("📊 Testing High-Order Markov Model Stress (order=6)");
    
    let start_time = Instant::now();
    
    // Test with high-order Markov model
    let max_order = 6;
    let sequence_length = 5000;
    
    // Create sequence with long-range dependencies
    let mut sequence = Vec::new();
    let patterns = vec![
        vec!["A", "B", "C", "D", "E", "F", "G"],
        vec!["X", "Y", "Z", "W", "V", "U", "T"],
        vec!["P", "Q", "R", "S", "T", "U", "V"],
    ];
    
    for i in 0..sequence_length {
        let pattern = &patterns[i % patterns.len()];
        let state = pattern[i % pattern.len()];
        sequence.push(state.to_string());
    }
    
    let mut detector = AnomalyDetector::new(max_order);
    
    let train_start = Instant::now();
    detector.train(&sequence).unwrap();
    let train_time = train_start.elapsed();
    
    let context_tree = detector.model().context_tree();
    let context_count = context_tree.context_count();
    
    let detect_start = Instant::now();
    let anomalies = detector.detect_anomalies(&sequence, 0.05);
    let detect_time = detect_start.elapsed();
    
    let total_time = start_time.elapsed();
    
    println!("High-Order Markov Results:");
    println!("  Max Order: {}", max_order);
    println!("  Sequence Length: {}", sequence_length);
    println!("  Contexts Created: {}", context_count);
    println!("  Training Time: {:?}", train_time);
    println!("  Detection Time: {:?}", detect_time);
    println!("  Total Time: {:?}", total_time);
    println!("  Anomalies Detected: {}", anomalies.len());
    
    // Memory efficiency check
    let unique_states: HashSet<String> = sequence.into_iter().collect();
    let alphabet_size = unique_states.len();
    let theoretical_max_contexts = (1..=max_order)
        .map(|order| alphabet_size.pow(order as u32))
        .sum::<usize>();
    
    let memory_efficiency = context_count as f64 / theoretical_max_contexts as f64;
    
    println!("  Memory Efficiency: {:.2}% ({}/{})", 
             memory_efficiency * 100.0, context_count, theoretical_max_contexts);
    
    assert!(
        memory_efficiency < 0.8,
        "Memory usage should be efficient: {:.2}%",
        memory_efficiency * 100.0
    );
    
    println!("✅ High-order Markov stress test passed");
}

#[test]
fn test_very_long_sequence_stress() {
    println!("📏 Testing Very Long Sequence Stress (100K elements)");
    
    let start_time = Instant::now();
    
    let sequence_length = 100_000;
    let alphabet = vec!["A", "B", "C", "D", "E"];
    
    // Generate pseudo-random but deterministic sequence
    let sequence: Vec<String> = (0..sequence_length)
        .map(|i| {
            let state_idx = (i * 7 + i * i / 100) % alphabet.len();
            alphabet[state_idx].to_string()
        })
        .collect();
    
    let mut detector = AnomalyDetector::new(4);
    
    let train_start = Instant::now();
    detector.train(&sequence).unwrap();
    let train_time = train_start.elapsed();
    
    // Test detection on subset to avoid excessive time
    let test_subset: Vec<String> = sequence.iter().step_by(10).cloned().collect();
    
    let detect_start = Instant::now();
    let anomalies = detector.detect_anomalies(&test_subset, 0.01);
    let detect_time = detect_start.elapsed();
    
    let total_time = start_time.elapsed();
    
    println!("Very Long Sequence Results:");
    println!("  Training Sequence Length: {}", sequence_length);
    println!("  Test Sequence Length: {}", test_subset.len());
    println!("  Training Time: {:?}", train_time);
    println!("  Detection Time: {:?}", detect_time);
    println!("  Total Time: {:?}", total_time);
    println!("  Anomalies Detected: {}", anomalies.len());
    
    // Performance requirements
    assert!(
        train_time.as_secs() < 60,
        "Training should complete within 60 seconds for 100K sequence: {:?}",
        train_time
    );
    
    // Memory usage should be reasonable
    let context_tree = detector.model().context_tree();
    let context_count = context_tree.context_count();
    
    assert!(
        context_count < 10_000,
        "Context count should be reasonable: {}",
        context_count
    );
    
    println!("✅ Very long sequence stress test passed");
}

#[test]
fn test_parallel_processing_stress() {
    println!("⚡ Testing Parallel Processing Stress");
    
    let start_time = Instant::now();
    
    // Create multiple sequences for parallel processing
    let num_sequences = 50;
    let sequence_length = 1000;
    
    let sequences: Vec<Vec<String>> = (0..num_sequences)
        .map(|seq_id| {
            (0..sequence_length)
                .map(|i| format!("S{}_{}", seq_id % 5, i % 10))
                .collect()
        })
        .collect();
    
    let batch_start = Instant::now();
    let results = batch_process_sequences(&sequences, 3, 0.05);
    let batch_time = batch_start.elapsed();
    
    let total_time = start_time.elapsed();
    
    // Verify all sequences were processed
    assert_eq!(results.len(), sequences.len());
    
    let total_anomalies: usize = results.iter().map(|r| r.len()).sum();
    
    println!("Parallel Processing Results:");
    println!("  Number of Sequences: {}", num_sequences);
    println!("  Sequence Length: {}", sequence_length);
    println!("  Batch Processing Time: {:?}", batch_time);
    println!("  Total Time: {:?}", total_time);
    println!("  Total Anomalies: {}", total_anomalies);
    println!("  Average Time per Sequence: {:?}", 
             batch_time / num_sequences as u32);
    
    // Performance requirements
    let avg_time_per_sequence = batch_time.as_millis() / num_sequences as u128;
    assert!(
        avg_time_per_sequence < 1000,
        "Average time per sequence should be < 1s: {}ms",
        avg_time_per_sequence
    );
    
    // Validate all results
    for (seq_idx, anomaly_set) in results.iter().enumerate() {
        for anomaly in anomaly_set {
            assert!(
                anomaly.likelihood.is_finite(),
                "Sequence {}: likelihood should be finite",
                seq_idx
            );
            assert!(
                anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0,
                "Sequence {}: anomaly strength out of bounds: {}",
                seq_idx, anomaly.anomaly_strength
            );
        }
    }
    
    println!("✅ Parallel processing stress test passed");
}

#[test]
fn test_memory_pressure_stress() {
    println!("💾 Testing Memory Pressure Stress");
    
    let start_time = Instant::now();
    
    // Test with conditions that create many contexts
    let alphabet_size = 20;
    let max_order = 4;
    let sequence_length = 20000;
    
    // Create sequence that maximizes context diversity
    let sequence: Vec<String> = (0..sequence_length)
        .map(|i| {
            // Create patterns that generate many different contexts
            let base = i % alphabet_size;
            let variation = (i / alphabet_size) % 3;
            format!("S{}_{}", base, variation)
        })
        .collect();
    
    let mut detector = AnomalyDetector::new(max_order);
    
    let train_start = Instant::now();
    detector.train(&sequence).unwrap();
    let train_time = train_start.elapsed();
    
    let context_tree = detector.model().context_tree();
    let context_count = context_tree.context_count();
    
    // Test detection under memory pressure
    let detect_start = Instant::now();
    let anomalies = detector.detect_anomalies(&sequence, 0.01);
    let detect_time = detect_start.elapsed();
    
    let total_time = start_time.elapsed();
    
    println!("Memory Pressure Results:");
    println!("  Alphabet Size: {}", alphabet_size);
    println!("  Max Order: {}", max_order);
    println!("  Sequence Length: {}", sequence_length);
    println!("  Contexts Created: {}", context_count);
    println!("  Training Time: {:?}", train_time);
    println!("  Detection Time: {:?}", detect_time);
    println!("  Total Time: {:?}", total_time);
    println!("  Anomalies Detected: {}", anomalies.len());
    
    // Calculate memory efficiency
    let unique_states: HashSet<String> = sequence.into_iter().collect();
    let actual_alphabet_size = unique_states.len();
    let theoretical_max = (1..=max_order)
        .map(|order| actual_alphabet_size.pow(order as u32))
        .sum::<usize>();
    
    let memory_efficiency = context_count as f64 / theoretical_max as f64;
    
    println!("  Actual Alphabet Size: {}", actual_alphabet_size);
    println!("  Theoretical Max Contexts: {}", theoretical_max);
    println!("  Memory Efficiency: {:.2}%", memory_efficiency * 100.0);
    
    // Memory should be used efficiently
    assert!(
        memory_efficiency < 0.9,
        "Memory efficiency should be reasonable: {:.2}%",
        memory_efficiency * 100.0
    );
    
    // All results should be valid
    for anomaly in &anomalies {
        assert!(anomaly.likelihood.is_finite());
        assert!(anomaly.information_score.is_finite());
        assert!(anomaly.anomaly_strength.is_finite());
    }
    
    println!("✅ Memory pressure stress test passed");
}

#[test]
fn test_edge_case_stress() {
    println!("🎯 Testing Edge Case Stress");
    
    // Test 1: Single repeated element
    println!("  Testing single repeated element...");
    let repeated_sequence: Vec<String> = vec!["X"; 10000].into_iter().map(String::from).collect();
    let mut detector1 = AnomalyDetector::new(3);
    detector1.train(&repeated_sequence).unwrap();
    
    let anomalies1 = detector1.detect_anomalies(&repeated_sequence, 0.1);
    println!("    Anomalies in repeated sequence: {}", anomalies1.len());
    
    // Test 2: Alternating pattern
    println!("  Testing alternating pattern...");
    let alternating: Vec<String> = (0..10000)
        .map(|i| if i % 2 == 0 { "A" } else { "B" }.to_string())
        .collect();
    let mut detector2 = AnomalyDetector::new(3);
    detector2.train(&alternating).unwrap();
    
    let anomalies2 = detector2.detect_anomalies(&alternating, 0.1);
    println!("    Anomalies in alternating pattern: {}", anomalies2.len());
    
    // Test 3: Random-like sequence
    println!("  Testing pseudo-random sequence...");
    let pseudo_random: Vec<String> = (0..10000)
        .map(|i: usize| {
            let hash = (i.wrapping_mul(1103515245).wrapping_add(12345)) % 100;
            format!("R{}", hash % 10)
        })
        .collect();
    let mut detector3 = AnomalyDetector::new(3);
    detector3.train(&pseudo_random).unwrap();
    
    let anomalies3 = detector3.detect_anomalies(&pseudo_random, 0.1);
    println!("    Anomalies in pseudo-random sequence: {}", anomalies3.len());
    
    // Test 4: Sequence with rare events
    println!("  Testing sequence with rare events...");
    let mut rare_events = vec!["COMMON"; 9900].into_iter().map(String::from).collect::<Vec<_>>();
    for i in 0..100 {
        rare_events.push(format!("RARE_{}", i));
    }
    let mut detector4 = AnomalyDetector::new(2);
    detector4.train(&rare_events).unwrap();
    
    let test_rare: Vec<String> = vec!["RARE_999", "ULTRA_RARE", "NEVER_SEEN"]
        .into_iter().map(String::from).collect();
    let anomalies4 = detector4.detect_anomalies(&test_rare, 0.5);
    println!("    Anomalies in rare events test: {}", anomalies4.len());
    
    // All tests should complete without panicking
    println!("✅ Edge case stress tests passed");
}

#[test]
fn test_scalability_analysis() {
    println!("📈 Testing Scalability Analysis");
    
    let sizes = vec![100, 500, 1000, 2000];
    let orders = vec![2, 3, 4];
    
    println!("Scalability Results:");
    println!("{:>8} {:>8} {:>12} {:>12} {:>10} {:>10}", 
             "Size", "Order", "Train(ms)", "Detect(ms)", "Contexts", "Anomalies");
    println!("{}", "-".repeat(70));
    
    for &size in &sizes {
        for &order in &orders {
            let _start_time = Instant::now();
            
            // Generate test sequence
            let states = vec!["A", "B", "C", "D", "E"];
            let sequence: Vec<String> = (0..size)
                .map(|i| states[(i * 7 + i * i) % states.len()].to_string())
                .collect();
            
            let mut detector = AnomalyDetector::new(order);
            
            let train_start = Instant::now();
            detector.train(&sequence).unwrap();
            let train_time = train_start.elapsed();
            
            let context_count = detector.model().context_tree().context_count();
            
            let detect_start = Instant::now();
            let anomalies = detector.detect_anomalies(&sequence, 0.01);
            let detect_time = detect_start.elapsed();
            
            println!("{:>8} {:>8} {:>12} {:>12} {:>10} {:>10}", 
                     size, order, 
                     train_time.as_millis(), 
                     detect_time.as_millis(),
                     context_count,
                     anomalies.len());
            
            // Performance assertions
            assert!(
                train_time.as_millis() < (size as u128 * order as u128),
                "Training time should scale reasonably: {}ms for size={}, order={}",
                train_time.as_millis(), size, order
            );
            
            assert!(
                detect_time.as_millis() < size as u128,
                "Detection time should scale linearly: {}ms for size={}",
                detect_time.as_millis(), size
            );
            
            // Memory efficiency
            let memory_efficiency = context_count as f64 / sequence.len() as f64;
            assert!(
                memory_efficiency < 1.0,
                "Memory efficiency should be < 1.0: {:.3} for size={}, order={}",
                memory_efficiency, size, order
            );
        }
    }
    
    println!("✅ Scalability analysis completed");
}

#[test]
fn test_numerical_precision_limits() {
    println!("🔬 Testing Numerical Precision Limits");
    
    // Test with very small probabilities
    let mut sequence = Vec::new();
    
    // Create highly skewed distribution
    for _ in 0..9999 {
        sequence.push("COMMON".to_string());
    }
    sequence.push("RARE".to_string());
    
    let mut detector = AnomalyDetector::new(2);
    detector.train(&sequence).unwrap();
    
    // Test with the rare event
    let rare_test: Vec<String> = vec!["COMMON", "RARE", "COMMON"]
        .into_iter().map(String::from).collect();
    
    let anomalies = detector.detect_anomalies(&rare_test, 1e-15);
    
    for anomaly in &anomalies {
        // Test numerical stability with extreme values
        assert!(
            anomaly.likelihood.is_finite(),
            "Likelihood should be finite even for rare events: {}",
            anomaly.likelihood
        );
        
        assert!(
            anomaly.log_likelihood.is_finite(),
            "Log-likelihood should be finite: {}",
            anomaly.log_likelihood
        );
        
        assert!(
            anomaly.information_score.is_finite(),
            "Information score should be finite: {}",
            anomaly.information_score
        );
        
        // Test that very small probabilities don't cause underflow
        if anomaly.likelihood > 0.0 {
            let log_likelihood_check = anomaly.likelihood.ln();
            assert!(
                log_likelihood_check.is_finite(),
                "Log of likelihood should be finite: ln({}) = {}",
                anomaly.likelihood, log_likelihood_check
            );
        }
    }
    
    println!("✅ Numerical precision limits test passed");
}