//! Performance tests for memory optimization.
//! These tests ensure the library uses memory efficiently.

#![allow(clippy::uninlined_format_args)]

use anomaly_grid::*;
use std::time::Instant;

#[test]
fn test_memory_usage_scaling_with_data_size() {
    println!("Testing memory usage scaling with data size...");

    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");

    let data_sizes = vec![100, 500, 1000, 2000, 5000];
    let mut memory_usages = Vec::new();

    for &size in &data_sizes {
        let training_data = generate_diverse_training_data(size, 10);

        let mut detector =
            AnomalyDetector::with_config(config.clone()).expect("Detector creation should succeed");

        detector
            .train(&training_data)
            .expect("Training should succeed");

        let metrics = detector.performance_metrics();
        let memory_usage = metrics.estimated_memory_bytes;
        memory_usages.push(memory_usage);

        println!(
            "Data size: {}, Memory usage: {} KB, Contexts: {}",
            size,
            memory_usage / 1024,
            metrics.context_count
        );

        // Validate reasonable memory usage
        assert!(
            memory_usage < 100 * 1024 * 1024, // 100MB limit
            "Memory usage {} bytes exceeds 100MB limit for size {}",
            memory_usage,
            size
        );

        // Memory efficiency check
        let memory_per_element = memory_usage as f64 / size as f64;
        assert!(
            memory_per_element < 10240.0, // 10KB per element max
            "Memory per element {} bytes exceeds 10KB for size {}",
            memory_per_element,
            size
        );
    }

    // Validate memory scaling characteristics
    for i in 1..memory_usages.len() {
        let size_ratio = data_sizes[i] as f64 / data_sizes[i - 1] as f64;
        let memory_ratio = memory_usages[i] as f64 / memory_usages[i - 1] as f64;

        println!(
            "Size ratio: {:.2}, Memory ratio: {:.2}",
            size_ratio, memory_ratio
        );

        // Memory should not grow faster than quadratic with data size
        assert!(
            memory_ratio <= size_ratio * size_ratio * 1.5,
            "Memory growth {} exceeds quadratic bound for size ratio {}",
            memory_ratio,
            size_ratio
        );
    }

    println!("✅ Memory usage scaling validation passed");
}

#[test]
fn test_memory_efficiency_with_alphabet_size() {
    println!("Testing memory efficiency with different alphabet sizes...");

    let alphabet_sizes = vec![5, 10, 20, 50, 100];
    let training_size = 1000;
    let max_order = 3;

    for &alphabet_size in &alphabet_sizes {
        let config = AnomalyGridConfig::default()
            .with_max_order(max_order)
            .expect("Valid config");

        let training_data = generate_diverse_training_data(training_size, alphabet_size);

        let mut detector =
            AnomalyDetector::with_config(config).expect("Detector creation should succeed");

        detector
            .train(&training_data)
            .expect("Training should succeed");

        let metrics = detector.performance_metrics();
        let memory_usage = metrics.estimated_memory_bytes;
        let context_count = metrics.context_count;

        println!(
            "Alphabet size: {}, Memory: {} KB, Contexts: {}, Efficiency: {:.1}%",
            alphabet_size,
            memory_usage / 1024,
            context_count,
            calculate_memory_efficiency(alphabet_size, max_order, context_count)
        );

        // Validate memory efficiency
        let theoretical_max_contexts = calculate_theoretical_max_contexts(alphabet_size, max_order);
        let efficiency = context_count as f64 / theoretical_max_contexts as f64;

        // Efficiency should be reasonable (not using all theoretical contexts)
        assert!(
            efficiency < 1.0,
            "Memory efficiency {} too high (possible memory waste) for alphabet {}",
            efficiency,
            alphabet_size
        );

        // Memory per context should be reasonable
        if context_count > 0 {
            let memory_per_context = memory_usage / context_count;
            assert!(
                memory_per_context < 5120, // 5KB per context max
                "Memory per context {} bytes exceeds 5KB for alphabet {}",
                memory_per_context,
                alphabet_size
            );
        }
    }

    println!("✅ Memory efficiency with alphabet size validation passed");
}

#[test]
fn test_memory_optimization_with_max_order() {
    println!("Testing memory optimization with different maximum orders...");

    let max_orders = vec![1, 2, 3, 4, 5, 6];
    let training_size = 1500;
    let alphabet_size = 15;

    let mut previous_memory = 0;

    for &max_order in &max_orders {
        let config = AnomalyGridConfig::default()
            .with_max_order(max_order)
            .expect("Valid config");

        let training_data = generate_diverse_training_data(training_size, alphabet_size);

        let mut detector =
            AnomalyDetector::with_config(config).expect("Detector creation should succeed");

        detector
            .train(&training_data)
            .expect("Training should succeed");

        let metrics = detector.performance_metrics();
        let memory_usage = metrics.estimated_memory_bytes;
        let context_count = metrics.context_count;

        println!(
            "Max order: {}, Memory: {} KB, Contexts: {}",
            max_order,
            memory_usage / 1024,
            context_count
        );

        // Validate memory growth with order
        if max_order > 1 && previous_memory > 0 {
            let memory_growth = memory_usage as f64 / previous_memory as f64;

            // Memory growth should be reasonable (not exponential)
            assert!(
                memory_growth < 20.0,
                "Memory growth {} too large for max order {}",
                memory_growth,
                max_order
            );
        }

        // Validate theoretical bounds
        let theoretical_max = calculate_theoretical_max_contexts(alphabet_size, max_order);
        assert!(
            context_count <= theoretical_max,
            "Context count {} exceeds theoretical maximum {} for order {}",
            context_count,
            theoretical_max,
            max_order
        );

        previous_memory = memory_usage;
    }

    println!("✅ Memory optimization with max order validation passed");
}

#[test]
fn test_memory_stability_during_operations() {
    println!("Testing memory stability during extended operations...");

    let config = AnomalyGridConfig::default()
        .with_max_order(4)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Train the detector
    let training_data = generate_diverse_training_data(2000, 12);
    detector
        .train(&training_data)
        .expect("Training should succeed");

    let initial_metrics = detector.performance_metrics();
    let initial_memory = initial_metrics.estimated_memory_bytes;
    let initial_contexts = initial_metrics.context_count;

    println!(
        "Initial state - Memory: {} KB, Contexts: {}",
        initial_memory / 1024,
        initial_contexts
    );

    // Perform extended detection operations
    let operations = 2000;
    let test_sequence = generate_test_sequence(25, 12);

    for i in 0..operations {
        let _ = detector
            .detect_anomalies(&test_sequence, 0.1)
            .expect("Detection should succeed");

        // Check memory stability periodically
        if i % 200 == 199 {
            let current_metrics = detector.performance_metrics();
            let current_memory = current_metrics.estimated_memory_bytes;
            let current_contexts = current_metrics.context_count;

            println!(
                "After {} operations - Memory: {} KB, Contexts: {}",
                i + 1,
                current_memory / 1024,
                current_contexts
            );

            // Memory should remain stable
            let memory_change = (current_memory as f64 / initial_memory as f64 - 1.0).abs();
            assert!(
                memory_change < 0.05,
                "Memory change {} exceeds 5% after {} operations",
                memory_change,
                i + 1
            );

            // Context count should remain stable
            assert_eq!(
                current_contexts,
                initial_contexts,
                "Context count changed from {} to {} after {} operations",
                initial_contexts,
                current_contexts,
                i + 1
            );
        }
    }

    let final_metrics = detector.performance_metrics();
    let final_memory = final_metrics.estimated_memory_bytes;
    let final_contexts = final_metrics.context_count;

    println!(
        "Final state - Memory: {} KB, Contexts: {}",
        final_memory / 1024,
        final_contexts
    );

    // Final validation
    let total_memory_change = (final_memory as f64 / initial_memory as f64 - 1.0).abs();
    assert!(
        total_memory_change < 0.02,
        "Total memory change {} exceeds 2% after {} operations",
        total_memory_change,
        operations
    );

    assert_eq!(
        final_contexts, initial_contexts,
        "Final context count {} differs from initial {}",
        final_contexts, initial_contexts
    );

    println!("✅ Memory stability during operations validation passed");
}

#[test]
fn test_memory_optimization_with_configuration() {
    println!("Testing memory optimization with different configurations...");

    let configurations = vec![
        (
            "Low memory",
            AnomalyGridConfig::default()
                .with_max_order(2)
                .unwrap()
                .with_smoothing_alpha(1.0)
                .unwrap(),
        ),
        (
            "Balanced",
            AnomalyGridConfig::default()
                .with_max_order(3)
                .unwrap()
                .with_smoothing_alpha(0.5)
                .unwrap(),
        ),
        (
            "High accuracy",
            AnomalyGridConfig::default()
                .with_max_order(5)
                .unwrap()
                .with_smoothing_alpha(0.1)
                .unwrap(),
        ),
    ];

    let training_size = 1000;
    let alphabet_size = 12;

    for (config_name, config) in configurations {
        let training_data = generate_diverse_training_data(training_size, alphabet_size);

        let mut detector =
            AnomalyDetector::with_config(config).expect("Detector creation should succeed");

        detector
            .train(&training_data)
            .expect("Training should succeed");

        let metrics = detector.performance_metrics();
        let memory_usage = metrics.estimated_memory_bytes;
        let context_count = metrics.context_count;

        println!(
            "Configuration '{}': Memory: {} KB, Contexts: {}",
            config_name,
            memory_usage / 1024,
            context_count
        );

        // Validate configuration-specific expectations
        match config_name {
            "Low memory" => {
                assert!(
                    memory_usage < 200 * 1024, // 200KB limit for low memory
                    "Low memory config uses {} bytes, exceeds 200KB limit",
                    memory_usage
                );
            }
            "Balanced" => {
                assert!(
                    memory_usage < 500 * 1024, // 500KB limit for balanced
                    "Balanced config uses {} bytes, exceeds 500KB limit",
                    memory_usage
                );
            }
            "High accuracy" => {
                assert!(
                    memory_usage < 2 * 1024 * 1024, // 2MB limit for high accuracy
                    "High accuracy config uses {} bytes, exceeds 2MB limit",
                    memory_usage
                );
            }
            _ => {}
        }

        // All configurations should have reasonable memory per context
        if context_count > 0 {
            let memory_per_context = memory_usage / context_count;
            assert!(
                memory_per_context < 8192, // 8KB per context max
                "Config '{}' memory per context {} bytes exceeds 8KB",
                config_name,
                memory_per_context
            );
        }
    }

    println!("✅ Memory optimization with configuration validation passed");
}

#[test]
fn test_memory_fragmentation_resistance() {
    println!("Testing memory fragmentation resistance...");

    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");

    // Test with multiple detector instances to simulate fragmentation
    let num_detectors = 10;
    let mut detectors = Vec::new();
    let mut total_memory = 0;

    for i in 0..num_detectors {
        let training_data = generate_diverse_training_data(500, 8);

        let mut detector =
            AnomalyDetector::with_config(config.clone()).expect("Detector creation should succeed");

        detector
            .train(&training_data)
            .expect("Training should succeed");

        let metrics = detector.performance_metrics();
        let memory_usage = metrics.estimated_memory_bytes;
        total_memory += memory_usage;

        println!("Detector {}: Memory: {} KB", i + 1, memory_usage / 1024);

        // Each detector should use reasonable memory
        assert!(
            memory_usage < 200 * 1024, // 200KB per detector max
            "Detector {} uses {} bytes, exceeds 200KB limit",
            i + 1,
            memory_usage
        );

        detectors.push(detector);
    }

    println!(
        "Total memory for {} detectors: {} KB",
        num_detectors,
        total_memory / 1024
    );

    // Total memory should be reasonable
    assert!(
        total_memory < 2 * 1024 * 1024, // 2MB total limit
        "Total memory {} bytes exceeds 2MB limit for {} detectors",
        total_memory,
        num_detectors
    );

    // Test operations on all detectors
    let test_sequence = generate_test_sequence(20, 8);

    for (i, detector) in detectors.iter().enumerate() {
        let start_time = Instant::now();
        let _ = detector
            .detect_anomalies(&test_sequence, 0.1)
            .expect("Detection should succeed");
        let detection_time = start_time.elapsed();

        // Performance should remain good despite potential fragmentation
        assert!(
            detection_time.as_micros() < 1000,
            "Detector {} detection time {} μs exceeds 1000 μs (fragmentation impact?)",
            i + 1,
            detection_time.as_micros()
        );
    }

    println!("✅ Memory fragmentation resistance validation passed");
}

#[test]
fn test_memory_efficiency_metrics() {
    println!("Testing memory efficiency metrics...");

    let test_cases = vec![
        (2, 5, 500),   // Small alphabet, low order
        (3, 10, 1000), // Medium alphabet, medium order
        (4, 15, 1500), // Large alphabet, high order
        (5, 20, 2000), // Very large alphabet, very high order
    ];

    for (max_order, alphabet_size, training_size) in test_cases {
        let config = AnomalyGridConfig::default()
            .with_max_order(max_order)
            .expect("Valid config");

        let training_data = generate_diverse_training_data(training_size, alphabet_size);

        let mut detector =
            AnomalyDetector::with_config(config).expect("Detector creation should succeed");

        detector
            .train(&training_data)
            .expect("Training should succeed");

        let metrics = detector.performance_metrics();
        let memory_usage = metrics.estimated_memory_bytes;
        let context_count = metrics.context_count;

        // Calculate efficiency metrics
        let theoretical_max = calculate_theoretical_max_contexts(alphabet_size, max_order);
        let context_efficiency = context_count as f64 / theoretical_max as f64;
        let memory_per_context = if context_count > 0 {
            memory_usage / context_count
        } else {
            0
        };
        let memory_per_element = memory_usage as f64 / training_size as f64;

        println!(
            "Order: {}, Alphabet: {}, Training: {}",
            max_order, alphabet_size, training_size
        );
        println!(
            "  Memory: {} KB, Contexts: {}/{} ({:.1}%)",
            memory_usage / 1024,
            context_count,
            theoretical_max,
            context_efficiency * 100.0
        );
        println!(
            "  Memory/context: {} bytes, Memory/element: {:.1} bytes",
            memory_per_context, memory_per_element
        );

        // Validate efficiency metrics
        assert!(
            context_efficiency < 1.0,
            "Context efficiency {} too high (potential waste) for order {} alphabet {}",
            context_efficiency,
            max_order,
            alphabet_size
        );

        assert!(
            memory_per_context < 4096,
            "Memory per context {} bytes exceeds 4KB for order {} alphabet {}",
            memory_per_context,
            max_order,
            alphabet_size
        );

        assert!(
            memory_per_element < 2048.0,
            "Memory per element {} bytes exceeds 2KB for order {} alphabet {}",
            memory_per_element,
            max_order,
            alphabet_size
        );
    }

    println!("✅ Memory efficiency metrics validation passed");
}

/// Generate diverse training data to test memory optimization
fn generate_diverse_training_data(size: usize, alphabet_size: usize) -> Vec<String> {
    let mut data = Vec::new();

    // Create alphabet
    let alphabet: Vec<String> = (0..alphabet_size).map(|i| format!("STATE_{}", i)).collect();

    // Generate diverse patterns to create realistic memory usage
    for i in 0..size {
        let state_index = match i % 7 {
            0 => i % alphabet_size,                            // Sequential
            1 => (i / 2) % alphabet_size,                      // Slower sequential
            2 => (i * 3) % alphabet_size,                      // Skip pattern
            3 => (i * i) % alphabet_size,                      // Quadratic
            4 => (i * 7 + 3) % alphabet_size,                  // Linear congruential
            5 => ((i as f64).sqrt() as usize) % alphabet_size, // Square root
            _ => (i.reverse_bits() % 1000) % alphabet_size,    // Bit reversal
        };

        data.push(alphabet[state_index].clone());
    }

    data
}

/// Generate test sequence for memory testing
fn generate_test_sequence(length: usize, alphabet_size: usize) -> Vec<String> {
    let mut sequence = Vec::new();

    let alphabet: Vec<String> = (0..alphabet_size).map(|i| format!("STATE_{}", i)).collect();

    for i in 0..length {
        let state_index = (i * 5 + 2) % alphabet_size;
        sequence.push(alphabet[state_index].clone());
    }

    sequence
}

/// Calculate theoretical maximum number of contexts
fn calculate_theoretical_max_contexts(alphabet_size: usize, max_order: usize) -> usize {
    (0..=max_order)
        .map(|order| alphabet_size.pow(order as u32))
        .sum()
}

/// Calculate memory efficiency percentage
fn calculate_memory_efficiency(
    alphabet_size: usize,
    max_order: usize,
    actual_contexts: usize,
) -> f64 {
    let theoretical_max = calculate_theoretical_max_contexts(alphabet_size, max_order);
    (actual_contexts as f64 / theoretical_max as f64) * 100.0
}
