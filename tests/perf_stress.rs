//! Performance tests for stress testing.
//! These tests ensure the library handles extreme conditions.

#![allow(clippy::uninlined_format_args)]

use anomaly_grid::*;
use std::time::Instant;

#[test]
fn test_high_volume_stress_testing() {
    println!("Testing high volume stress conditions...");

    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Train with large dataset
    let large_training_data = generate_stress_training_data(10000, 20);
    println!("Training with {} elements...", large_training_data.len());

    let train_start = Instant::now();
    detector
        .train(&large_training_data)
        .expect("Training should succeed");
    let train_time = train_start.elapsed();

    let metrics = detector.performance_metrics();
    println!("Training completed in {:?}", train_time);
    println!(
        "Contexts learned: {}, Memory: {} KB",
        metrics.context_count,
        metrics.estimated_memory_bytes / 1024
    );

    // Validate training performance under stress
    assert!(
        train_time.as_secs() < 30,
        "Training time {} sec exceeds 30 sec limit for large dataset",
        train_time.as_secs()
    );

    assert!(
        metrics.estimated_memory_bytes < 100 * 1024 * 1024, // 100MB limit
        "Memory usage {} bytes exceeds 100MB limit",
        metrics.estimated_memory_bytes
    );

    // High volume detection stress test
    let stress_sequences = 5000;
    let sequence_length = 50;

    println!(
        "Performing high volume detection: {} sequences...",
        stress_sequences
    );

    let detection_start = Instant::now();
    let mut total_anomalies = 0;

    for i in 0..stress_sequences {
        let test_sequence = generate_stress_test_sequence(sequence_length, 20, i);
        let anomalies = detector
            .detect_anomalies(&test_sequence, 0.1)
            .expect("Detection should succeed");
        total_anomalies += anomalies.len();

        // Progress reporting
        if i % 1000 == 999 {
            let elapsed = detection_start.elapsed();
            let throughput = (i + 1) as f64 / elapsed.as_secs_f64();
            println!(
                "  Processed {} sequences, throughput: {:.0} seq/sec",
                i + 1,
                throughput
            );
        }
    }

    let total_detection_time = detection_start.elapsed();
    let overall_throughput = stress_sequences as f64 / total_detection_time.as_secs_f64();

    println!("High volume stress test completed:");
    println!("  Total sequences: {}", stress_sequences);
    println!("  Total time: {:?}", total_detection_time);
    println!("  Throughput: {:.0} seq/sec", overall_throughput);
    println!("  Total anomalies found: {}", total_anomalies);

    // Validate stress test performance
    assert!(
        overall_throughput >= 200.0,
        "Stress test throughput {} seq/sec below 200 threshold",
        overall_throughput
    );

    assert!(
        total_detection_time.as_secs() < 60,
        "Stress test time {} sec exceeds 60 sec limit",
        total_detection_time.as_secs()
    );

    println!("✅ High volume stress testing validation passed");
}

#[test]
fn test_sustained_load_stress_testing() {
    println!("Testing sustained load stress conditions...");

    let config = AnomalyGridConfig::default()
        .with_max_order(4)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Train the detector
    let training_data = generate_stress_training_data(3000, 15);
    detector
        .train(&training_data)
        .expect("Training should succeed");

    let initial_metrics = detector.performance_metrics();
    let initial_memory = initial_metrics.estimated_memory_bytes;

    // Sustained load test parameters
    let duration_seconds = 10; // 10 seconds of sustained load
    let target_throughput = 50; // sequences per second
    let sequence_length = 30;

    println!(
        "Running sustained load test for {} seconds at {} seq/sec...",
        duration_seconds, target_throughput
    );

    let test_start = Instant::now();
    let test_duration = std::time::Duration::from_secs(duration_seconds);
    let mut sequences_processed = 0;
    let mut total_anomalies = 0;
    let mut performance_samples = Vec::new();

    while test_start.elapsed() < test_duration {
        let batch_start = Instant::now();
        let batch_size = 50; // Process in batches

        for i in 0..batch_size {
            let test_sequence =
                generate_stress_test_sequence(sequence_length, 15, sequences_processed + i);
            let anomalies = detector
                .detect_anomalies(&test_sequence, 0.1)
                .expect("Detection should succeed");
            total_anomalies += anomalies.len();
        }

        let batch_time = batch_start.elapsed();
        let batch_throughput = batch_size as f64 / batch_time.as_secs_f64();
        performance_samples.push(batch_throughput);

        sequences_processed += batch_size;

        // Monitor performance every 5 seconds
        if sequences_processed % (target_throughput * 5) == 0 && sequences_processed > 0 {
            let elapsed = test_start.elapsed();
            let current_throughput = sequences_processed as f64 / elapsed.as_secs_f64();
            let current_metrics = detector.performance_metrics();

            println!(
                "  {} sec: {} sequences, {:.0} seq/sec, {} KB memory",
                elapsed.as_secs(),
                sequences_processed,
                current_throughput,
                current_metrics.estimated_memory_bytes / 1024
            );

            // Validate sustained performance
            assert!(
                current_throughput >= target_throughput as f64 * 0.8,
                "Sustained throughput {} seq/sec below 80% of target {}",
                current_throughput,
                target_throughput
            );

            // Validate memory stability
            let memory_growth =
                current_metrics.estimated_memory_bytes as f64 / initial_memory as f64;
            assert!(
                memory_growth < 1.2,
                "Memory growth {} exceeds 20% during sustained load",
                memory_growth
            );
        }

        // No artificial throttling — the loop body's work itself sets the
        // pace, and the assertion above already ensures we honour memory
        // bounds. (v0.6: removed `thread::sleep(10ms)` which made the
        // test time-sensitive on slow CI runners.)
    }

    let total_time = test_start.elapsed();
    let final_throughput = sequences_processed as f64 / total_time.as_secs_f64();

    // Calculate performance statistics
    let avg_batch_throughput =
        performance_samples.iter().sum::<f64>() / performance_samples.len() as f64;
    let min_batch_throughput = performance_samples
        .iter()
        .fold(f64::INFINITY, |a, &b| a.min(b));
    let max_batch_throughput = performance_samples.iter().fold(0.0f64, |a, &b| a.max(b));

    println!("Sustained load test completed:");
    println!("  Duration: {:?}", total_time);
    println!("  Sequences processed: {}", sequences_processed);
    println!("  Overall throughput: {:.0} seq/sec", final_throughput);
    println!(
        "  Batch throughput - Avg: {:.0}, Min: {:.0}, Max: {:.0}",
        avg_batch_throughput, min_batch_throughput, max_batch_throughput
    );
    println!("  Total anomalies: {}", total_anomalies);

    // Final validation
    assert!(
        final_throughput >= target_throughput as f64 * 0.9,
        "Final throughput {} seq/sec below 90% of target {}",
        final_throughput,
        target_throughput
    );

    assert!(
        min_batch_throughput >= target_throughput as f64 * 0.5,
        "Minimum batch throughput {} seq/sec too low",
        min_batch_throughput
    );

    let final_metrics = detector.performance_metrics();
    let final_memory_growth = final_metrics.estimated_memory_bytes as f64 / initial_memory as f64;
    assert!(
        final_memory_growth < 1.1,
        "Final memory growth {} exceeds 10%",
        final_memory_growth
    );

    println!("✅ Sustained load stress testing validation passed");
}

#[test]
fn test_extreme_configuration_stress() {
    println!("Testing extreme configuration stress conditions...");

    let extreme_configs = vec![
        (
            "High order",
            AnomalyGridConfig::default().with_max_order(8).unwrap(),
        ),
        (
            "Large alphabet",
            AnomalyGridConfig::default().with_max_order(3).unwrap(),
        ),
        (
            "Low smoothing",
            AnomalyGridConfig::default()
                .with_max_order(4)
                .unwrap()
                .with_smoothing_alpha(0.01)
                .unwrap(),
        ),
        (
            "High smoothing",
            AnomalyGridConfig::default()
                .with_max_order(3)
                .unwrap()
                .with_smoothing_alpha(10.0)
                .unwrap(),
        ),
    ];

    for (config_name, config) in extreme_configs {
        println!("\nTesting extreme configuration: {}", config_name);

        let mut detector =
            AnomalyDetector::with_config(config).expect("Detector creation should succeed");

        // Determine appropriate parameters for this configuration
        let (training_size, alphabet_size) = match config_name {
            "High order" => (2000, 10),
            "Large alphabet" => (3000, 50),
            "Low smoothing" => (1500, 15),
            "High smoothing" => (1000, 12),
            _ => (1000, 10),
        };

        let training_data = generate_stress_training_data(training_size, alphabet_size);

        // Test training under extreme conditions
        let train_start = Instant::now();
        detector
            .train(&training_data)
            .expect("Training should succeed");
        let train_time = train_start.elapsed();

        let metrics = detector.performance_metrics();

        println!("  Training time: {:?}", train_time);
        println!(
            "  Contexts: {}, Memory: {} KB",
            metrics.context_count,
            metrics.estimated_memory_bytes / 1024
        );

        // Validate extreme configuration performance
        assert!(
            train_time.as_secs() < 60,
            "Extreme config '{}' training time {} sec exceeds 60 sec",
            config_name,
            train_time.as_secs()
        );

        assert!(
            metrics.estimated_memory_bytes < 500 * 1024 * 1024, // 500MB limit
            "Extreme config '{}' memory {} bytes exceeds 500MB",
            config_name,
            metrics.estimated_memory_bytes
        );

        // Test detection under extreme conditions
        let test_sequences = 100;
        let sequence_length = 40;

        let detection_start = Instant::now();

        for i in 0..test_sequences {
            let test_sequence = generate_stress_test_sequence(sequence_length, alphabet_size, i);
            let _ = detector
                .detect_anomalies(&test_sequence, 0.1)
                .expect("Detection should succeed");
        }

        let detection_time = detection_start.elapsed();
        let detection_throughput = test_sequences as f64 / detection_time.as_secs_f64();

        println!(
            "  Detection throughput: {:.0} seq/sec",
            detection_throughput
        );

        // Validate detection performance under extreme conditions
        assert!(
            detection_throughput >= 10.0,
            "Extreme config '{}' detection throughput {} seq/sec below 10",
            config_name,
            detection_throughput
        );
    }

    println!("✅ Extreme configuration stress testing validation passed");
}

#[test]
fn test_memory_pressure_stress() {
    println!("Testing memory pressure stress conditions...");

    let config = AnomalyGridConfig::default()
        .with_max_order(5)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Create memory pressure with large, diverse training data
    let large_alphabet_size = 100;
    let large_training_size = 5000;

    println!(
        "Creating memory pressure with {} states and {} training elements...",
        large_alphabet_size, large_training_size
    );

    let training_data = generate_stress_training_data(large_training_size, large_alphabet_size);

    let train_start = Instant::now();
    detector
        .train(&training_data)
        .expect("Training should succeed");
    let train_time = train_start.elapsed();

    let metrics = detector.performance_metrics();
    let memory_usage = metrics.estimated_memory_bytes;

    println!("Memory pressure training completed:");
    println!("  Training time: {:?}", train_time);
    println!("  Memory usage: {} MB", memory_usage / (1024 * 1024));
    println!("  Contexts learned: {}", metrics.context_count);

    // Validate memory pressure handling
    assert!(
        train_time.as_secs() < 120,
        "Memory pressure training time {} sec exceeds 120 sec",
        train_time.as_secs()
    );

    assert!(
        memory_usage < 1024 * 1024 * 1024, // 1GB limit
        "Memory pressure usage {} bytes exceeds 1GB",
        memory_usage
    );

    // Test detection under memory pressure
    let stress_detection_sequences = 500;
    let sequence_length = 60;

    println!("Testing detection under memory pressure...");

    let detection_start = Instant::now();
    let mut successful_detections = 0;

    for i in 0..stress_detection_sequences {
        let test_sequence = generate_stress_test_sequence(sequence_length, large_alphabet_size, i);

        match detector.detect_anomalies(&test_sequence, 0.1) {
            Ok(_) => successful_detections += 1,
            Err(e) => {
                println!("Detection failed at sequence {}: {}", i, e);
                // Allow some failures under extreme memory pressure
                if successful_detections < i / 2 {
                    panic!("Too many detection failures under memory pressure");
                }
            }
        }

        if i % 100 == 99 {
            let elapsed = detection_start.elapsed();
            let throughput = (i + 1) as f64 / elapsed.as_secs_f64();
            println!(
                "  Processed {} sequences, throughput: {:.0} seq/sec",
                i + 1,
                throughput
            );
        }
    }

    let detection_time = detection_start.elapsed();
    let success_rate = successful_detections as f64 / stress_detection_sequences as f64;
    let throughput = successful_detections as f64 / detection_time.as_secs_f64();

    println!("Memory pressure detection completed:");
    println!(
        "  Successful detections: {}/{} ({:.1}%)",
        successful_detections,
        stress_detection_sequences,
        success_rate * 100.0
    );
    println!("  Throughput: {:.0} seq/sec", throughput);

    // Validate performance under memory pressure
    assert!(
        success_rate >= 0.95,
        "Success rate {} below 95% under memory pressure",
        success_rate
    );

    assert!(
        throughput >= 50.0,
        "Throughput {} seq/sec below 50 under memory pressure",
        throughput
    );

    println!("✅ Memory pressure stress testing validation passed");
}

#[test]
fn test_edge_case_stress_combinations() {
    println!("Testing edge case stress combinations...");

    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Train with normal data
    let training_data = generate_stress_training_data(1000, 10);
    detector
        .train(&training_data)
        .expect("Training should succeed");

    let edge_case_combinations = vec![
        ("Empty sequences", vec![vec![]]),
        (
            "Single element sequences",
            vec![vec!["STATE_0".to_string()]],
        ),
        (
            "Very long sequences",
            vec![vec!["STATE_0".to_string(); 1000]],
        ),
        ("Highly repetitive", vec![vec!["REPEAT".to_string(); 100]]),
        (
            "Unknown elements",
            vec![vec!["UNKNOWN_1".to_string(), "UNKNOWN_2".to_string()]],
        ),
        (
            "Mixed edge cases",
            vec![
                vec![],
                vec!["SINGLE".to_string()],
                vec!["REPEAT".to_string(); 50],
                vec!["UNKNOWN".to_string(), "ELEMENTS".to_string()],
            ],
        ),
    ];

    for (case_name, test_sequences) in edge_case_combinations {
        println!("\nTesting edge case combination: {}", case_name);

        let start_time = Instant::now();
        let mut successful_detections = 0;
        let mut total_detections = 0;

        // Stress test with many repetitions of edge cases
        let repetitions = 100;

        for rep in 0..repetitions {
            for (seq_idx, sequence) in test_sequences.iter().enumerate() {
                total_detections += 1;

                match detector.detect_anomalies(sequence, 0.1) {
                    Ok(_) => successful_detections += 1,
                    Err(e) => {
                        println!("  Edge case failure at rep {}, seq {}: {}", rep, seq_idx, e);
                        // Some edge cases might legitimately fail
                    }
                }
            }
        }

        let total_time = start_time.elapsed();
        let success_rate = successful_detections as f64 / total_detections as f64;
        let throughput = successful_detections as f64 / total_time.as_secs_f64();

        println!(
            "  Results: {}/{} successful ({:.1}%), {:.0} seq/sec",
            successful_detections,
            total_detections,
            success_rate * 100.0,
            throughput
        );

        // Validate edge case handling
        assert!(
            success_rate >= 0.9,
            "Edge case '{}' success rate {} below 90%",
            case_name,
            success_rate
        );

        // Throughput threshold scales with sequence length: longer sequences
        // are inherently slower per-sequence.
        let max_seq_len = test_sequences.iter().map(|s| s.len()).max().unwrap_or(1);
        let min_throughput = if max_seq_len > 100 { 50.0 } else { 1000.0 };
        assert!(
            throughput >= min_throughput,
            "Edge case '{}' throughput {:.0} seq/sec below {:.0}",
            case_name,
            throughput,
            min_throughput
        );
    }

    println!("✅ Edge case stress combinations validation passed");
}

#[test]
fn test_concurrent_stress_simulation() {
    println!("Testing concurrent stress simulation...");

    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Train the detector
    let training_data = generate_stress_training_data(2000, 12);
    detector
        .train(&training_data)
        .expect("Training should succeed");

    // Simulate high concurrency with rapid sequential processing
    let simulated_workers = 8;
    let sequences_per_worker = 500;
    let sequence_length = 25;

    let total_sequences = simulated_workers * sequences_per_worker;

    println!(
        "Simulating {} concurrent workers with {} sequences each...",
        simulated_workers, sequences_per_worker
    );

    // Generate all sequences upfront
    let all_sequences: Vec<Vec<String>> = (0..total_sequences)
        .map(|i| generate_stress_test_sequence(sequence_length, 12, i))
        .collect();

    let start_time = Instant::now();
    let mut successful_detections = 0;
    let mut worker_times = Vec::new();

    // Simulate concurrent processing by processing worker batches rapidly
    for worker in 0..simulated_workers {
        let worker_start = Instant::now();
        let start_idx = worker * sequences_per_worker;
        let end_idx = start_idx + sequences_per_worker;

        for sequence in &all_sequences[start_idx..end_idx] {
            match detector.detect_anomalies(sequence, 0.1) {
                Ok(_) => successful_detections += 1,
                Err(_) => {
                    // Allow some failures under stress
                }
            }
        }

        let worker_time = worker_start.elapsed();
        worker_times.push(worker_time);

        let worker_throughput = sequences_per_worker as f64 / worker_time.as_secs_f64();
        println!(
            "  Worker {}: {:?}, {:.0} seq/sec",
            worker + 1,
            worker_time,
            worker_throughput
        );
    }

    let total_time = start_time.elapsed();
    let overall_throughput = successful_detections as f64 / total_time.as_secs_f64();
    let success_rate = successful_detections as f64 / total_sequences as f64;

    // Calculate worker performance statistics
    let avg_worker_time =
        worker_times.iter().sum::<std::time::Duration>() / worker_times.len() as u32;
    let max_worker_time = worker_times.iter().max().unwrap();
    let min_worker_time = worker_times.iter().min().unwrap();

    // Trim outliers for variance calculation to reduce scheduler jitter impact
    let trimmed_variance = if worker_times.len() > 2 {
        let mut times = worker_times.clone();
        times.sort();
        let trimmed_min = times[1];
        let trimmed_max = times[times.len() - 2];
        trimmed_max.as_nanos() as f64 / trimmed_min.as_nanos() as f64
    } else {
        max_worker_time.as_nanos() as f64 / min_worker_time.as_nanos() as f64
    };

    println!("Concurrent stress simulation results:");
    println!("  Total sequences: {}", total_sequences);
    println!(
        "  Successful: {} ({:.1}%)",
        successful_detections,
        success_rate * 100.0
    );
    println!("  Overall throughput: {:.0} seq/sec", overall_throughput);
    println!(
        "  Worker times - Avg: {:?}, Min: {:?}, Max: {:?}",
        avg_worker_time, min_worker_time, max_worker_time
    );

    // Validate concurrent stress performance
    assert!(
        success_rate >= 0.95,
        "Concurrent stress success rate {} below 95%",
        success_rate
    );

    assert!(
        overall_throughput >= 500.0,
        "Concurrent stress throughput {} seq/sec below 500",
        overall_throughput
    );

    // Worker performance should be reasonably consistent
    let time_variance = trimmed_variance;
    assert!(
        time_variance < 3.0,
        "Worker time variance {} too high (performance inconsistency)",
        time_variance
    );

    println!("✅ Concurrent stress simulation validation passed");
}

/// Generate training data for stress testing
fn generate_stress_training_data(size: usize, alphabet_size: usize) -> Vec<String> {
    let mut data = Vec::new();

    let alphabet: Vec<String> = (0..alphabet_size)
        .map(|i| format!("STRESS_STATE_{}", i))
        .collect();

    // Generate complex patterns for stress testing
    for i in 0..size {
        let state_index = match i % 8 {
            0 => i % alphabet_size,                            // Sequential
            1 => (i / 2) % alphabet_size,                      // Slower sequential
            2 => (i * 3) % alphabet_size,                      // Skip pattern
            3 => (i * i) % alphabet_size,                      // Quadratic
            4 => (i * 7 + 11) % alphabet_size,                 // Linear congruential
            5 => (i.count_ones() as usize) % alphabet_size,    // Bit count
            6 => ((i as f64).sqrt() as usize) % alphabet_size, // Square root
            _ => (i.reverse_bits() % 10000) % alphabet_size,   // Bit reversal
        };

        data.push(alphabet[state_index].clone());
    }

    data
}

/// Generate test sequence for stress testing
fn generate_stress_test_sequence(length: usize, alphabet_size: usize, seed: usize) -> Vec<String> {
    let mut sequence = Vec::new();

    let alphabet: Vec<String> = (0..alphabet_size)
        .map(|i| format!("STRESS_STATE_{}", i))
        .collect();

    // Generate sequence with complex seed-based patterns
    for i in 0..length {
        let state_index = match (i + seed) % 5 {
            0 => ((i + seed) * 17 + 23) % alphabet_size,
            1 => ((i + seed) * (i + seed)) % alphabet_size,
            2 => ((i + seed).count_ones() as usize + seed) % alphabet_size,
            3 => (((i + seed) as f64).sin().abs() * alphabet_size as f64) as usize % alphabet_size,
            _ => ((i + seed).reverse_bits() % 1000 + seed) % alphabet_size,
        };

        sequence.push(alphabet[state_index].clone());
    }

    sequence
}
