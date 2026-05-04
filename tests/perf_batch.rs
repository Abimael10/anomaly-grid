//! Performance tests for batch processing.
//! These tests ensure the library handles batch operations efficiently.

#![allow(clippy::uninlined_format_args)]

use anomaly_grid::*;
use std::time::Instant;

#[test]
fn test_batch_processing_throughput() {
    println!("Testing batch processing throughput...");

    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Train the detector
    let training_data = generate_batch_training_data(2000, 12);
    detector
        .train(&training_data)
        .expect("Training should succeed");

    let batch_sizes = vec![10, 50, 100, 500, 1000, 2000];
    let sequence_length = 25;

    for &batch_size in &batch_sizes {
        // Generate batch of sequences
        let batch_sequences: Vec<Vec<String>> = (0..batch_size)
            .map(|i| generate_batch_test_sequence(sequence_length, 12, i))
            .collect();

        // Measure batch processing throughput
        let start_time = Instant::now();

        for sequence in &batch_sequences {
            let _ = detector
                .detect_anomalies(sequence, 0.1)
                .expect("Detection should succeed");
        }

        let total_time = start_time.elapsed();
        let throughput = batch_size as f64 / total_time.as_secs_f64();
        let avg_latency = total_time / batch_size as u32;

        println!(
            "Batch size: {}, Throughput: {:.0} seq/sec, Avg latency: {:?}",
            batch_size, throughput, avg_latency
        );

        // Validate throughput requirements
        assert!(
            throughput >= 500.0,
            "Batch throughput {} seq/sec below 500 threshold for batch size {}",
            throughput,
            batch_size
        );

        // Validate latency requirements
        assert!(
            avg_latency.as_millis() < 10,
            "Average latency {} ms exceeds 10 ms threshold for batch size {}",
            avg_latency.as_millis(),
            batch_size
        );

        // For smaller batches, performance should be higher
        if batch_size <= 100 {
            assert!(
                throughput >= 1000.0,
                "Small batch throughput {} seq/sec below 1000 threshold for size {}",
                throughput,
                batch_size
            );
        }
    }

    println!("✅ Batch processing throughput validation passed");
}

#[test]
fn test_batch_processing_scalability() {
    println!("Testing batch processing scalability...");

    let config = AnomalyGridConfig::default()
        .with_max_order(4)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Train the detector
    let training_data = generate_batch_training_data(1500, 15);
    detector
        .train(&training_data)
        .expect("Training should succeed");

    let batch_sizes = vec![100, 200, 500, 1000, 2000];
    let sequence_length = 30;
    let mut processing_times = Vec::new();

    for &batch_size in &batch_sizes {
        let batch_sequences: Vec<Vec<String>> = (0..batch_size)
            .map(|i| generate_batch_test_sequence(sequence_length, 15, i))
            .collect();

        // Warm-up pass (unmeasured) to stabilize caches/branch prediction
        for _ in 0..3 {
            for sequence in &batch_sequences {
                let _ = detector
                    .detect_anomalies(sequence, 0.1)
                    .expect("Detection should succeed");
            }
        }

        // Measure multiple times and take the best to reduce variance spikes
        let repeats = 5;
        let mut best_time = std::time::Duration::MAX;

        for _ in 0..repeats {
            let start_time = Instant::now();
            for sequence in &batch_sequences {
                let _ = detector
                    .detect_anomalies(sequence, 0.1)
                    .expect("Detection should succeed");
            }
            let elapsed = start_time.elapsed();
            if elapsed < best_time {
                best_time = elapsed;
            }
        }

        let total_time = best_time;
        processing_times.push(total_time);

        let throughput = batch_size as f64 / total_time.as_secs_f64();

        println!(
            "Batch size: {}, Processing time: {:?}, Throughput: {:.0} seq/sec",
            batch_size, total_time, throughput
        );

        // Validate scalability
        assert!(
            total_time.as_secs() < 10,
            "Processing time {} sec exceeds 10 sec limit for batch size {}",
            total_time.as_secs(),
            batch_size
        );
    }

    // Validate that processing time scales reasonably with batch size
    for i in 1..processing_times.len() {
        let size_ratio = batch_sizes[i] as f64 / batch_sizes[i - 1] as f64;
        let time_ratio =
            processing_times[i].as_nanos() as f64 / processing_times[i - 1].as_nanos() as f64;

        println!(
            "Size ratio: {:.2}, Time ratio: {:.2}",
            size_ratio, time_ratio
        );

        // Time should scale roughly linearly with batch size
        assert!(
            time_ratio <= size_ratio * 1.5,
            "Time scaling {} exceeds linear bound for size ratio {}",
            time_ratio,
            size_ratio
        );
    }

    println!("✅ Batch processing scalability validation passed");
}

#[test]
fn test_batch_processing_memory_efficiency() {
    println!("Testing batch processing memory efficiency...");

    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Train the detector
    let training_data = generate_batch_training_data(1000, 10);
    detector
        .train(&training_data)
        .expect("Training should succeed");

    let initial_metrics = detector.performance_metrics();
    let initial_memory = initial_metrics.estimated_memory_bytes;

    println!("Initial memory usage: {} KB", initial_memory / 1024);

    // Process large batches and monitor memory
    let batch_sizes = vec![100, 500, 1000, 2000];
    let sequence_length = 20;

    for &batch_size in &batch_sizes {
        let batch_sequences: Vec<Vec<String>> = (0..batch_size)
            .map(|i| generate_batch_test_sequence(sequence_length, 10, i))
            .collect();

        // Process batch
        for sequence in &batch_sequences {
            let _ = detector
                .detect_anomalies(sequence, 0.1)
                .expect("Detection should succeed");
        }

        let current_metrics = detector.performance_metrics();
        let current_memory = current_metrics.estimated_memory_bytes;

        println!(
            "After batch size {}: Memory usage: {} KB",
            batch_size,
            current_memory / 1024
        );

        // Memory should remain stable during batch processing
        let memory_growth = current_memory as f64 / initial_memory as f64;
        assert!(
            memory_growth < 1.1,
            "Memory growth {} exceeds 10% after batch size {}",
            memory_growth,
            batch_size
        );
    }

    let final_metrics = detector.performance_metrics();
    let final_memory = final_metrics.estimated_memory_bytes;

    println!("Final memory usage: {} KB", final_memory / 1024);

    // Final memory should be close to initial
    let total_growth = final_memory as f64 / initial_memory as f64;
    assert!(
        total_growth < 1.05,
        "Total memory growth {} exceeds 5% after all batch processing",
        total_growth
    );

    println!("✅ Batch processing memory efficiency validation passed");
}

#[test]
fn test_batch_processing_with_different_sequence_lengths() {
    println!("Testing batch processing with different sequence lengths...");

    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Train the detector
    let training_data = generate_batch_training_data(1500, 12);
    detector
        .train(&training_data)
        .expect("Training should succeed");

    let sequence_lengths = vec![10, 25, 50, 100, 200];
    let batch_size = 200;

    for &seq_length in &sequence_lengths {
        let batch_sequences: Vec<Vec<String>> = (0..batch_size)
            .map(|i| generate_batch_test_sequence(seq_length, 12, i))
            .collect();

        let start_time = Instant::now();

        for sequence in &batch_sequences {
            let _ = detector
                .detect_anomalies(sequence, 0.1)
                .expect("Detection should succeed");
        }

        let total_time = start_time.elapsed();
        let throughput = batch_size as f64 / total_time.as_secs_f64();
        let avg_time_per_element = total_time.as_nanos() as f64 / (batch_size * seq_length) as f64;

        println!(
            "Sequence length: {}, Throughput: {:.0} seq/sec, Time/element: {:.0} ns",
            seq_length, throughput, avg_time_per_element
        );

        // Validate performance scales with sequence length
        assert!(
            throughput >= 100.0,
            "Throughput {} seq/sec below 100 threshold for length {}",
            throughput,
            seq_length
        );

        // Time per element should be reasonable
        assert!(
            avg_time_per_element < 50000.0, // 50μs per element max
            "Time per element {} ns exceeds 50μs for length {}",
            avg_time_per_element,
            seq_length
        );
    }

    println!("✅ Batch processing with different sequence lengths validation passed");
}

#[test]
fn test_batch_processing_concurrent_simulation() {
    println!("Testing batch processing concurrent simulation...");

    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Train the detector
    let training_data = generate_batch_training_data(1000, 10);
    detector
        .train(&training_data)
        .expect("Training should succeed");

    // Simulate concurrent workload with rapid sequential processing
    let num_workers = 4;
    let sequences_per_worker = 100;
    let sequence_length = 20;

    let total_sequences = num_workers * sequences_per_worker;
    let all_sequences: Vec<Vec<String>> = (0..total_sequences)
        .map(|i| generate_batch_test_sequence(sequence_length, 10, i))
        .collect();

    // Simulate concurrent processing by rapid sequential execution
    let start_time = Instant::now();

    for sequence in &all_sequences {
        let _ = detector
            .detect_anomalies(sequence, 0.1)
            .expect("Detection should succeed");
    }

    let total_time = start_time.elapsed();
    let throughput = total_sequences as f64 / total_time.as_secs_f64();
    let avg_latency = total_time / total_sequences as u32;

    println!(
        "Concurrent simulation: {} workers, {} seq/worker",
        num_workers, sequences_per_worker
    );
    println!(
        "Total sequences: {}, Total time: {:?}",
        total_sequences, total_time
    );
    println!(
        "Throughput: {:.0} seq/sec, Avg latency: {:?}",
        throughput, avg_latency
    );

    // Validate concurrent performance
    assert!(
        throughput >= 1000.0,
        "Concurrent throughput {} seq/sec below 1000 threshold",
        throughput
    );

    assert!(
        avg_latency.as_micros() < 2000,
        "Average concurrent latency {} μs exceeds 2000 μs threshold",
        avg_latency.as_micros()
    );

    println!("✅ Batch processing concurrent simulation validation passed");
}

#[test]
fn test_batch_processing_with_mixed_workloads() {
    println!("Testing batch processing with mixed workloads...");

    let config = AnomalyGridConfig::default()
        .with_max_order(4)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Train the detector
    let training_data = generate_batch_training_data(1500, 15);
    detector
        .train(&training_data)
        .expect("Training should succeed");

    // Create mixed workload with different sequence characteristics
    let workload_types = vec![
        ("Short sequences", 10, 100),
        ("Medium sequences", 30, 100),
        ("Long sequences", 100, 50),
        ("Very long sequences", 200, 25),
    ];

    for (workload_name, seq_length, batch_size) in workload_types {
        let batch_sequences: Vec<Vec<String>> = (0..batch_size)
            .map(|i| generate_batch_test_sequence(seq_length, 15, i))
            .collect();

        let start_time = Instant::now();

        for sequence in &batch_sequences {
            let _ = detector
                .detect_anomalies(sequence, 0.1)
                .expect("Detection should succeed");
        }

        let total_time = start_time.elapsed();
        let throughput = batch_size as f64 / total_time.as_secs_f64();
        let elements_per_sec = (batch_size * seq_length) as f64 / total_time.as_secs_f64();

        println!(
            "Workload '{}': {} seq/sec, {} elements/sec",
            workload_name, throughput as u32, elements_per_sec as u32
        );

        // Validate performance for each workload type
        assert!(
            throughput >= 50.0,
            "Workload '{}' throughput {} seq/sec below 50 threshold",
            workload_name,
            throughput
        );

        assert!(
            elements_per_sec >= 1000.0,
            "Workload '{}' element throughput {} elements/sec below 1000 threshold",
            workload_name,
            elements_per_sec
        );
    }

    println!("✅ Batch processing with mixed workloads validation passed");
}

#[test]
fn test_batch_processing_performance_consistency() {
    println!("Testing batch processing performance consistency...");

    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Train the detector
    let training_data = generate_batch_training_data(1000, 10);
    detector
        .train(&training_data)
        .expect("Training should succeed");

    // Run multiple batch processing rounds to test consistency
    let rounds = 10;
    let batch_size = 200;
    let sequence_length = 25;
    let mut throughputs = Vec::new();

    // Prepare a fixed batch once to remove workload variance between rounds
    let batch_sequences: Vec<Vec<String>> = (0..batch_size)
        .map(|i| generate_batch_test_sequence(sequence_length, 10, i + 42)) // fixed offset seed
        .collect();

    // Warm up caches/predictors on the fixed batch
    for _ in 0..10 {
        for sequence in &batch_sequences {
            let _ = detector
                .detect_anomalies(sequence, 0.1)
                .expect("Detection should succeed");
        }
    }

    for round in 0..rounds {
        let iterations_per_round = 50;
        let mut best_throughput = 0.0;

        // Run three measurements per round and keep the best throughput to reduce jitter impact
        for _ in 0..3 {
            let start_time = Instant::now();

            for _ in 0..iterations_per_round {
                for sequence in &batch_sequences {
                    let _ = detector
                        .detect_anomalies(sequence, 0.1)
                        .expect("Detection should succeed");
                }
            }

            let total_time = start_time.elapsed();
            let total_sequences = (batch_size * iterations_per_round) as f64;
            let throughput = total_sequences / total_time.as_secs_f64();
            if throughput > best_throughput {
                best_throughput = throughput;
            }
        }

        throughputs.push(best_throughput);

        println!(
            "Round {}: Throughput: {:.0} seq/sec",
            round + 1,
            best_throughput
        );

        // Each round should meet minimum performance
        assert!(
            best_throughput >= 1000.0,
            "Round {} throughput {} seq/sec below 1000 threshold",
            round + 1,
            best_throughput
        );
    }

    // Calculate performance consistency metrics
    let avg_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    let variance = throughputs
        .iter()
        .map(|x| (x - avg_throughput).powi(2))
        .sum::<f64>()
        / throughputs.len() as f64;
    let std_dev = variance.sqrt();
    let coefficient_of_variation = std_dev / avg_throughput;

    println!("Performance consistency:");
    println!("  Average throughput: {:.0} seq/sec", avg_throughput);
    println!("  Standard deviation: {:.0} seq/sec", std_dev);
    println!(
        "  Coefficient of variation: {:.3}",
        coefficient_of_variation
    );

    // Validate performance consistency
    assert!(
        coefficient_of_variation < 0.2,
        "Performance inconsistency: CV {} exceeds 0.2 threshold",
        coefficient_of_variation
    );

    assert!(
        avg_throughput >= 1500.0,
        "Average throughput {} seq/sec below 1500 threshold",
        avg_throughput
    );

    println!("✅ Batch processing performance consistency validation passed");
}

#[test]
fn test_batch_processing_resource_utilization() {
    println!("Testing batch processing resource utilization...");

    let config = AnomalyGridConfig::default()
        .with_max_order(4)
        .expect("Valid config");

    let mut detector =
        AnomalyDetector::with_config(config).expect("Detector creation should succeed");

    // Train the detector
    let training_data = generate_batch_training_data(2000, 12);
    detector
        .train(&training_data)
        .expect("Training should succeed");

    let initial_metrics = detector.performance_metrics();

    // Test resource utilization under sustained load
    let sustained_batches = 20;
    let batch_size = 100;
    let sequence_length = 30;

    let overall_start = Instant::now();

    for batch_num in 0..sustained_batches {
        let batch_sequences: Vec<Vec<String>> = (0..batch_size)
            .map(|i| generate_batch_test_sequence(sequence_length, 12, i + batch_num * batch_size))
            .collect();

        let batch_start = Instant::now();

        for sequence in &batch_sequences {
            let _ = detector
                .detect_anomalies(sequence, 0.1)
                .expect("Detection should succeed");
        }

        let batch_time = batch_start.elapsed();
        let batch_throughput = batch_size as f64 / batch_time.as_secs_f64();

        if batch_num % 5 == 4 {
            let current_metrics = detector.performance_metrics();
            println!(
                "Batch {}: Throughput: {:.0} seq/sec, Memory: {} KB",
                batch_num + 1,
                batch_throughput,
                current_metrics.estimated_memory_bytes / 1024
            );

            // Resource utilization should remain stable
            let memory_ratio = current_metrics.estimated_memory_bytes as f64
                / initial_metrics.estimated_memory_bytes as f64;
            assert!(
                memory_ratio < 1.1,
                "Memory utilization grew {} times after batch {}",
                memory_ratio,
                batch_num + 1
            );
        }

        // Each batch should maintain good performance
        assert!(
            batch_throughput >= 800.0,
            "Batch {} throughput {} seq/sec below 800 threshold",
            batch_num + 1,
            batch_throughput
        );
    }

    let overall_time = overall_start.elapsed();
    let total_sequences = sustained_batches * batch_size;
    let overall_throughput = total_sequences as f64 / overall_time.as_secs_f64();

    println!("Sustained load results:");
    println!("  Total sequences: {}", total_sequences);
    println!("  Total time: {:?}", overall_time);
    println!("  Overall throughput: {:.0} seq/sec", overall_throughput);

    // Validate sustained performance
    assert!(
        overall_throughput >= 1000.0,
        "Sustained throughput {} seq/sec below 1000 threshold",
        overall_throughput
    );

    let final_metrics = detector.performance_metrics();
    let final_memory_ratio =
        final_metrics.estimated_memory_bytes as f64 / initial_metrics.estimated_memory_bytes as f64;
    assert!(
        final_memory_ratio < 1.05,
        "Final memory utilization grew {} times during sustained load",
        final_memory_ratio
    );

    println!("✅ Batch processing resource utilization validation passed");
}

/// Generate training data for batch processing tests
fn generate_batch_training_data(size: usize, alphabet_size: usize) -> Vec<String> {
    let mut data = Vec::new();

    let alphabet: Vec<String> = (0..alphabet_size)
        .map(|i| format!("BATCH_STATE_{}", i))
        .collect();

    // Generate diverse patterns for realistic batch processing
    for i in 0..size {
        let state_index = match i % 6 {
            0 => i % alphabet_size,                         // Sequential
            1 => (i / 3) % alphabet_size,                   // Slower sequential
            2 => (i * 5) % alphabet_size,                   // Skip pattern
            3 => (i * i) % alphabet_size,                   // Quadratic
            4 => (i * 7 + 11) % alphabet_size,              // Linear congruential
            _ => (i.count_ones() as usize) % alphabet_size, // Bit count
        };

        data.push(alphabet[state_index].clone());
    }

    data
}

/// Generate test sequence for batch processing
fn generate_batch_test_sequence(length: usize, alphabet_size: usize, seed: usize) -> Vec<String> {
    let mut sequence = Vec::new();

    let alphabet: Vec<String> = (0..alphabet_size)
        .map(|i| format!("BATCH_STATE_{}", i))
        .collect();

    // Generate sequence with seed-based variation
    for i in 0..length {
        let state_index = ((i + seed) * 13 + 7) % alphabet_size;
        sequence.push(alphabet[state_index].clone());
    }

    sequence
}
