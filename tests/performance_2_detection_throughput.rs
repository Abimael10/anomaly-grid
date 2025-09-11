//! Performance Test 2: Detection Throughput
//!
//! This test validates the detection performance characteristics of the anomaly detection
//! library, measuring throughput, latency, and scalability of the detection operations
//! under various conditions and workloads.

use anomaly_grid::*;
use std::time::Instant;

#[test]
fn test_detection_latency_single_sequence() {
    println!("Testing detection latency for single sequences...");
    
    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");
    
    let mut detector = AnomalyDetector::with_config(config)
        .expect("Detector creation should succeed");
    
    // Train the detector
    let training_data = generate_training_data(1000, 10);
    detector.train(&training_data).expect("Training should succeed");
    
    let sequence_lengths = vec![5, 10, 20, 50, 100];
    
    for &length in &sequence_lengths {
        let test_sequence = generate_test_sequence(length, 10);
        
        // Warm up
        for _ in 0..10 {
            let _ = detector.detect_anomalies(&test_sequence, 0.1);
        }
        
        // Measure latency
        let iterations = 1000;
        let start_time = Instant::now();
        
        for _ in 0..iterations {
            let _ = detector.detect_anomalies(&test_sequence, 0.1)
                .expect("Detection should succeed");
        }
        
        let total_time = start_time.elapsed();
        let avg_latency = total_time / iterations;
        
        println!("Sequence length: {}, Average latency: {:?}", length, avg_latency);
        
        // Validate latency requirements
        assert!(avg_latency.as_micros() < 5000, // 5ms threshold
               "Average latency {} μs exceeds 5000 μs threshold for length {}",
               avg_latency.as_micros(), length);
        
        // For short sequences, latency should be very low
        if length <= 20 {
            assert!(avg_latency.as_micros() < 500, // 500μs threshold
                   "Average latency {} μs exceeds 500 μs threshold for short length {}",
                   avg_latency.as_micros(), length);
        }
    }
    
    println!("✅ Detection latency validation passed");
}

#[test]
fn test_detection_throughput_batch_processing() {
    println!("Testing detection throughput for batch processing...");
    
    let config = AnomalyGridConfig::default()
        .with_max_order(4)
        .expect("Valid config");
    
    let mut detector = AnomalyDetector::with_config(config)
        .expect("Detector creation should succeed");
    
    // Train the detector
    let training_data = generate_training_data(2000, 15);
    detector.train(&training_data).expect("Training should succeed");
    
    let batch_sizes = vec![10, 50, 100, 500, 1000];
    let sequence_length = 20;
    
    for &batch_size in &batch_sizes {
        let test_sequences: Vec<Vec<String>> = (0..batch_size)
            .map(|_| generate_test_sequence(sequence_length, 15))
            .collect();
        
        // Warm up
        for _ in 0..5 {
            for sequence in &test_sequences {
                let _ = detector.detect_anomalies(sequence, 0.1);
            }
        }
        
        // Measure throughput
        let start_time = Instant::now();
        
        for sequence in &test_sequences {
            let _ = detector.detect_anomalies(sequence, 0.1)
                .expect("Detection should succeed");
        }
        
        let total_time = start_time.elapsed();
        let throughput = batch_size as f64 / total_time.as_secs_f64();
        
        println!("Batch size: {}, Throughput: {:.0} detections/second", batch_size, throughput);
        
        // Validate minimum throughput
        assert!(throughput >= 1000.0,
               "Throughput {} detections/sec below 1000 threshold for batch size {}",
               throughput, batch_size);
        
        // For smaller batches, throughput should be higher due to less overhead
        if batch_size <= 100 {
            assert!(throughput >= 1000.0,
                   "Small batch throughput {} seq/sec below 1000 threshold for size {}",
                   throughput, batch_size);
        }
    }
    
    println!("✅ Detection throughput validation passed");
}

#[test]
fn test_detection_scalability_with_sequence_length() {
    println!("Testing detection scalability with sequence length...");
    
    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");
    
    let mut detector = AnomalyDetector::with_config(config)
        .expect("Detector creation should succeed");
    
    // Train the detector
    let training_data = generate_training_data(1500, 12);
    detector.train(&training_data).expect("Training should succeed");
    
    let sequence_lengths = vec![10, 25, 50, 100, 200, 500];
    let mut detection_times = Vec::new();
    
    for &length in &sequence_lengths {
        let test_sequence = generate_test_sequence(length, 12);
        
        // Measure detection time
        let iterations = 100;
        let start_time = Instant::now();
        
        for _ in 0..iterations {
            let _ = detector.detect_anomalies(&test_sequence, 0.1)
                .expect("Detection should succeed");
        }
        
        let total_time = start_time.elapsed();
        let avg_time = total_time / iterations;
        detection_times.push(avg_time);
        
        println!("Sequence length: {}, Average detection time: {:?}", length, avg_time);
        
        // Validate that detection time is reasonable
        assert!(avg_time.as_millis() < 20,
               "Detection time {} ms exceeds 20 ms threshold for length {}",
               avg_time.as_millis(), length);
    }
    
    // Validate that detection time scales reasonably with sequence length
    for i in 1..detection_times.len() {
        let length_ratio = sequence_lengths[i] as f64 / sequence_lengths[i-1] as f64;
        let time_ratio = detection_times[i].as_nanos() as f64 / detection_times[i-1].as_nanos() as f64;
        
        // Time growth should be roughly linear with sequence length
        assert!(time_ratio <= length_ratio * 2.0,
               "Detection time growth {} exceeds linear bound for length ratio {}",
               time_ratio, length_ratio);
    }
    
    println!("✅ Detection scalability validation passed");
}

#[test]
fn test_detection_performance_with_different_thresholds() {
    println!("Testing detection performance with different thresholds...");
    
    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");
    
    let mut detector = AnomalyDetector::with_config(config)
        .expect("Detector creation should succeed");
    
    // Train the detector
    let training_data = generate_training_data(1000, 10);
    detector.train(&training_data).expect("Training should succeed");
    
    let test_sequence = generate_test_sequence(50, 10);
    let thresholds = vec![0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8];
    
    for &threshold in &thresholds {
        // Measure detection time with this threshold
        let iterations = 500;
        let start_time = Instant::now();
        
        for _ in 0..iterations {
            let _ = detector.detect_anomalies(&test_sequence, threshold)
                .expect("Detection should succeed");
        }
        
        let total_time = start_time.elapsed();
        let avg_time = total_time / iterations;
        
        println!("Threshold: {}, Average detection time: {:?}", threshold, avg_time);
        
        // Validate that threshold doesn't significantly affect performance
        assert!(avg_time.as_micros() < 2000,
               "Detection time {} μs exceeds 2000 μs threshold for threshold {}",
               avg_time.as_micros(), threshold);
    }
    
    println!("✅ Detection performance with thresholds validation passed");
}

#[test]
fn test_detection_memory_stability() {
    println!("Testing detection memory stability...");
    
    let config = AnomalyGridConfig::default()
        .with_max_order(4)
        .expect("Valid config");
    
    let mut detector = AnomalyDetector::with_config(config)
        .expect("Detector creation should succeed");
    
    // Train the detector
    let training_data = generate_training_data(1500, 12);
    detector.train(&training_data).expect("Training should succeed");
    
    let initial_metrics = detector.performance_metrics();
    let initial_memory = initial_metrics.estimated_memory_bytes;
    
    println!("Initial memory usage: {} KB", initial_memory / 1024);
    
    // Perform many detection operations
    let iterations = 1000;
    let test_sequence = generate_test_sequence(30, 12);
    
    for i in 0..iterations {
        let _ = detector.detect_anomalies(&test_sequence, 0.1)
            .expect("Detection should succeed");
        
        // Check memory periodically
        if i % 100 == 99 {
            let current_metrics = detector.performance_metrics();
            let current_memory = current_metrics.estimated_memory_bytes;
            
            println!("After {} detections, memory usage: {} KB", 
                    i + 1, current_memory / 1024);
            
            // Memory should remain stable (no significant growth)
            let memory_growth = current_memory as f64 / initial_memory as f64;
            assert!(memory_growth < 1.1,
                   "Memory growth {} exceeds 10% after {} detections",
                   memory_growth, i + 1);
        }
    }
    
    let final_metrics = detector.performance_metrics();
    let final_memory = final_metrics.estimated_memory_bytes;
    
    println!("Final memory usage: {} KB", final_memory / 1024);
    
    // Final memory should be very close to initial memory
    let total_growth = final_memory as f64 / initial_memory as f64;
    assert!(total_growth < 1.05,
           "Total memory growth {} exceeds 5% after {} detections",
           total_growth, iterations);
    
    println!("✅ Detection memory stability validation passed");
}

#[test]
fn test_detection_concurrent_performance() {
    println!("Testing detection concurrent performance simulation...");
    
    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");
    
    let mut detector = AnomalyDetector::with_config(config)
        .expect("Detector creation should succeed");
    
    // Train the detector
    let training_data = generate_training_data(1000, 10);
    detector.train(&training_data).expect("Training should succeed");
    
    // Simulate concurrent workload by rapid sequential processing
    let concurrent_sequences = 100;
    let sequence_length = 25;
    
    let test_sequences: Vec<Vec<String>> = (0..concurrent_sequences)
        .map(|_| generate_test_sequence(sequence_length, 10))
        .collect();
    
    // Measure performance under concurrent-like load
    let start_time = Instant::now();
    
    for sequence in &test_sequences {
        let _ = detector.detect_anomalies(sequence, 0.1)
            .expect("Detection should succeed");
    }
    
    let total_time = start_time.elapsed();
    let throughput = concurrent_sequences as f64 / total_time.as_secs_f64();
    let avg_latency = total_time / concurrent_sequences;
    
    println!("Concurrent sequences: {}, Total time: {:?}", concurrent_sequences, total_time);
    println!("Throughput: {:.0} detections/second", throughput);
    println!("Average latency: {:?}", avg_latency);
    
    // Validate concurrent performance
    assert!(throughput >= 2000.0,
           "Concurrent throughput {} detections/sec below 2000 threshold",
           throughput);
    
    assert!(avg_latency.as_micros() < 1000,
           "Average concurrent latency {} μs exceeds 1000 μs threshold",
           avg_latency.as_micros());
    
    println!("✅ Detection concurrent performance validation passed");
}

#[test]
fn test_detection_performance_with_different_orders() {
    println!("Testing detection performance with different maximum orders...");
    
    let max_orders = vec![1, 2, 3, 4, 5, 6];
    let sequence_length = 40;
    let alphabet_size = 12;
    
    for &max_order in &max_orders {
        let config = AnomalyGridConfig::default()
            .with_max_order(max_order)
            .expect("Valid config");
        
        let mut detector = AnomalyDetector::with_config(config)
            .expect("Detector creation should succeed");
        
        // Train the detector
        let training_data = generate_training_data(1000, alphabet_size);
        detector.train(&training_data).expect("Training should succeed");
        
        let test_sequence = generate_test_sequence(sequence_length, alphabet_size);
        
        // Measure detection performance
        let iterations = 200;
        let start_time = Instant::now();
        
        for _ in 0..iterations {
            let _ = detector.detect_anomalies(&test_sequence, 0.1)
                .expect("Detection should succeed");
        }
        
        let total_time = start_time.elapsed();
        let avg_time = total_time / iterations;
        let throughput = iterations as f64 / total_time.as_secs_f64();
        
        println!("Max order: {}, Average time: {:?}, Throughput: {:.0}/sec", 
                max_order, avg_time, throughput);
        
        // Validate that performance remains reasonable with higher orders
        assert!(avg_time.as_millis() < 5,
               "Detection time {} ms exceeds 5 ms threshold for max order {}",
               avg_time.as_millis(), max_order);
        
        assert!(throughput >= 100.0,
               "Throughput {} detections/sec below 100 threshold for max order {}",
               throughput, max_order);
    }
    
    println!("✅ Detection performance with different orders validation passed");
}

#[test]
fn test_detection_performance_edge_cases() {
    println!("Testing detection performance with edge cases...");
    
    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");
    
    let mut detector = AnomalyDetector::with_config(config)
        .expect("Detector creation should succeed");
    
    // Train the detector
    let training_data = generate_training_data(1000, 10);
    detector.train(&training_data).expect("Training should succeed");
    
    let edge_cases = vec![
        ("Empty sequence", vec![]),
        ("Single element", vec!["STATE_0".to_string()]),
        ("Two elements", vec!["STATE_0".to_string(), "STATE_1".to_string()]),
        ("Repeated element", vec!["STATE_0".to_string(); 20]),
        ("Unknown elements", vec!["UNKNOWN_1".to_string(), "UNKNOWN_2".to_string()]),
    ];
    
    for (case_name, test_sequence) in edge_cases {
        // Measure detection time for edge case
        let iterations = 1000;
        let start_time = Instant::now();
        
        for _ in 0..iterations {
            let _ = detector.detect_anomalies(&test_sequence, 0.1)
                .expect("Detection should succeed");
        }
        
        let total_time = start_time.elapsed();
        let avg_time = total_time / iterations;
        
        println!("Edge case '{}': Average time: {:?}", case_name, avg_time);
        
        // Edge cases should be handled reasonably quickly
        assert!(avg_time.as_micros() < 1000,
               "Edge case '{}' detection time {} μs exceeds 1000 μs threshold",
               case_name, avg_time.as_micros());
    }
    
    println!("✅ Detection performance edge cases validation passed");
}

/// Generate training data with specified size and alphabet size
fn generate_training_data(size: usize, alphabet_size: usize) -> Vec<String> {
    let mut data = Vec::new();
    
    // Create alphabet
    let alphabet: Vec<String> = (0..alphabet_size)
        .map(|i| format!("STATE_{}", i))
        .collect();
    
    // Generate realistic patterns
    for i in 0..size {
        let state_index = match i % 5 {
            0 => i % alphabet_size,                    // Sequential pattern
            1 => (i / 2) % alphabet_size,              // Slower sequential
            2 => (i * 3) % alphabet_size,              // Skip pattern
            3 => (i * i) % alphabet_size,              // Quadratic pattern
            _ => (i * 7 + 3) % alphabet_size,          // Linear congruential
        };
        
        data.push(alphabet[state_index].clone());
    }
    
    data
}

/// Generate test sequence with specified length and alphabet size
fn generate_test_sequence(length: usize, alphabet_size: usize) -> Vec<String> {
    let mut sequence = Vec::new();
    
    // Create alphabet
    let alphabet: Vec<String> = (0..alphabet_size)
        .map(|i| format!("STATE_{}", i))
        .collect();
    
    // Generate test sequence with some anomalous patterns
    for i in 0..length {
        let state_index = if i % 10 == 7 {
            // Inject some anomalous patterns
            (i * 13 + 7) % alphabet_size
        } else {
            // Normal patterns
            (i * 2) % alphabet_size
        };
        
        sequence.push(alphabet[state_index].clone());
    }
    
    sequence
}