//! Performance Test 1: Training Scalability
//!
//! This test validates the training performance characteristics of the anomaly detection
//! library across different data sizes, alphabet sizes, and maximum orders. It ensures
//! that the training performance scales appropriately and meets production requirements.

use anomaly_grid::*;
use std::time::Instant;

#[test]
fn test_training_time_complexity_with_data_size() {
    println!("Testing training time complexity with varying data sizes...");
    
    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");
    
    let data_sizes = vec![100, 500, 1000, 2000, 5000];
    let mut training_times = Vec::new();
    
    for &size in &data_sizes {
        let training_data = generate_training_data(size, 10); // 10-state alphabet
        
        let mut detector = AnomalyDetector::with_config(config.clone())
            .expect("Detector creation should succeed");
        
        let start_time = Instant::now();
        detector.train(&training_data).expect("Training should succeed");
        let training_time = start_time.elapsed();
        
        training_times.push(training_time);
        
        println!("Data size: {}, Training time: {:?}", size, training_time);
        
        // Validate that training time is reasonable (should be sub-second for these sizes)
        assert!(training_time.as_millis() < 5000, 
               "Training time {} ms exceeds 5 second threshold for size {}", 
               training_time.as_millis(), size);
    }
    
    // Validate that training time grows reasonably with data size
    // Should be roughly linear or sub-quadratic
    for i in 1..training_times.len() {
        let size_ratio = data_sizes[i] as f64 / data_sizes[i-1] as f64;
        let time_ratio = training_times[i].as_nanos() as f64 / training_times[i-1].as_nanos() as f64;
        
        // Time growth should not exceed quadratic growth
        assert!(time_ratio <= size_ratio * size_ratio * 2.0,
               "Training time growth {} exceeds quadratic bound for size ratio {}",
               time_ratio, size_ratio);
    }
    
    println!("✅ Training time complexity validation passed");
}

#[test]
fn test_training_memory_efficiency() {
    println!("Testing training memory efficiency...");
    
    let alphabet_sizes = vec![5, 10, 20, 50];
    let max_orders = vec![2, 3, 4, 5];
    
    for &alphabet_size in &alphabet_sizes {
        for &max_order in &max_orders {
            let config = AnomalyGridConfig::default()
                .with_max_order(max_order)
                .expect("Valid config");
            
            let training_data = generate_training_data(1000, alphabet_size);
            
            let mut detector = AnomalyDetector::with_config(config)
                .expect("Detector creation should succeed");
            
            detector.train(&training_data).expect("Training should succeed");
            
            let metrics = detector.performance_metrics();
            let memory_usage = metrics.estimated_memory_bytes;
            let context_count = metrics.context_count;
            
            println!("Alphabet: {}, Order: {}, Contexts: {}, Memory: {} KB", 
                    alphabet_size, max_order, context_count, memory_usage / 1024);
            
            // Validate memory efficiency
            let theoretical_max_contexts = (0..=max_order)
                .map(|order| alphabet_size.pow(order as u32))
                .sum::<usize>();
            
            // Actual contexts should be much less than theoretical maximum
            assert!(context_count <= theoretical_max_contexts,
                   "Context count {} exceeds theoretical maximum {}",
                   context_count, theoretical_max_contexts);
            
            // Memory per context should be reasonable (less than 10KB per context)
            if context_count > 0 {
                let memory_per_context = memory_usage / context_count;
                assert!(memory_per_context < 10240,
                       "Memory per context {} bytes exceeds 10KB threshold",
                       memory_per_context);
            }
        }
    }
    
    println!("✅ Training memory efficiency validation passed");
}

#[test]
fn test_training_convergence_speed() {
    println!("Testing training convergence speed...");
    
    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");
    
    // Test convergence with different amounts of training data
    let convergence_sizes = vec![50, 100, 200, 500, 1000];
    
    for &size in &convergence_sizes {
        let training_data = generate_training_data(size, 8);
        
        let mut detector = AnomalyDetector::with_config(config.clone())
            .expect("Detector creation should succeed");
        
        let start_time = Instant::now();
        detector.train(&training_data).expect("Training should succeed");
        let training_time = start_time.elapsed();
        
        let metrics = detector.performance_metrics();
        let context_count = metrics.context_count;
        
        println!("Training size: {}, Contexts learned: {}, Time: {:?}", 
                size, context_count, training_time);
        
        // Validate that we learn contexts efficiently
        if size >= 100 {
            assert!(context_count > 0, "Should learn some contexts with {} training examples", size);
        }
        
        // Validate convergence speed (should be fast for small datasets)
        if size <= 200 {
            assert!(training_time.as_millis() < 100,
                   "Training time {} ms too slow for small dataset size {}",
                   training_time.as_millis(), size);
        }
    }
    
    println!("✅ Training convergence speed validation passed");
}

#[test]
fn test_training_with_different_alphabet_sizes() {
    println!("Testing training performance with different alphabet sizes...");
    
    let alphabet_sizes = vec![2, 5, 10, 20, 50, 100];
    let training_size = 1000;
    let max_order = 3;
    
    for &alphabet_size in &alphabet_sizes {
        let config = AnomalyGridConfig::default()
            .with_max_order(max_order)
            .expect("Valid config");
        
        let training_data = generate_training_data(training_size, alphabet_size);
        
        let mut detector = AnomalyDetector::with_config(config)
            .expect("Detector creation should succeed");
        
        let start_time = Instant::now();
        detector.train(&training_data).expect("Training should succeed");
        let training_time = start_time.elapsed();
        
        let metrics = detector.performance_metrics();
        
        println!("Alphabet size: {}, Training time: {:?}, Contexts: {}, Memory: {} KB",
                alphabet_size, training_time, metrics.context_count, 
                metrics.estimated_memory_bytes / 1024);
        
        // Validate that training time is reasonable
        assert!(training_time.as_millis() < 2000,
               "Training time {} ms exceeds 2 second threshold for alphabet size {}",
               training_time.as_millis(), alphabet_size);
        
        // Validate that context count grows reasonably with alphabet size
        if alphabet_size <= 20 {
            assert!(metrics.context_count > 0,
                   "Should learn contexts for alphabet size {}", alphabet_size);
        }
    }
    
    println!("✅ Training with different alphabet sizes validation passed");
}

#[test]
fn test_training_with_different_max_orders() {
    println!("Testing training performance with different maximum orders...");
    
    let max_orders = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let training_size = 1000;
    let alphabet_size = 10;
    
    let mut previous_time = std::time::Duration::from_nanos(0);
    
    for &max_order in &max_orders {
        let config = AnomalyGridConfig::default()
            .with_max_order(max_order)
            .expect("Valid config");
        
        let training_data = generate_training_data(training_size, alphabet_size);
        
        let mut detector = AnomalyDetector::with_config(config)
            .expect("Detector creation should succeed");
        
        let start_time = Instant::now();
        detector.train(&training_data).expect("Training should succeed");
        let training_time = start_time.elapsed();
        
        let metrics = detector.performance_metrics();
        
        println!("Max order: {}, Training time: {:?}, Contexts: {}, Memory: {} KB",
                max_order, training_time, metrics.context_count, 
                metrics.estimated_memory_bytes / 1024);
        
        // Validate that training time is reasonable
        assert!(training_time.as_millis() < 3000,
               "Training time {} ms exceeds 3 second threshold for max order {}",
               training_time.as_millis(), max_order);
        
        // Validate that training time doesn't grow too quickly with order
        if max_order > 1 && previous_time.as_nanos() > 0 {
            let time_ratio = training_time.as_nanos() as f64 / previous_time.as_nanos() as f64;
            assert!(time_ratio < 10.0,
                   "Training time growth {} too large for max order {}",
                   time_ratio, max_order);
        }
        
        previous_time = training_time;
    }
    
    println!("✅ Training with different max orders validation passed");
}

#[test]
fn test_training_throughput_measurement() {
    println!("Testing training throughput measurement...");
    
    let config = AnomalyGridConfig::default()
        .with_max_order(3)
        .expect("Valid config");
    
    let training_sizes = vec![500, 1000, 2000, 5000];
    
    for &size in &training_sizes {
        let training_data = generate_training_data(size, 10);
        
        let mut detector = AnomalyDetector::with_config(config.clone())
            .expect("Detector creation should succeed");
        
        let start_time = Instant::now();
        detector.train(&training_data).expect("Training should succeed");
        let training_time = start_time.elapsed();
        
        let throughput = size as f64 / training_time.as_secs_f64();
        
        println!("Training size: {}, Throughput: {:.0} elements/second", size, throughput);
        
        // Validate minimum throughput (should process at least 1000 elements/second)
        assert!(throughput >= 1000.0,
               "Training throughput {} elements/sec below 1000 threshold for size {}",
               throughput, size);
        
        // For smaller datasets, throughput should be higher
        if size <= 1000 {
            assert!(throughput >= 5000.0,
                   "Training throughput {} elements/sec below 5000 threshold for small size {}",
                   throughput, size);
        }
    }
    
    println!("✅ Training throughput measurement validation passed");
}

#[test]
fn test_training_memory_growth_patterns() {
    println!("Testing training memory growth patterns...");
    
    let config = AnomalyGridConfig::default()
        .with_max_order(4)
        .expect("Valid config");
    
    let data_sizes = vec![100, 200, 500, 1000, 2000];
    let mut memory_usages = Vec::new();
    
    for &size in &data_sizes {
        let training_data = generate_training_data(size, 15);
        
        let mut detector = AnomalyDetector::with_config(config.clone())
            .expect("Detector creation should succeed");
        
        detector.train(&training_data).expect("Training should succeed");
        
        let metrics = detector.performance_metrics();
        let memory_usage = metrics.estimated_memory_bytes;
        memory_usages.push(memory_usage);
        
        println!("Data size: {}, Memory usage: {} KB, Contexts: {}", 
                size, memory_usage / 1024, metrics.context_count);
        
        // Validate reasonable memory usage
        assert!(memory_usage < 50 * 1024 * 1024, // 50MB limit
               "Memory usage {} bytes exceeds 50MB limit for size {}", 
               memory_usage, size);
    }
    
    // Validate that memory growth is reasonable
    for i in 1..memory_usages.len() {
        let size_ratio = data_sizes[i] as f64 / data_sizes[i-1] as f64;
        let memory_ratio = memory_usages[i] as f64 / memory_usages[i-1] as f64;
        
        // Memory growth should not exceed cubic growth
        assert!(memory_ratio <= size_ratio * size_ratio * size_ratio * 2.0,
               "Memory growth {} exceeds cubic bound for size ratio {}",
               memory_ratio, size_ratio);
    }
    
    println!("✅ Training memory growth patterns validation passed");
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
        let state_index = match i % 4 {
            0 => i % alphabet_size,                    // Sequential pattern
            1 => (i / 2) % alphabet_size,              // Slower sequential
            2 => (i * 3) % alphabet_size,              // Skip pattern
            _ => (i * i) % alphabet_size,              // Quadratic pattern
        };
        
        data.push(alphabet[state_index].clone());
    }
    
    data
}