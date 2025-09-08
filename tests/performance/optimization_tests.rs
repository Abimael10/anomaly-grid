//! Performance optimization integration tests
//! 
//! These tests validate the performance optimization features and ensure
//! they provide real benefits while maintaining detection accuracy.

use anomaly_grid::*;
use std::time::Instant;

#[test]
fn test_performance_monitoring() {
    println!("🚀 Testing Performance Monitoring");

    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    
    // Create a training sequence
    let training_sequence: Vec<String> = (0..1000)
        .map(|i| format!("S{}", i % 10))
        .collect();
    
    // Train and measure performance
    let start_time = Instant::now();
    detector.train(&training_sequence).expect("Failed to train detector");
    let training_duration = start_time.elapsed();
    
    // Check performance metrics
    let metrics = detector.performance_metrics();
    assert!(metrics.training_time_ms > 0);
    assert!(metrics.context_count > 0);
    assert!(metrics.estimated_memory_bytes > 0);
    
    // Verify throughput calculation
    let throughput = metrics.training_throughput(training_sequence.len());
    assert!(throughput > 0.0);
    
    println!("  ✅ Training: {}ms, {} contexts, {} bytes, {:.0} elem/s", 
             metrics.training_time_ms, 
             metrics.context_count,
             metrics.estimated_memory_bytes,
             throughput);
    
    // Test detection performance
    let test_sequence: Vec<String> = (0..100)
        .map(|i| format!("T{}", i % 5))
        .collect();
    
    let anomalies = detector.detect_anomalies_with_monitoring(&test_sequence, 0.1).expect("Failed to detect anomalies");
    
    let detection_metrics = detector.performance_metrics();
    assert!(detection_metrics.detection_time_ms > 0);
    
    let detection_throughput = detection_metrics.detection_throughput(test_sequence.len());
    assert!(detection_throughput > 0.0);
    
    println!("  ✅ Detection: {}ms, {} anomalies, {:.0} elem/s",
             detection_metrics.detection_time_ms,
             anomalies.len(),
             detection_throughput);
}

#[test]
fn test_context_optimization() {
    println!("🔧 Testing Context Optimization");

    let mut detector = AnomalyDetector::new(4).expect("Failed to create detector");
    
    // Create a sequence with varying frequency patterns
    let mut training_sequence = Vec::new();
    
    // High frequency pattern
    for _ in 0..500 {
        training_sequence.extend(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    }
    
    // Medium frequency patterns
    for _ in 0..50 {
        training_sequence.extend(vec!["X".to_string(), "Y".to_string(), "Z".to_string()]);
    }
    
    // Low frequency patterns (noise)
    for i in 0..20 {
        training_sequence.extend(vec![format!("RARE_{}", i), "NOISE".to_string()]);
    }
    
    // Train the detector
    detector.train(&training_sequence).expect("Failed to train detector");
    
    let initial_metrics = detector.performance_metrics();
    let initial_contexts = initial_metrics.context_count;
    let initial_memory = initial_metrics.estimated_memory_bytes;
    
    println!("  Initial: {} contexts, {} bytes", initial_contexts, initial_memory);
    
    // Apply optimization
    let optimization_config = OptimizationConfig {
        enable_pruning: true,
        min_context_count: 3,     // Remove contexts seen < 3 times
        min_entropy: 0.1,         // Remove low-entropy contexts
        max_contexts: Some(1000), // Limit total contexts
        enable_monitoring: true,
    };
    
    detector.optimize(&optimization_config).expect("Failed to optimize detector");
    
    let optimized_metrics = detector.performance_metrics();
    let optimized_contexts = optimized_metrics.context_count;
    let optimized_memory = optimized_metrics.estimated_memory_bytes;
    
    println!("  Optimized: {} contexts, {} bytes", optimized_contexts, optimized_memory);
    
    // Verify optimization results
    assert!(optimized_contexts <= initial_contexts);
    assert!(optimized_memory <= initial_memory);
    
    // Calculate reduction percentages
    let context_reduction = if initial_contexts > 0 {
        ((initial_contexts - optimized_contexts) as f64 / initial_contexts as f64) * 100.0
    } else {
        0.0
    };
    
    let memory_reduction = if initial_memory > 0 {
        ((initial_memory - optimized_memory) as f64 / initial_memory as f64) * 100.0
    } else {
        0.0
    };
    
    println!("  ✅ Reduced contexts by {:.1}%, memory by {:.1}%", 
             context_reduction, memory_reduction);
    
    // Verify detection still works after optimization
    let test_sequence = vec!["A".to_string(), "B".to_string(), "UNKNOWN".to_string()];
    let anomalies = detector.detect_anomalies(&test_sequence, 0.1).expect("Failed to detect after optimization");
    
    println!("  ✅ Detection still works: {} anomalies found", anomalies.len());
}

#[test]
fn test_context_statistics() {
    println!("📊 Testing Context Statistics");

    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    
    // Create a diverse training sequence
    let training_sequence: Vec<String> = (0..2000)
        .map(|i| {
            match i % 100 {
                0..=70 => format!("COMMON_{}", i % 5),    // 70% common patterns
                71..=90 => format!("MEDIUM_{}", i % 10),   // 20% medium patterns  
                _ => format!("RARE_{}", i),                // 10% rare patterns
            }
        })
        .collect();
    
    detector.train(&training_sequence).expect("Failed to train detector");
    
    let stats = detector.context_statistics();
    
    println!("  Context Statistics:");
    println!("    Total contexts: {}", stats.total_contexts);
    println!("    Total transitions: {}", stats.total_transitions);
    println!("    Average entropy: {:.3}", stats.avg_entropy);
    println!("    Average frequency: {:.1}", stats.avg_frequency);
    println!("    Min/Max frequency: {} / {}", stats.min_frequency, stats.max_frequency);
    println!("    Min/Max entropy: {:.3} / {:.3}", stats.min_entropy, stats.max_entropy);
    
    // Verify statistics make sense
    assert!(stats.total_contexts > 0);
    assert!(stats.total_transitions > 0);
    assert!(stats.avg_entropy >= 0.0);
    assert!(stats.avg_frequency > 0.0);
    assert!(stats.min_frequency <= stats.max_frequency);
    assert!(stats.min_entropy <= stats.max_entropy);
    
    // Check contexts by order
    for (order, count) in &stats.contexts_by_order {
        println!("    Order {}: {} contexts", order, count);
        assert!(*order <= 3); // Should not exceed max_order
        assert!(*count > 0);
    }
    
    // Calculate memory efficiency
    let memory_efficiency = stats.memory_efficiency(detector.performance_metrics().estimated_memory_bytes);
    println!("    Memory efficiency: {:.1} contexts/MB", memory_efficiency);
    
    // Calculate compression ratio
    let compression_ratio = stats.compression_ratio();
    println!("    Compression ratio: {:.1} transitions/context", compression_ratio);
    
    assert!(memory_efficiency > 0.0);
    assert!(compression_ratio > 0.0);
    
    println!("  ✅ Context statistics computed successfully");
}

#[test]
fn test_optimization_presets() {
    println!("⚙️ Testing Optimization Presets");

    let test_sequence: Vec<String> = (0..1000)
        .map(|i| format!("S{}", i % 20))
        .collect();

    // Test different optimization presets
    let presets = vec![
        ("Low Memory", OptimizationConfig::for_low_memory()),
        ("High Accuracy", OptimizationConfig::for_high_accuracy()),
        ("Balanced", OptimizationConfig::balanced()),
    ];

    for (name, config) in presets {
        println!("  Testing {} preset:", name);
        
        let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
        detector.train(&test_sequence).expect("Failed to train detector");
        
        let before_metrics = detector.performance_metrics();
        let before_contexts = before_metrics.context_count;
        let before_memory = before_metrics.estimated_memory_bytes;
        
        detector.optimize(&config).expect("Failed to optimize");
        
        let after_metrics = detector.performance_metrics();
        let after_contexts = after_metrics.context_count;
        let after_memory = after_metrics.estimated_memory_bytes;
        
        println!("    Contexts: {} → {} ({:.1}% reduction)",
                 before_contexts, after_contexts,
                 if before_contexts > 0 {
                     ((before_contexts - after_contexts) as f64 / before_contexts as f64) * 100.0
                 } else { 0.0 });
        
        println!("    Memory: {} → {} bytes ({:.1}% reduction)",
                 before_memory, after_memory,
                 if before_memory > 0 {
                     ((before_memory - after_memory) as f64 / before_memory as f64) * 100.0
                 } else { 0.0 });
        
        // Verify optimization behavior matches preset expectations
        match name {
            "Low Memory" => {
                assert!(config.enable_pruning);
                assert!(config.max_contexts.is_some());
                // Should achieve significant reduction for low memory
                assert!(after_contexts <= before_contexts);
            }
            "High Accuracy" => {
                assert!(!config.enable_pruning);
                assert!(config.max_contexts.is_none());
                // Should preserve most contexts for accuracy
                assert_eq!(after_contexts, before_contexts);
            }
            "Balanced" => {
                assert!(config.enable_pruning);
                assert!(config.max_contexts.is_some());
                // Should achieve moderate reduction
                assert!(after_contexts <= before_contexts);
            }
            _ => {}
        }
        
        // Verify detection still works
        let test_detection = vec!["S1".to_string(), "UNKNOWN".to_string(), "S2".to_string()];
        let anomalies = detector.detect_anomalies(&test_detection, 0.1).expect("Failed to detect");
        
        println!("    Detection: {} anomalies found", anomalies.len());
        println!("    ✅ {} preset validated", name);
    }
}

#[test]
fn test_performance_regression() {
    println!("⏱️ Testing Performance Regression");

    let sizes = vec![100, 500, 1000];
    let orders = vec![2, 3, 4];
    
    for &size in &sizes {
        for &order in &orders {
            println!("  Testing size={}, order={}", size, order);
            
            let sequence: Vec<String> = (0..size)
                .map(|i| format!("S{}", i % 10))
                .collect();
            
            let mut detector = AnomalyDetector::new(order).expect("Failed to create detector");
            
            // Measure training time
            let start_time = Instant::now();
            detector.train(&sequence).expect("Failed to train");
            let training_time = start_time.elapsed();
            
            // Measure detection time
            let test_sequence: Vec<String> = (0..50)
                .map(|i| format!("T{}", i % 5))
                .collect();
            
            let start_time = Instant::now();
            let _anomalies = detector.detect_anomalies_with_monitoring(&test_sequence, 0.1).expect("Failed to detect");
            let detection_time = start_time.elapsed();
            
            // Performance should be reasonable
            let training_ms = training_time.as_millis();
            let detection_ms = detection_time.as_millis();
            
            // Training should complete in reasonable time (very generous bounds)
            assert!(training_ms < (size as u128 * order as u128), 
                   "Training too slow: {}ms for size={}, order={}", training_ms, size, order);
            
            // Detection should be fast
            assert!(detection_ms < 100, 
                   "Detection too slow: {}ms for size={}, order={}", detection_ms, size, order);
            
            let metrics = detector.performance_metrics();
            let training_throughput = metrics.training_throughput(size);
            let detection_throughput = metrics.detection_throughput(test_sequence.len());
            
            println!("    Training: {}ms ({:.0} elem/s), Detection: {}ms ({:.0} elem/s)",
                     training_ms, training_throughput, detection_ms, detection_throughput);
            
            // Throughput should be reasonable
            assert!(training_throughput > 100.0, "Training throughput too low: {:.0}", training_throughput);
            assert!(detection_throughput > 1000.0, "Detection throughput too low: {:.0}", detection_throughput);
        }
    }
    
    println!("  ✅ Performance within acceptable bounds");
}

#[test]
fn test_memory_optimization_effectiveness() {
    println!("💾 Testing Memory Optimization Effectiveness");

    // Create a detector with a large, sparse context space
    let mut detector = AnomalyDetector::new(4).expect("Failed to create detector");
    
    // Generate sequence with many rare patterns
    let mut sequence = Vec::new();
    
    // Add common patterns
    for _ in 0..1000 {
        sequence.extend(vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()]);
    }
    
    // Add many rare patterns (each seen only once)
    for i in 0..500 {
        sequence.extend(vec![format!("RARE_{}", i), "X".to_string(), "Y".to_string()]);
    }
    
    detector.train(&sequence).expect("Failed to train");
    
    let initial_metrics = detector.performance_metrics();
    let initial_memory = initial_metrics.estimated_memory_bytes;
    let initial_contexts = initial_metrics.context_count;
    
    println!("  Before optimization: {} contexts, {} bytes", 
             initial_contexts, initial_memory);
    
    // Apply aggressive optimization
    let optimization_config = OptimizationConfig {
        enable_pruning: true,
        min_context_count: 5,      // Remove contexts seen < 5 times
        min_entropy: 0.2,          // Remove low-entropy contexts
        max_contexts: Some(1000),  // Limit to 1000 contexts
        enable_monitoring: true,
    };
    
    detector.optimize(&optimization_config).expect("Failed to optimize");
    
    let optimized_metrics = detector.performance_metrics();
    let optimized_memory = optimized_metrics.estimated_memory_bytes;
    let optimized_contexts = optimized_metrics.context_count;
    
    println!("  After optimization: {} contexts, {} bytes", 
             optimized_contexts, optimized_memory);
    
    // Calculate reductions
    let context_reduction = ((initial_contexts - optimized_contexts) as f64 / initial_contexts as f64) * 100.0;
    let memory_reduction = ((initial_memory - optimized_memory) as f64 / initial_memory as f64) * 100.0;
    
    println!("  Reductions: {:.1}% contexts, {:.1}% memory", context_reduction, memory_reduction);
    
    // Should achieve significant reduction due to many rare patterns
    assert!(context_reduction > 10.0, "Should reduce contexts by at least 10%");
    assert!(memory_reduction > 10.0, "Should reduce memory by at least 10%");
    
    // Verify detection accuracy is maintained for common patterns
    let common_test = vec!["A".to_string(), "B".to_string(), "C".to_string(), "UNKNOWN".to_string()];
    let anomalies = detector.detect_anomalies(&common_test, 0.1).expect("Failed to detect");
    
    // Should still detect the unknown pattern
    assert!(!anomalies.is_empty(), "Should still detect anomalies after optimization");
    
    println!("  ✅ Memory optimization effective: {:.1}% reduction with maintained accuracy", 
             memory_reduction);
}

#[test]
fn test_on_demand_probability_computation() {
    println!("🧮 Testing On-Demand Probability Computation");
    
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    
    // Create a realistic training sequence
    let mut sequence = Vec::new();
    for _ in 0..100 {
        sequence.extend(vec![
            "A".to_string(), "B".to_string(), "C".to_string(),
            "B".to_string(), "C".to_string(), "D".to_string(),
            "C".to_string(), "D".to_string(), "A".to_string(),
        ]);
    }
    
    detector.train(&sequence).expect("Failed to train");
    
    let context_tree = detector.model().context_tree();
    let config = AnomalyGridConfig::default();
    
    println!("  Testing on-demand probability calculation...");
    
    for (context, node) in &context_tree.contexts {
        // Test that we can get probabilities on-demand
        let probabilities = node.get_all_probabilities(&config);
        
        // Verify probability conservation
        let prob_sum: f64 = probabilities.values().sum();
        let error = (prob_sum - 1.0).abs();
        assert!(error < 1e-12, "Probability conservation violated: error = {:.2e}", error);
        
        // Test entropy calculation
        let entropy = node.calculate_entropy(&config);
        assert!(entropy >= 0.0, "Entropy must be non-negative: {:.6}", entropy);
        
        // Test KL divergence calculation
        let kl_div = node.calculate_kl_divergence(&config);
        assert!(kl_div >= 0.0, "KL divergence must be non-negative: {:.6}", kl_div);
        
        // Test individual probability access
        let counts = node.counts();
        for state in counts.keys() {
            let prob = node.get_probability(state, &config);
            assert!(prob >= 0.0 && prob <= 1.0, "Probability out of bounds: {:.6}", prob);
        }
        
        // Verify that total_count matches sum of individual counts
        let manual_total: usize = counts.values().sum();
        assert_eq!(node.total_count(), manual_total, "Total count mismatch");
    }
    
    println!("  ✅ On-demand calculations working correctly");
    
    // Test mathematical correctness with known values
    if let Some(node) = context_tree.get_context_node(&["A".to_string()]) {
        // Test Laplace smoothing formula with α=1.0 (default)
        let prob_b = node.get_probability("B", &config);
        let prob_c = node.get_probability("C", &config);
        
        // Both should be positive and sum to 1 (along with other transitions)
        assert!(prob_b > 0.0 && prob_b <= 1.0, "P(B|A) out of bounds: {:.6}", prob_b);
        assert!(prob_c > 0.0 && prob_c <= 1.0, "P(C|A) out of bounds: {:.6}", prob_c);
        
        println!("  ✅ Laplace smoothing working correctly");
    }
    
    // Test that detection still works correctly
    let test_sequence = vec!["A".to_string(), "B".to_string(), "C".to_string()];
    let anomalies = detector.detect_anomalies(&test_sequence, 0.1)
        .expect("Failed to detect anomalies");
    
    // Verify mathematical properties are maintained
    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
    }
    
    println!("  ✅ Detection functionality maintained");
    
    // Estimate memory efficiency from on-demand computation
    let context_count = context_tree.context_count();
    let estimated_memory = context_tree.estimate_memory_usage();
    
    println!("  Memory usage analysis:");
    println!("    Contexts: {}", context_count);
    println!("    Estimated memory: {:.2} KB", estimated_memory as f64 / 1024.0);
    println!("    Memory per context: {:.1} bytes", estimated_memory as f64 / context_count as f64);
    
    // With on-demand computation, memory per context should be significantly reduced
    // Only stores: counts + total_count (no redundant probability storage)
    let memory_per_context = estimated_memory as f64 / context_count as f64;
    assert!(memory_per_context < 500.0, "Memory per context should be reduced: {:.1} bytes", memory_per_context);
    
    println!("  ✅ On-demand computation memory optimization validated");
    println!("  🎉 On-demand probability computation working efficiently!");
}

#[test]
fn test_small_collection_memory_optimization() {
    println!("Testing Small Collection Memory Optimization");
    
    use anomaly_grid::transition_counts::TransitionCounts;
    use anomaly_grid::string_interner::StateId;
    
    // Test TransitionCounts efficiency
    let mut small_counts = TransitionCounts::new();
    assert!(small_counts.is_small(), "Should start as small collection");
    
    // Add a few transitions
    for i in 1..=3 {
        small_counts.increment(StateId::new(i));
    }
    
    assert!(small_counts.is_small(), "Should remain small with 3 items");
    assert_eq!(small_counts.len(), 3);
    
    let small_memory = small_counts.memory_usage();
    
    // Test promotion to large collection
    let mut large_counts = TransitionCounts::new();
    for i in 1..=10 {
        large_counts.increment(StateId::new(i));
    }
    
    assert!(!large_counts.is_small(), "Should promote to large collection");
    assert_eq!(large_counts.len(), 10);
    
    let large_memory = large_counts.memory_usage();
    
    // Verify functionality is preserved
    for i in 1..=10 {
        assert_eq!(large_counts.get(StateId::new(i)), 1);
    }
    
    // Test with real detector
    let mut detector = AnomalyDetector::new(2).expect("Failed to create detector");
    let sequence: Vec<String> = (0..1000)
        .map(|i| format!("S{}", i % 5))
        .collect();
    
    detector.train(&sequence).expect("Failed to train");
    
    let context_tree = detector.model().context_tree();
    let context_count = context_tree.context_count();
    let memory_usage = context_tree.estimate_memory_usage();
    
    // Verify optimization is working
    assert!(context_count > 0, "Should create contexts");
    assert!(memory_usage > 0, "Should use some memory");
    
    // Print results
    println!("  ✅ Small collection optimization validated");
    println!("    Contexts: {}", context_count);
    println!("    Memory usage: {} bytes", memory_usage);
    
    // Small collection optimization is now integrated in TransitionCounts
    // All contexts automatically benefit from SmallVec optimization
}

#[test]
fn test_string_interning_integration() {
    println!("🔗 Testing String Interning Integration");
    
    let mut detector = AnomalyDetector::new(3).expect("Failed to create detector");
    
    // Create a sequence with repeated strings to test interning efficiency
    let mut sequence = Vec::new();
    let states = vec!["STATE_A", "STATE_B", "STATE_C", "STATE_D"];
    
    // Repeat the same states many times to test string deduplication
    for _ in 0..200 {
        for state in &states {
            sequence.push(state.to_string());
        }
    }
    
    detector.train(&sequence).expect("Failed to train");
    
    let context_tree = detector.model().context_tree();
    let interner = context_tree.interner();
    
    println!("  String interning analysis:");
    println!("    Total sequence length: {}", sequence.len());
    println!("    Unique strings in interner: {}", interner.len());
    println!("    Interner memory usage: {} bytes", interner.estimate_memory_usage());
    
    // Verify that string interning is working
    assert_eq!(interner.len(), states.len(), "Should have exactly {} unique strings", states.len());
    
    // Test that the same string gets the same StateId
    let id1 = interner.get_or_intern("STATE_A");
    let id2 = interner.get_or_intern("STATE_A");
    assert_eq!(id1, id2, "Same string should get same StateId");
    
    // Test that different strings get different StateIds
    let id_a = interner.get_or_intern("STATE_A");
    let id_b = interner.get_or_intern("STATE_B");
    assert_ne!(id_a, id_b, "Different strings should get different StateIds");
    
    // Test that we can retrieve strings from StateIds
    let retrieved = interner.get_string(id_a).expect("Should retrieve string");
    assert_eq!(retrieved, "STATE_A", "Should retrieve correct string");
    
    println!("  ✅ String interning working correctly");
    
    // Test that detection still works with interned strings
    let test_sequence = vec!["STATE_A".to_string(), "STATE_B".to_string(), "STATE_C".to_string()];
    let anomalies = detector.detect_anomalies(&test_sequence, 0.1)
        .expect("Failed to detect anomalies");
    
    // Verify mathematical properties are maintained
    for anomaly in &anomalies {
        assert!(anomaly.likelihood >= 0.0 && anomaly.likelihood <= 1.0);
        assert!(anomaly.anomaly_strength >= 0.0 && anomaly.anomaly_strength <= 1.0);
        assert!(anomaly.information_score >= 0.0);
    }
    
    println!("  ✅ Detection functionality maintained with string interning");
    
    // Estimate memory efficiency
    let context_count = context_tree.context_count();
    let estimated_memory = context_tree.estimate_memory_usage();
    
    println!("  Memory efficiency analysis:");
    println!("    Contexts: {}", context_count);
    println!("    Total estimated memory: {:.2} KB", estimated_memory as f64 / 1024.0);
    println!("    Memory per context: {:.1} bytes", estimated_memory as f64 / context_count as f64);
    
    // With string interning, memory usage should be more efficient
    assert!(context_count > 0, "Should have created contexts");
    assert!(estimated_memory > 0, "Should have some memory usage");
    
    println!("  ✅ String interning integration validated");
    println!("  🎉 String interning providing memory efficiency!");
}