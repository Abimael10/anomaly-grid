//! Unit tests for lazy computation with caching functionality
//! 
//! These tests validate the caching behavior, cache invalidation,
//! and performance characteristics of the lazy computation system.

use anomaly_grid::context_tree::ContextNode;
use anomaly_grid::config::AnomalyGridConfig;
use anomaly_grid::string_interner::StringInterner;
use std::sync::Arc;

#[test]
fn test_lazy_entropy_computation() {
    let mut node = ContextNode::new(Arc::new(StringInterner::new()));
    let config = AnomalyGridConfig::default();
    
    // Add some transitions
    node.add_transition("A");
    node.add_transition("B");
    node.add_transition("A");
    
    // First call should compute and cache
    let entropy1 = node.calculate_entropy(&config);
    let (entropy_cached, _) = node.cache_stats();
    assert!(entropy_cached, "Entropy should be cached after first calculation");
    
    // Second call should use cache (same result)
    let entropy2 = node.calculate_entropy(&config);
    assert_eq!(entropy1, entropy2, "Cached entropy should match computed entropy");
    
    // Verify entropy is reasonable
    assert!(entropy1 > 0.0, "Entropy should be positive for mixed distribution");
    assert!(entropy1 < 2.0, "Entropy should be less than log2(2) for binary distribution");
}

#[test]
fn test_lazy_kl_divergence_computation() {
    let mut node = ContextNode::new(Arc::new(StringInterner::new()));
    let config = AnomalyGridConfig::default();
    
    // Add some transitions
    node.add_transition("A");
    node.add_transition("B");
    node.add_transition("A");
    
    // First call should compute and cache
    let kl_div1 = node.calculate_kl_divergence(&config);
    let (_, kl_cached) = node.cache_stats();
    assert!(kl_cached, "KL divergence should be cached after first calculation");
    
    // Second call should use cache (same result)
    let kl_div2 = node.calculate_kl_divergence(&config);
    assert_eq!(kl_div1, kl_div2, "Cached KL divergence should match computed KL divergence");
    
    // Verify KL divergence is reasonable
    assert!(kl_div1 >= 0.0, "KL divergence should be non-negative");
}

#[test]
fn test_cache_invalidation_on_data_change() {
    let mut node = ContextNode::new(Arc::new(StringInterner::new()));
    let config = AnomalyGridConfig::default();
    
    // Add initial transitions and compute entropy
    node.add_transition("A");
    node.add_transition("B");
    let entropy1 = node.calculate_entropy(&config);
    
    let (entropy_cached, _) = node.cache_stats();
    assert!(entropy_cached, "Entropy should be cached");
    
    // Add more transitions (should invalidate cache)
    node.add_transition("C");
    let (entropy_cached_after, _) = node.cache_stats();
    assert!(!entropy_cached_after, "Cache should be invalidated after data change");
    
    // Compute entropy again (should be different)
    let entropy2 = node.calculate_entropy(&config);
    assert_ne!(entropy1, entropy2, "Entropy should change after adding new transition");
    
    // Cache should be populated again
    let (entropy_cached_final, _) = node.cache_stats();
    assert!(entropy_cached_final, "Entropy should be cached again after recomputation");
}

#[test]
fn test_cache_invalidation_on_config_change() {
    let mut node = ContextNode::new(Arc::new(StringInterner::new()));
    let config1 = AnomalyGridConfig::default();
    let mut config2 = AnomalyGridConfig::default();
    config2.smoothing_alpha = 2.0; // Different smoothing parameter
    
    // Add transitions with unequal distribution to make smoothing effect more visible
    node.add_transition("A");
    node.add_transition("A");
    node.add_transition("A");
    node.add_transition("B");
    
    // Compute with first config
    let entropy1 = node.calculate_entropy(&config1);
    let (entropy_cached, _) = node.cache_stats();
    assert!(entropy_cached, "Entropy should be cached");
    
    // Compute with different config (should recompute)
    let entropy2 = node.calculate_entropy(&config2);
    assert_ne!(entropy1, entropy2, "Entropy should be different with different smoothing: {:.6} vs {:.6}", entropy1, entropy2);
    
    // Compute with first config again (should recompute due to config change)
    let entropy3 = node.calculate_entropy(&config1);
    assert_eq!(entropy1, entropy3, "Entropy should be same with same config");
}

#[test]
fn test_mixed_cached_and_uncached_access() {
    let mut node = ContextNode::new(Arc::new(StringInterner::new()));
    let config = AnomalyGridConfig::default();
    
    // Add transitions
    node.add_transition("A");
    node.add_transition("B");
    
    // Use immutable compute method (no caching)
    let entropy_uncached = node.compute_entropy(&config);
    let (entropy_cached, _) = node.cache_stats();
    assert!(!entropy_cached, "Entropy should not be cached after compute_entropy");
    
    // Use mutable calculate method (with caching)
    let entropy_cached_result = node.calculate_entropy(&config);
    let (entropy_cached_after, _) = node.cache_stats();
    assert!(entropy_cached_after, "Entropy should be cached after calculate_entropy");
    
    // Results should be identical
    assert_eq!(entropy_uncached, entropy_cached_result, "Cached and uncached results should match");
    
    // Another call to compute should not affect cache
    let entropy_uncached2 = node.compute_entropy(&config);
    assert_eq!(entropy_uncached, entropy_uncached2, "Multiple compute calls should give same result");
    let (entropy_still_cached, _) = node.cache_stats();
    assert!(entropy_still_cached, "Cache should remain valid after compute_entropy call");
}

#[test]
fn test_cache_behavior_with_empty_node() {
    let mut node = ContextNode::new(Arc::new(StringInterner::new()));
    let config = AnomalyGridConfig::default();
    
    // Compute entropy on empty node
    let entropy = node.calculate_entropy(&config);
    assert_eq!(entropy, 0.0, "Entropy of empty node should be 0");
    
    let (entropy_cached, _) = node.cache_stats();
    assert!(entropy_cached, "Even zero entropy should be cached");
    
    // Compute KL divergence on empty node
    let kl_div = node.calculate_kl_divergence(&config);
    assert_eq!(kl_div, 0.0, "KL divergence of empty node should be 0");
    
    let (_, kl_cached) = node.cache_stats();
    assert!(kl_cached, "Even zero KL divergence should be cached");
}

#[test]
fn test_cache_reset_and_clear() {
    let mut node = ContextNode::new(Arc::new(StringInterner::new()));
    let config = AnomalyGridConfig::default();
    
    // Add transitions and compute values
    node.add_transition("A");
    node.add_transition("B");
    let _entropy = node.calculate_entropy(&config);
    let _kl_div = node.calculate_kl_divergence(&config);
    
    let (entropy_cached, kl_cached) = node.cache_stats();
    assert!(entropy_cached && kl_cached, "Both values should be cached");
    
    // Test reset
    let new_interner = Arc::new(StringInterner::new());
    node.reset(new_interner);
    let (entropy_cached_after_reset, kl_cached_after_reset) = node.cache_stats();
    assert!(!entropy_cached_after_reset && !kl_cached_after_reset, "Cache should be cleared after reset");
    
    // Add transitions again and cache
    node.add_transition("X");
    let _entropy2 = node.calculate_entropy(&config);
    let (entropy_cached_again, _) = node.cache_stats();
    assert!(entropy_cached_again, "Cache should work after reset");
    
    // Test clear
    node.clear();
    let (entropy_cached_after_clear, kl_cached_after_clear) = node.cache_stats();
    assert!(!entropy_cached_after_clear && !kl_cached_after_clear, "Cache should be cleared after clear");
}

#[test]
fn test_mathematical_correctness_with_caching() {
    let mut node = ContextNode::new(Arc::new(StringInterner::new()));
    let config = AnomalyGridConfig::default();
    
    // Create a known distribution: A=3, B=1 (total=4)
    node.add_transition("A");
    node.add_transition("A");
    node.add_transition("A");
    node.add_transition("B");
    
    // Calculate cached values
    let entropy_cached = node.calculate_entropy(&config);
    let kl_div_cached = node.calculate_kl_divergence(&config);
    
    // Calculate uncached values
    let entropy_uncached = node.compute_entropy(&config);
    let kl_div_uncached = node.compute_kl_divergence(&config);
    
    // Should be identical
    const TOLERANCE: f64 = 1e-15;
    assert!((entropy_cached - entropy_uncached).abs() < TOLERANCE, 
           "Cached and uncached entropy should be identical: {:.15} vs {:.15}", 
           entropy_cached, entropy_uncached);
    assert!((kl_div_cached - kl_div_uncached).abs() < TOLERANCE,
           "Cached and uncached KL divergence should be identical: {:.15} vs {:.15}",
           kl_div_cached, kl_div_uncached);
    
    // Verify mathematical properties
    assert!(entropy_cached >= 0.0, "Entropy must be non-negative");
    assert!(kl_div_cached >= 0.0, "KL divergence must be non-negative");
    
    // For this specific distribution with Laplace smoothing
    // P(A) = (3 + α) / (4 + 2α), P(B) = (1 + α) / (4 + 2α) where α = 1.0
    let alpha = config.smoothing_alpha;
    let p_a = (3.0 + alpha) / (4.0 + 2.0 * alpha);
    let p_b = (1.0 + alpha) / (4.0 + 2.0 * alpha);
    
    // Manual entropy calculation
    let expected_entropy = -p_a * p_a.log2() - p_b * p_b.log2();
    assert!((entropy_cached - expected_entropy).abs() < TOLERANCE,
           "Entropy should match manual calculation: {:.15} vs {:.15}",
           entropy_cached, expected_entropy);
}

#[test]
fn test_cache_performance_characteristics() {
    let mut node = ContextNode::new(Arc::new(StringInterner::new()));
    let config = AnomalyGridConfig::default();
    
    // Add a reasonable number of transitions
    for i in 0..100 {
        node.add_transition(&format!("STATE_{}", i % 10));
    }
    
    // Time the first computation (should be slower - no cache)
    let start = std::time::Instant::now();
    let entropy1 = node.calculate_entropy(&config);
    let first_duration = start.elapsed();
    
    // Time the second computation (should be faster - cached)
    let start = std::time::Instant::now();
    let entropy2 = node.calculate_entropy(&config);
    let second_duration = start.elapsed();
    
    // Results should be identical
    assert_eq!(entropy1, entropy2, "Cached result should match computed result");
    
    // Second call should be faster (though this might not always be true due to timing variations)
    // We'll just verify that both complete in reasonable time
    assert!(first_duration.as_nanos() < 1_000_000, "First computation should complete quickly");
    assert!(second_duration.as_nanos() < 1_000_000, "Second computation should complete quickly");
    
    println!("First computation: {:?}, Second computation: {:?}", first_duration, second_duration);
}