//! Performance optimization utilities for Anomaly Grid
//!
//! This module provides practical performance improvements focused on
//! memory efficiency and computational optimization for anomaly detection.

use crate::context_tree::ContextTree;
use crate::error::AnomalyGridResult;
use crate::string_interner::StateId;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Simple performance metrics for monitoring
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Training time in milliseconds
    pub training_time_ms: u64,
    /// Detection time in milliseconds  
    pub detection_time_ms: u64,
    /// Number of contexts created
    pub context_count: usize,
    /// Estimated memory usage in bytes
    pub estimated_memory_bytes: usize,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new() -> Self {
        Self {
            training_time_ms: 0,
            detection_time_ms: 0,
            context_count: 0,
            estimated_memory_bytes: 0,
        }
    }

    /// Calculate training throughput (elements per second)
    pub fn training_throughput(&self, sequence_length: usize) -> f64 {
        if self.training_time_ms == 0 {
            return 0.0;
        }
        (sequence_length as f64) / (self.training_time_ms as f64 / 1000.0)
    }

    /// Calculate detection throughput (elements per second)
    pub fn detection_throughput(&self, sequence_length: usize) -> f64 {
        if self.detection_time_ms == 0 {
            return 0.0;
        }
        (sequence_length as f64) / (self.detection_time_ms as f64 / 1000.0)
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Context pruning for memory optimization
impl ContextTree {
    /// Remove contexts with low frequency counts
    ///
    /// This removes contexts that have been observed fewer than `min_count` times,
    /// which can significantly reduce memory usage for large alphabets.
    pub fn prune_low_frequency_contexts(&mut self, min_count: usize) -> usize {
        if min_count == 0 {
            return 0;
        }

        self.rebuild_filtered(|_, node| node.total_count() >= min_count)
    }

    /// Remove contexts with low entropy (deterministic contexts)
    ///
    /// This removes contexts where the entropy is below the threshold,
    /// indicating highly predictable transitions.
    ///
    /// Entropy is computed using the last configuration used during training.
    pub fn prune_low_entropy_contexts(&mut self, min_entropy: f64) -> usize {
        if min_entropy <= 0.0 {
            return 0;
        }

        let cfg = self.last_config.clone();
        let gv = self.global_vocab_size();
        self.rebuild_filtered(|_, node| node.compute_entropy(&cfg, gv) >= min_entropy)
    }

    /// Keep only the most frequent contexts up to a maximum count
    ///
    /// This is useful for memory-constrained environments where you want to keep
    /// only the most important contexts.
    ///
    /// Returns the number of contexts removed.
    pub fn limit_context_count(&mut self, max_contexts: usize) -> usize {
        if max_contexts == 0 {
            return 0;
        }

        let original_count = self.trie().context_count();
        if original_count <= max_contexts {
            return 0;
        }

        // Collect contexts with their frequencies
        let mut contexts: Vec<(Vec<StateId>, usize)> = self
            .trie()
            .iter_contexts()
            .map(|(state_ids, node)| (state_ids, node.total_count()))
            .collect();

        // Keep the most frequent contexts
        contexts.sort_by(|a, b| b.1.cmp(&a.1));
        let keep: HashSet<Vec<StateId>> = contexts
            .into_iter()
            .take(max_contexts)
            .map(|(ids, _)| ids)
            .collect();

        self.rebuild_filtered(|state_ids, _| keep.contains(state_ids))
    }

    /// Estimate memory usage of the context tree
    ///
    /// This provides an estimate of the total memory used by the context tree,
    /// including the trie structure, context nodes, and transition counts.
    pub fn estimate_memory_usage(&self) -> usize {
        self.trie().memory_usage()
    }

    /// Get context statistics for analysis
    ///
    /// Returns detailed statistics about the context tree structure,
    /// including distribution by order and memory usage patterns.
    pub fn get_context_statistics(&self) -> ContextStatistics {
        let mut stats = ContextStatistics::new();
        stats.total_contexts = self.trie().context_count();

        for (state_ids, node) in self.trie().iter_contexts() {
            let order = state_ids.len();
            *stats.contexts_by_order.entry(order).or_insert(0) += 1;
            stats.total_transitions += node.total_transitions();
        }

        if stats.total_contexts > 0 {
            stats.avg_frequency = stats.total_transitions as f64 / stats.total_contexts as f64;
        }

        stats
    }
}

/// Statistics about context tree structure and usage
#[derive(Debug, Clone)]
pub struct ContextStatistics {
    /// Total number of contexts
    pub total_contexts: usize,
    /// Total number of transitions across all contexts
    pub total_transitions: usize,
    /// Sum of entropy across all contexts
    pub total_entropy: f64,
    /// Average entropy per context
    pub avg_entropy: f64,
    /// Average frequency per context
    pub avg_frequency: f64,
    /// Minimum frequency observed
    pub min_frequency: usize,
    /// Maximum frequency observed
    pub max_frequency: usize,
    /// Minimum entropy observed
    pub min_entropy: f64,
    /// Maximum entropy observed
    pub max_entropy: f64,
    /// Number of contexts by order
    pub contexts_by_order: HashMap<usize, usize>,
    /// Number of unique transitions by context order
    pub transitions_by_context: HashMap<usize, usize>,
}

impl ContextStatistics {
    /// Create new context statistics
    pub fn new() -> Self {
        Self {
            total_contexts: 0,
            total_transitions: 0,
            total_entropy: 0.0,
            avg_entropy: 0.0,
            avg_frequency: 0.0,
            min_frequency: usize::MAX,
            max_frequency: 0,
            min_entropy: f64::INFINITY,
            max_entropy: 0.0,
            contexts_by_order: HashMap::new(),
            transitions_by_context: HashMap::new(),
        }
    }

    /// Get memory efficiency (contexts per MB)
    pub fn memory_efficiency(&self, memory_bytes: usize) -> f64 {
        if memory_bytes == 0 {
            return 0.0;
        }
        (self.total_contexts as f64) / (memory_bytes as f64 / 1_048_576.0)
    }

    /// Get compression ratio (transitions per context)
    pub fn compression_ratio(&self) -> f64 {
        if self.total_contexts == 0 {
            return 0.0;
        }
        self.total_transitions as f64 / self.total_contexts as f64
    }
}

impl Default for ContextStatistics {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    /// Enable context pruning
    pub enable_pruning: bool,
    /// Minimum count for context pruning
    pub min_context_count: usize,
    /// Minimum entropy for context pruning
    pub min_entropy: f64,
    /// Maximum number of contexts to keep
    pub max_contexts: Option<usize>,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_pruning: false,
            min_context_count: 2,
            min_entropy: 0.1,
            max_contexts: None,
            enable_monitoring: true,
        }
    }
}

impl OptimizationConfig {
    /// Create configuration for memory-constrained environments
    pub fn for_low_memory() -> Self {
        Self {
            enable_pruning: true,
            min_context_count: 3,
            min_entropy: 0.2,
            max_contexts: Some(10_000),
            enable_monitoring: true,
        }
    }

    /// Create configuration for high-accuracy requirements
    pub fn for_high_accuracy() -> Self {
        Self {
            enable_pruning: false,
            min_context_count: 1,
            min_entropy: 0.0,
            max_contexts: None,
            enable_monitoring: true,
        }
    }

    /// Create configuration for balanced performance
    pub fn balanced() -> Self {
        Self {
            enable_pruning: true,
            min_context_count: 2,
            min_entropy: 0.05,
            max_contexts: Some(100_000),
            enable_monitoring: true,
        }
    }
}

/// Apply performance optimizations to a context tree
pub fn optimize_context_tree(
    tree: &mut ContextTree,
    config: &OptimizationConfig,
) -> AnomalyGridResult<PerformanceMetrics> {
    let start_time = Instant::now();
    let _initial_count = tree.context_count();

    if config.enable_pruning {
        // Apply frequency-based pruning
        if config.min_context_count > 1 {
            tree.prune_low_frequency_contexts(config.min_context_count);
        }

        // Apply entropy-based pruning
        if config.min_entropy > 0.0 {
            tree.prune_low_entropy_contexts(config.min_entropy);
        }

        // Apply maximum context limit
        if let Some(max_contexts) = config.max_contexts {
            tree.limit_context_count(max_contexts);
        }
    }

    let optimization_time = start_time.elapsed();

    let mut metrics = PerformanceMetrics::new();
    metrics.training_time_ms = optimization_time.as_millis() as u64;
    metrics.context_count = tree.context_count();
    metrics.estimated_memory_bytes = tree.estimate_memory_usage();

    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();

        metrics.training_time_ms = 100; // 0.1 second
        metrics.detection_time_ms = 50; // 0.05 seconds

        // Test throughput calculations
        // 1000 elements / 0.1 seconds = 10,000 elements/second
        assert!((metrics.training_throughput(1000) - 10_000.0).abs() < f64::EPSILON);
        // 500 elements / 0.05 seconds = 10,000 elements/second
        assert!((metrics.detection_throughput(500) - 10_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_context_pruning() {
        let mut tree = ContextTree::new(2).expect("Failed to create tree");

        // Build contexts using the proper API
        let config = crate::config::AnomalyGridConfig::default();

        // Create high frequency sequence
        let high_freq_sequence: Vec<String> = std::iter::repeat_n("X".to_string(), 5)
            .chain(std::iter::repeat_n("A".to_string(), 10))
            .collect();
        tree.build_from_sequence(&high_freq_sequence, &config)
            .expect("Failed to build");

        // Create low frequency sequence
        let low_freq_sequence = vec!["Y".to_string(), "B".to_string()];
        tree.build_from_sequence(&low_freq_sequence, &config)
            .expect("Failed to build");

        let initial_count = tree.context_count();
        assert!(initial_count > 0);

        // Prune contexts with frequency < 5
        let pruned = tree.prune_low_frequency_contexts(5);

        assert!(pruned > 0);
        assert!(tree.context_count() < initial_count);
    }

    #[test]
    fn test_memory_estimation() {
        let mut tree = ContextTree::new(2).expect("Failed to create tree");

        // Build contexts using the proper API
        let config = crate::config::AnomalyGridConfig::default();
        let sequence = vec!["X".to_string(), "A".to_string(), "B".to_string()];
        tree.build_from_sequence(&sequence, &config)
            .expect("Failed to build");

        let memory_usage = tree.estimate_memory_usage();
        assert!(memory_usage > 0);
    }

    #[test]
    fn test_context_statistics() {
        let mut tree = ContextTree::new(2).expect("Failed to create tree");

        // Build contexts using the proper API
        let config = crate::config::AnomalyGridConfig::default();

        // Create sequences that will generate contexts of different orders
        let sequence1 = vec!["X".to_string(), "A".to_string(), "B".to_string()];
        tree.build_from_sequence(&sequence1, &config)
            .expect("Failed to build");

        let sequence2 = vec!["Y".to_string(), "Z".to_string(), "C".to_string()];
        tree.build_from_sequence(&sequence2, &config)
            .expect("Failed to build");

        let stats = tree.get_context_statistics();

        assert!(stats.total_contexts > 0);
        // With max_order=2, we should have contexts of order 1 and 2
        assert!(stats.contexts_by_order.contains_key(&1));
        assert!(stats.contexts_by_order.contains_key(&2));
    }

    #[test]
    fn test_optimization_config() {
        let low_memory = OptimizationConfig::for_low_memory();
        assert!(low_memory.enable_pruning);
        assert!(low_memory.max_contexts.is_some());

        let high_accuracy = OptimizationConfig::for_high_accuracy();
        assert!(!high_accuracy.enable_pruning);
        assert!(high_accuracy.max_contexts.is_none());

        let balanced = OptimizationConfig::balanced();
        assert!(balanced.enable_pruning);
        assert!(balanced.max_contexts.is_some());
    }

    #[test]
    fn test_optimize_context_tree() {
        let mut tree = ContextTree::new(2).expect("Failed to create tree");

        // Build contexts using the proper API
        let config_build = crate::config::AnomalyGridConfig::default();

        // Create sequences with different patterns to generate various contexts
        for i in 1..=5 {
            let sequence: Vec<String> = (0..i + 2).map(|j| format!("S{}", j % 3)).collect();
            tree.build_from_sequence(&sequence, &config_build)
                .expect("Failed to build");
        }

        let initial_count = tree.context_count();
        assert!(initial_count > 0);

        // Use optimization config
        let config = OptimizationConfig {
            enable_pruning: true,
            min_context_count: 2,
            min_entropy: 0.0,
            max_contexts: Some(8),
            enable_monitoring: true,
        };

        let metrics = optimize_context_tree(&mut tree, &config).expect("Failed to optimize");

        assert!(tree.context_count() <= initial_count);
        assert_eq!(metrics.context_count, tree.context_count());
        assert!(tree.context_count() > 0);
        assert!(metrics.estimated_memory_bytes > 0);
    }
}
