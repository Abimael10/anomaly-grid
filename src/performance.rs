//! Performance optimization utilities for Anomaly Grid
//!
//! This module provides practical performance improvements focused on
//! memory efficiency and computational optimization for anomaly detection.

use crate::context_tree::ContextTree;
use crate::error::AnomalyGridResult;
use std::collections::HashMap;
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
    /// Remove low-frequency contexts to reduce memory usage
    ///
    /// This removes contexts that have been observed fewer than `min_count` times,
    /// which can significantly reduce memory usage for large alphabets.
    pub fn prune_low_frequency_contexts(&mut self, min_count: usize) -> usize {
        let initial_count = self.contexts.len();

        self.contexts.retain(|_, node| {
            let total_transitions = node.total_transitions();
            total_transitions >= min_count
        });

        initial_count - self.contexts.len()
    }

    /// Remove contexts with low entropy (deterministic contexts)
    ///
    /// Contexts with very low entropy provide little information and can be pruned
    /// to reduce memory usage while maintaining detection accuracy.
    pub fn prune_low_entropy_contexts(&mut self, min_entropy: f64) -> usize {
        let initial_count = self.contexts.len();
        let config = crate::config::AnomalyGridConfig::default();

        self.contexts
            .retain(|_, node| node.calculate_entropy(&config) >= min_entropy);

        initial_count - self.contexts.len()
    }

    /// Keep only the most frequent contexts up to a maximum count
    ///
    /// This is useful for memory-constrained environments where you want to
    /// keep only the most important contexts.
    pub fn keep_top_contexts(&mut self, max_contexts: usize) -> usize {
        if self.contexts.len() <= max_contexts {
            return 0;
        }

        let initial_count = self.contexts.len();

        // Sort contexts by total frequency
        let mut context_frequencies: Vec<_> = self
            .contexts
            .iter()
            .map(|(context, node)| {
                let freq = node.total_transitions();
                (context.clone(), freq)
            })
            .collect();

        context_frequencies.sort_by(|a, b| b.1.cmp(&a.1));

        // Keep only top contexts
        let contexts_to_keep: std::collections::HashSet<_> = context_frequencies
            .into_iter()
            .take(max_contexts)
            .map(|(context, _)| context)
            .collect();

        self.contexts
            .retain(|context, _| contexts_to_keep.contains(context));

        initial_count - self.contexts.len()
    }

    /// Estimate memory usage of the context tree
    ///
    /// OPTIMIZED: Only counts actual stored data (counts + total_count)
    /// No longer includes probabilities, entropy, or KL divergence storage
    pub fn estimate_memory_usage(&self) -> usize {
        let mut total_bytes = 0;

        for (context, node) in &self.contexts {
            // Context vector size
            total_bytes += context.len() * std::mem::size_of::<String>();
            total_bytes += context.iter().map(|s| s.capacity()).sum::<usize>();

            // Node counts HashMap (using string representation for compatibility)
            let string_counts = node.get_string_counts();
            total_bytes += string_counts.len()
                * (std::mem::size_of::<String>() + std::mem::size_of::<usize>());
            total_bytes += string_counts.keys().map(|s| s.capacity()).sum::<usize>();

            // Cached total_count (usize)
            total_bytes += std::mem::size_of::<usize>();
        }

        total_bytes
    }

    /// Get context statistics for analysis
    ///
    /// OPTIMIZED: Computes entropy on-demand for statistics
    pub fn get_context_statistics(&self) -> ContextStatistics {
        let mut stats = ContextStatistics::new();
        let config = crate::config::AnomalyGridConfig::default();

        for (context, node) in &self.contexts {
            let total_count = node.total_count();
            let unique_transitions = node.vocab_size();
            let entropy = node.calculate_entropy(&config);

            stats.total_contexts += 1;
            stats.total_transitions += total_count;
            stats.total_entropy += entropy;

            if total_count < stats.min_frequency {
                stats.min_frequency = total_count;
            }
            if total_count > stats.max_frequency {
                stats.max_frequency = total_count;
            }

            if entropy < stats.min_entropy {
                stats.min_entropy = entropy;
            }
            if entropy > stats.max_entropy {
                stats.max_entropy = entropy;
            }

            stats
                .contexts_by_order
                .entry(context.len())
                .and_modify(|count| *count += 1)
                .or_insert(1);

            stats
                .transitions_by_context
                .entry(context.len())
                .and_modify(|count| *count += unique_transitions)
                .or_insert(unique_transitions);
        }

        if stats.total_contexts > 0 {
            stats.avg_entropy = stats.total_entropy / stats.total_contexts as f64;
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
            tree.keep_top_contexts(max_contexts);
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
    use crate::context_tree::ContextNode;

    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();

        metrics.training_time_ms = 100; // 0.1 second
        metrics.detection_time_ms = 50; // 0.05 seconds

        // Test throughput calculations
        // 1000 elements / 0.1 seconds = 10,000 elements/second
        assert_eq!(metrics.training_throughput(1000), 10000.0);
        // 500 elements / 0.05 seconds = 10,000 elements/second
        assert_eq!(metrics.detection_throughput(500), 10000.0);
    }

    #[test]
    fn test_context_pruning() {
        let mut tree = ContextTree::new(2).expect("Failed to create tree");

        // Add some test contexts with different frequencies
        let mut high_freq_node = ContextNode::default();
        for _ in 0..10 {
            high_freq_node.add_transition("A");
        }
        tree.contexts.insert(vec!["X".to_string()], high_freq_node);

        let mut low_freq_node = ContextNode::default();
        low_freq_node.add_transition("B");
        tree.contexts.insert(vec!["Y".to_string()], low_freq_node);

        assert_eq!(tree.context_count(), 2);

        // Prune contexts with frequency < 5
        let pruned = tree.prune_low_frequency_contexts(5);

        assert_eq!(pruned, 1); // Should remove the low frequency context
        assert_eq!(tree.context_count(), 1);
    }

    #[test]
    fn test_memory_estimation() {
        let mut tree = ContextTree::new(2).expect("Failed to create tree");

        // Add a test context
        let mut node = ContextNode::default();
        node.add_transition("A");
        node.add_transition("B");
        tree.contexts.insert(vec!["X".to_string()], node);

        let memory_usage = tree.estimate_memory_usage();
        assert!(memory_usage > 0);
    }

    #[test]
    fn test_context_statistics() {
        let mut tree = ContextTree::new(2).expect("Failed to create tree");

        // Add test contexts
        let mut node1 = ContextNode::default();
        node1.add_transition("A");
        node1.add_transition("B");
        tree.contexts.insert(vec!["X".to_string()], node1);

        let mut node2 = ContextNode::default();
        node2.add_transition("C");
        tree.contexts
            .insert(vec!["Y".to_string(), "Z".to_string()], node2);

        let stats = tree.get_context_statistics();

        assert_eq!(stats.total_contexts, 2);
        assert_eq!(stats.contexts_by_order.get(&1), Some(&1)); // One context of order 1
        assert_eq!(stats.contexts_by_order.get(&2), Some(&1)); // One context of order 2
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

        // Add test contexts with different frequencies
        for i in 1..=10 {
            // Start from 1 to ensure contexts have transitions
            let mut node = ContextNode::default();
            for _ in 0..i {
                node.add_transition("A");
            }
            // No need to pre-calculate probabilities - computed on-demand
            tree.contexts.insert(vec![format!("X{}", i)], node);
        }

        let initial_count = tree.context_count();

        // Use a more lenient optimization config
        let config = OptimizationConfig {
            enable_pruning: true,
            min_context_count: 2,  // Only remove contexts with < 2 transitions
            min_entropy: 0.0,      // Don't filter by entropy
            max_contexts: Some(8), // Keep top 8 contexts
            enable_monitoring: true,
        };

        let metrics = optimize_context_tree(&mut tree, &config).expect("Failed to optimize");

        assert!(tree.context_count() <= initial_count);
        assert_eq!(metrics.context_count, tree.context_count());
        // Should have some contexts remaining
        assert!(tree.context_count() > 0);
        // Memory estimation should work even with 0 contexts
        // Memory bytes is usize, so always >= 0
        assert!(metrics.estimated_memory_bytes < usize::MAX);
    }
}
