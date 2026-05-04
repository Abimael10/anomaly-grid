//! Performance and memory optimisation utilities.
//!
//! Trie pruning + a tiny [`PerformanceMetrics`] struct for monitoring.
//! Pruning rebuilds the trie keeping only contexts that pass a predicate
//! (frequency, entropy, top-N) — see [`ContextTree::rebuild_filtered`]
//! in the parent crate.

use crate::context_tree::ContextTree;
use crate::error::AnomalyGridResult;
use crate::string_interner::StateId;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Lightweight metrics for training/detection runs.
#[derive(Debug, Clone, Default)]
pub struct PerformanceMetrics {
    pub training_time_ms: u64,
    pub detection_time_ms: u64,
    pub context_count: usize,
    pub estimated_memory_bytes: usize,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Elements per second during training.
    pub fn training_throughput(&self, sequence_length: usize) -> f64 {
        if self.training_time_ms == 0 {
            return 0.0;
        }
        sequence_length as f64 / (self.training_time_ms as f64 / 1000.0)
    }

    /// Elements per second during detection.
    pub fn detection_throughput(&self, sequence_length: usize) -> f64 {
        if self.detection_time_ms == 0 {
            return 0.0;
        }
        sequence_length as f64 / (self.detection_time_ms as f64 / 1000.0)
    }
}

impl ContextTree {
    /// Drop contexts observed fewer than `min_count` times.
    /// Returns the number of contexts removed.
    pub fn prune_low_frequency_contexts(&mut self, min_count: usize) -> AnomalyGridResult<usize> {
        if min_count == 0 {
            return Ok(0);
        }
        self.rebuild_filtered(|_, node| node.total_count() >= min_count)
    }

    /// Drop contexts whose conditional distribution has entropy below
    /// `min_entropy` (highly predictable contexts).
    pub fn prune_low_entropy_contexts(&mut self, min_entropy: f64) -> AnomalyGridResult<usize> {
        if min_entropy <= 0.0 {
            return Ok(0);
        }
        let cfg = self.last_config.clone();
        let gv = self.global_vocab_size();
        self.rebuild_filtered(|_, node| node.compute_entropy(&cfg, gv) >= min_entropy)
    }

    /// Keep only the top-`max_contexts` most frequently observed contexts.
    pub fn limit_context_count(&mut self, max_contexts: usize) -> AnomalyGridResult<usize> {
        if max_contexts == 0 {
            return Ok(0);
        }
        let original_count = self.trie().context_count();
        if original_count <= max_contexts {
            return Ok(0);
        }

        let mut contexts: Vec<(Vec<StateId>, usize)> = self
            .trie()
            .iter_contexts()
            .map(|(state_ids, node)| (state_ids, node.total_count()))
            .collect();
        contexts.sort_by(|a, b| b.1.cmp(&a.1));
        let keep: HashSet<Vec<StateId>> = contexts
            .into_iter()
            .take(max_contexts)
            .map(|(ids, _)| ids)
            .collect();

        self.rebuild_filtered(|state_ids, _| keep.contains(state_ids))
    }

    /// Estimate the memory used by the trie.
    pub fn estimate_memory_usage(&self) -> usize {
        self.trie().memory_usage()
    }

    /// Aggregate context-level statistics for diagnostics.
    pub fn get_context_statistics(&self) -> ContextStatistics {
        let mut stats = ContextStatistics::new();
        stats.total_contexts = self.trie().context_count();

        for (state_ids, node) in self.trie().iter_contexts() {
            let order = state_ids.len();
            *stats.contexts_by_order.entry(order).or_insert(0) += 1;
            stats.total_transitions += node.total_count();
        }

        if stats.total_contexts > 0 {
            stats.avg_frequency = stats.total_transitions as f64 / stats.total_contexts as f64;
        }
        stats
    }
}

/// Aggregate statistics about a context tree.
#[derive(Debug, Clone, Default)]
pub struct ContextStatistics {
    pub total_contexts: usize,
    pub total_transitions: usize,
    pub avg_frequency: f64,
    pub contexts_by_order: HashMap<usize, usize>,
}

impl ContextStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Contexts per megabyte of memory.
    pub fn memory_efficiency(&self, memory_bytes: usize) -> f64 {
        if memory_bytes == 0 {
            return 0.0;
        }
        self.total_contexts as f64 / (memory_bytes as f64 / 1_048_576.0)
    }

    /// Average transitions per context.
    pub fn compression_ratio(&self) -> f64 {
        if self.total_contexts == 0 {
            return 0.0;
        }
        self.total_transitions as f64 / self.total_contexts as f64
    }
}

/// Configuration for [`optimize_context_tree`].
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub enable_pruning: bool,
    pub min_context_count: usize,
    pub min_entropy: f64,
    pub max_contexts: Option<usize>,
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
    pub fn for_low_memory() -> Self {
        Self {
            enable_pruning: true,
            min_context_count: 3,
            min_entropy: 0.2,
            max_contexts: Some(10_000),
            enable_monitoring: true,
        }
    }

    pub fn for_high_accuracy() -> Self {
        Self {
            enable_pruning: false,
            min_context_count: 1,
            min_entropy: 0.0,
            max_contexts: None,
            enable_monitoring: true,
        }
    }

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

/// Apply pruning + size limits to a context tree, in place.
pub fn optimize_context_tree(
    tree: &mut ContextTree,
    config: &OptimizationConfig,
) -> AnomalyGridResult<PerformanceMetrics> {
    let start_time = Instant::now();

    if config.enable_pruning {
        if config.min_context_count > 1 {
            let _ = tree.prune_low_frequency_contexts(config.min_context_count)?;
        }
        if config.min_entropy > 0.0 {
            let _ = tree.prune_low_entropy_contexts(config.min_entropy)?;
        }
        if let Some(max_contexts) = config.max_contexts {
            let _ = tree.limit_context_count(max_contexts)?;
        }
    }

    Ok(PerformanceMetrics {
        training_time_ms: start_time.elapsed().as_millis() as u64,
        detection_time_ms: 0,
        context_count: tree.context_count(),
        estimated_memory_bytes: tree.estimate_memory_usage(),
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::config::AnomalyGridConfig;

    #[test]
    fn performance_metrics_throughput() {
        let mut metrics = PerformanceMetrics::new();
        metrics.training_time_ms = 100;
        metrics.detection_time_ms = 50;
        assert!((metrics.training_throughput(1000) - 10_000.0).abs() < f64::EPSILON);
        assert!((metrics.detection_throughput(500) - 10_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn pruning_drops_low_frequency_contexts() -> AnomalyGridResult<()> {
        let mut tree = ContextTree::new(2)?;
        let config = AnomalyGridConfig::default();

        let high_freq: Vec<String> = std::iter::repeat_n("X".to_string(), 5)
            .chain(std::iter::repeat_n("A".to_string(), 10))
            .collect();
        tree.build_from_sequence(&high_freq, &config)?;

        let low_freq = vec!["Y".to_string(), "B".to_string()];
        tree.build_from_sequence(&low_freq, &config)?;

        let initial = tree.context_count();
        assert!(initial > 0);

        let pruned = tree.prune_low_frequency_contexts(5)?;
        assert!(pruned > 0);
        assert!(tree.context_count() < initial);
        Ok(())
    }

    #[test]
    fn statistics_distribute_orders() -> AnomalyGridResult<()> {
        let mut tree = ContextTree::new(2)?;
        let config = AnomalyGridConfig::default();
        tree.build_from_sequence(
            &["X".to_string(), "A".to_string(), "B".to_string()],
            &config,
        )?;
        tree.build_from_sequence(
            &["Y".to_string(), "Z".to_string(), "C".to_string()],
            &config,
        )?;
        let stats = tree.get_context_statistics();
        assert!(stats.total_contexts > 0);
        assert!(stats.contexts_by_order.contains_key(&1));
        assert!(stats.contexts_by_order.contains_key(&2));
        Ok(())
    }

    #[test]
    fn optimize_respects_config() -> AnomalyGridResult<()> {
        let mut tree = ContextTree::new(2)?;
        let cfg_build = AnomalyGridConfig::default();

        for i in 1..=5 {
            let sequence: Vec<String> = (0..i + 2).map(|j| format!("S{}", j % 3)).collect();
            tree.build_from_sequence(&sequence, &cfg_build)?;
        }

        let initial = tree.context_count();
        assert!(initial > 0);

        let config = OptimizationConfig {
            enable_pruning: true,
            min_context_count: 2,
            min_entropy: 0.0,
            max_contexts: Some(8),
            enable_monitoring: true,
        };
        let metrics = optimize_context_tree(&mut tree, &config)?;
        assert!(tree.context_count() <= initial);
        assert_eq!(metrics.context_count, tree.context_count());
        assert!(tree.context_count() > 0);
        Ok(())
    }
}
