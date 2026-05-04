//! Context Tree module for variable-order Markov model implementation
//!
//! This module implements context storage and probability estimation for building
//! variable-order Markov models with information-theoretic measures.
//!
//! MEMORY OPTIMIZATIONS:
//! - String interning: Uses StateId instead of String to reduce duplication
//! - On-demand computation: Probabilities calculated when needed, not stored
//! - Cached totals: Avoids recomputing transition counts repeatedly

use crate::config::AnomalyGridConfig;
use crate::context_trie::ContextTrie;
use crate::error::{AnomalyGridError, AnomalyGridResult};
use crate::string_interner::{StateId, StringInterner};
use crate::transition_counts::TransitionCounts;
use std::collections::HashMap;
use std::sync::Arc;

/// A node in the context tree that stores transition statistics.
///
/// All probability queries use the **global** alphabet size (|Σ|) so that
/// Laplace-smoothed distributions are properly normalised across the full
/// symbol set, not just the symbols observed in this context.
#[derive(Debug, Clone)]
pub struct ContextNode {
    counts: TransitionCounts,
    total_count: usize,
    interner: Arc<StringInterner>,
}

impl ContextNode {
    pub fn new(interner: Arc<StringInterner>) -> Self {
        Self {
            counts: TransitionCounts::new(),
            total_count: 0,
            interner,
        }
    }

    pub fn add_transition(&mut self, next_state: &str) {
        let state_id = self.interner.get_or_intern(next_state);
        self.counts.increment(state_id);
        self.total_count += 1;
    }

    pub fn add_transition_by_id(&mut self, state_id: StateId) {
        self.counts.increment(state_id);
        self.total_count += 1;
    }

    pub fn total_count(&self) -> usize {
        self.total_count
    }

    pub fn get_count(&self, next_state: &str) -> usize {
        let state_id = self.interner.get_or_intern(next_state);
        self.counts.get(state_id)
    }

    pub fn get_count_by_id(&self, state_id: StateId) -> usize {
        self.counts.get(state_id)
    }

    /// Number of distinct symbols observed after this context.
    pub fn vocab_size(&self) -> usize {
        self.counts.len()
    }

    pub fn get_state_counts(&self) -> impl Iterator<Item = (StateId, usize)> + '_ {
        self.counts.iter()
    }

    pub fn total_transitions(&self) -> usize {
        self.total_count
    }

    pub fn get_string_counts(&self) -> HashMap<String, usize> {
        self.counts
            .iter()
            .filter_map(|(state_id, count)| self.interner.get_string(state_id).map(|s| (s, count)))
            .collect()
    }

    pub fn counts(&self) -> HashMap<String, usize> {
        self.get_string_counts()
    }

    // ── probability ────────────────────────────────────────────────────

    /// Laplace-smoothed conditional probability using **global** alphabet.
    ///
    /// P(state | context) = (count + α) / (N + α·|Σ|)
    ///
    /// When no transitions have been observed, returns 1/|Σ| (uniform).
    pub fn get_probability(
        &self,
        next_state: &str,
        config: &AnomalyGridConfig,
        global_vocab_size: usize,
    ) -> f64 {
        let state_id = self.interner.get_or_intern(next_state);
        self.get_probability_by_id(state_id, config, global_vocab_size)
    }

    pub fn get_probability_by_id(
        &self,
        state_id: StateId,
        config: &AnomalyGridConfig,
        global_vocab_size: usize,
    ) -> f64 {
        let v = (global_vocab_size as f64).max(1.0);
        if self.total_count == 0 {
            return 1.0 / v;
        }
        let count = self.get_count_by_id(state_id) as f64;
        (count + config.smoothing_alpha)
            / config.smoothing_alpha.mul_add(v, self.total_count as f64)
    }

    // ── information-theoretic measures ─────────────────────────────────

    /// Shannon entropy H(X) = −∑ P(x) log₂ P(x), summed over observed symbols.
    pub fn compute_entropy(&self, config: &AnomalyGridConfig, global_vocab_size: usize) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        self.counts
            .keys()
            .map(|state_id| {
                let p = self.get_probability_by_id(state_id, config, global_vocab_size);
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .sum()
    }

    /// KL divergence D_KL(P ‖ U) from the uniform distribution over the
    /// global alphabet.
    pub fn compute_kl_divergence(
        &self,
        config: &AnomalyGridConfig,
        global_vocab_size: usize,
    ) -> f64 {
        if self.total_count == 0 || global_vocab_size == 0 {
            return 0.0;
        }
        let uniform_prob = 1.0 / global_vocab_size as f64;
        self.counts
            .keys()
            .map(|state_id| {
                let p = self.get_probability_by_id(state_id, config, global_vocab_size);
                if p > 0.0 { p * (p / uniform_prob).log2() } else { 0.0 }
            })
            .sum()
    }

    /// All probabilities as string→f64 map (diagnostic / test use).
    pub fn get_all_probabilities(
        &self,
        config: &AnomalyGridConfig,
        global_vocab_size: usize,
    ) -> HashMap<String, f64> {
        self.counts
            .keys()
            .filter_map(|state_id| {
                self.interner.get_string(state_id).map(|s| {
                    (s, self.get_probability_by_id(state_id, config, global_vocab_size))
                })
            })
            .collect()
    }
}

/// Context tree for storing variable-order Markov chain contexts
///
/// Uses trie-based storage for memory efficiency through prefix sharing
#[derive(Debug, Clone)]
pub struct ContextTree {
    /// Trie-based storage for memory-efficient prefix sharing
    trie: ContextTrie,
    /// Maximum context order (length)
    pub max_order: usize,
    /// String interner for converting between strings and StateIds
    interner: Arc<StringInterner>,
    /// Last-used configuration for probability calculations
    pub(crate) last_config: AnomalyGridConfig,
}

impl ContextTree {
    /// Create a new context tree with specified maximum order
    pub fn new(max_order: usize) -> AnomalyGridResult<Self> {
        if max_order == 0 {
            return Err(AnomalyGridError::invalid_max_order(max_order));
        }

        let interner = Arc::new(StringInterner::new());
        let trie = ContextTrie::new(max_order, Arc::clone(&interner));
        let last_config = AnomalyGridConfig::default();

        Ok(Self {
            trie,
            max_order,
            interner,
            last_config,
        })
    }

    /// Create a new context tree with existing string interner
    pub fn with_interner(
        max_order: usize,
        interner: Arc<StringInterner>,
    ) -> AnomalyGridResult<Self> {
        if max_order == 0 {
            return Err(AnomalyGridError::invalid_max_order(max_order));
        }

        let trie = ContextTrie::new(max_order, Arc::clone(&interner));
        let last_config = AnomalyGridConfig::default();

        Ok(Self {
            trie,
            max_order,
            interner,
            last_config,
        })
    }

    /// Build the context tree from a training sequence
    ///
    /// # Complexity
    /// - Time: O(n × max_order × |alphabet|) where n = sequence length
    /// - Space: O(|alphabet|^max_order) in worst case
    ///
    /// # Performance Guarantees
    /// - Memory usage is bounded by config.memory_limit if set
    /// - Processing time scales linearly with sequence length
    /// - Uses string interning to reduce memory duplication
    pub fn build_from_sequence(
        &mut self,
        sequence: &[String],
        config: &AnomalyGridConfig,
    ) -> AnomalyGridResult<()> {
        // Validate sequence length
        if sequence.len() < config.min_sequence_length {
            return Err(AnomalyGridError::sequence_too_short(
                config.min_sequence_length,
                sequence.len(),
                "context tree building",
            ));
        }

        // Extract contexts of all orders from 1 to max_order
        for window_size in 1..=self.max_order {
            for window in sequence.windows(window_size + 1) {
                // Check memory limit before adding new context
                if let Some(limit) = config.memory_limit {
                    if self.trie.context_count() >= limit {
                        return Err(AnomalyGridError::memory_limit_exceeded(
                            self.trie.context_count(),
                            limit,
                        ));
                    }
                }

                // Convert context to StateIds for trie storage
                let context_state_ids: Vec<StateId> = window[..window_size]
                    .iter()
                    .map(|s| self.interner.get_or_intern(s))
                    .collect();
                let next_state = &window[window_size];

                // Get or create context node in trie
                let node = self.trie.get_or_create_context_data(&context_state_ids);
                node.add_transition(next_state);
            }
        }

        // Store last-used config for future probability queries
        self.last_config = config.clone();

        Ok(())
    }

    /// Global alphabet size derived from the interner.
    pub fn global_vocab_size(&self) -> usize {
        self.interner.len()
    }

    /// Get the transition probability for a given context and next state.
    pub fn get_transition_probability(&self, context: &[String], next_state: &str) -> Option<f64> {
        let context_state_ids: Vec<StateId> = context
            .iter()
            .map(|s| self.interner.get_or_intern(s))
            .collect();
        let gv = self.interner.len();

        self.trie
            .get_context_data(&context_state_ids)
            .map(|node| node.get_probability(next_state, &self.last_config, gv))
    }

    /// Get the transition probability with custom config.
    pub fn get_transition_probability_with_config(
        &self,
        context: &[String],
        next_state: &str,
        config: &AnomalyGridConfig,
    ) -> Option<f64> {
        let context_state_ids: Vec<StateId> = context
            .iter()
            .map(|s| self.interner.get_or_intern(s))
            .collect();
        let gv = self.interner.len();

        self.trie
            .get_context_data(&context_state_ids)
            .map(|node| node.get_probability(next_state, config, gv))
    }

    /// Get the transition probability using StateIds directly.
    pub fn get_transition_probability_by_ids(
        &self,
        context_ids: &[StateId],
        next_state_id: StateId,
        config: &AnomalyGridConfig,
    ) -> Option<f64> {
        let gv = self.interner.len();
        self.trie
            .get_context_data(context_ids)
            .map(|node| node.get_probability_by_id(next_state_id, config, gv))
    }

    /// Get a context node for the given context
    pub fn get_context_node(&self, context: &[String]) -> Option<&ContextNode> {
        // Convert context to StateIds
        let context_state_ids: Vec<StateId> = context
            .iter()
            .map(|s| self.interner.get_or_intern(s))
            .collect();

        self.trie.get_context_data(&context_state_ids)
    }

    /// Get the total count for a given context (for adaptive context selection)
    pub fn get_context_count(&self, context: &[String]) -> Option<usize> {
        self.get_context_node(context)
            .map(ContextNode::total_count)
    }

    /// Get the total count for a given context by StateIds
    pub fn get_context_count_by_ids(&self, context_ids: &[StateId]) -> Option<usize> {
        self.trie
            .get_context_data(context_ids)
            .map(ContextNode::total_count)
    }

    /// Get all contexts of a specific order
    pub fn get_contexts_of_order(&self, order: usize) -> Vec<Vec<String>> {
        self.trie
            .iter_contexts()
            .filter_map(|(state_ids, _)| {
                if state_ids.len() == order {
                    // Convert StateIds back to strings
                    let strings: Option<Vec<String>> = state_ids
                        .iter()
                        .map(|&state_id| self.interner.get_string(state_id))
                        .collect();
                    strings
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get the number of contexts stored
    pub fn context_count(&self) -> usize {
        self.trie.context_count()
    }

    /// Get access to the string interner
    pub fn interner(&self) -> &Arc<StringInterner> {
        &self.interner
    }

    /// Get all contexts as a HashMap for compatibility with existing code
    ///
    /// Note: This creates a temporary HashMap and should be used sparingly
    /// for compatibility with existing tests and code that expects the old interface
    pub fn contexts(&self) -> HashMap<Vec<String>, ContextNode> {
        let mut contexts = HashMap::new();

        for (state_ids, node) in self.trie.iter_contexts() {
            // Convert StateIds back to strings
            if let Some(strings) = state_ids
                .iter()
                .map(|&state_id| self.interner.get_string(state_id))
                .collect::<Option<Vec<String>>>()
            {
                contexts.insert(strings, node.clone());
            }
        }

        contexts
    }

    /// Get the trie for internal operations
    pub(crate) fn trie(&self) -> &ContextTrie {
        &self.trie
    }

    /// Rebuild the trie using a filter predicate; returns number of removed contexts
    pub(crate) fn rebuild_filtered<F>(&mut self, mut keep: F) -> usize
    where
        F: FnMut(&[StateId], &ContextNode) -> bool,
    {
        let original_count = self.trie.context_count();
        let mut new_trie = ContextTrie::new(self.max_order, Arc::clone(&self.interner));

        for (state_ids, node) in self.trie.iter_contexts() {
            if keep(&state_ids, node) {
                let new_node = new_trie.get_or_create_context_data(&state_ids);
                for (state_id, count) in node.get_state_counts() {
                    for _ in 0..count {
                        new_node.add_transition_by_id(state_id);
                    }
                }
            }
        }

        // Avoid pruning everything; if nothing would remain, keep original trie
        if new_trie.context_count() == 0 {
            0
        } else {
            let removed = original_count.saturating_sub(new_trie.context_count());
            self.trie = new_trie;
            removed
        }
    }
}
