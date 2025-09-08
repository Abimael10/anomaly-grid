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
use crate::error::{AnomalyGridError, AnomalyGridResult};
use crate::string_interner::{StateId, StringInterner};
use std::collections::HashMap;
use std::sync::Arc;

/// A node in the context tree that stores transition statistics
/// 
/// Uses StateId for compact storage and computes probabilities on-demand
/// to minimize memory.
#[derive(Debug, Clone)]
pub struct ContextNode {
    /// Raw transition counts using interned state IDs for memory efficiency
    counts: HashMap<StateId, usize>,
    /// Cached total count to avoid recomputation
    total_count: usize,
    /// String interner for converting between strings and StateIds
    interner: Arc<StringInterner>,
}

impl ContextNode {
    /// Create a new empty context node with string interner
    pub fn new(interner: Arc<StringInterner>) -> Self {
        Self {
            counts: HashMap::new(),
            total_count: 0,
            interner,
        }
    }

    /// Add a transition to this context using string interning
    pub fn add_transition(&mut self, next_state: &str) {
        let state_id = self.interner.get_or_intern(next_state);
        *self.counts.entry(state_id).or_insert(0) += 1;
        self.total_count += 1;
    }

    /// Add a transition using StateId directly (internal use)
    pub fn add_transition_by_id(&mut self, state_id: StateId) {
        *self.counts.entry(state_id).or_insert(0) += 1;
        self.total_count += 1;
    }

    /// Get the total number of transitions from this context
    pub fn total_count(&self) -> usize {
        self.total_count
    }

    /// Get the count for a specific next state
    pub fn get_count(&self, next_state: &str) -> usize {
        let state_id = self.interner.get_or_intern(next_state);
        self.counts.get(&state_id).copied().unwrap_or(0)
    }

    /// Get the count for a StateId directly (internal use)
    pub fn get_count_by_id(&self, state_id: StateId) -> usize {
        self.counts.get(&state_id).copied().unwrap_or(0)
    }

    /// Get the number of unique next states
    pub fn vocab_size(&self) -> usize {
        self.counts.len()
    }

    /// Get all state IDs with their counts (internal use)
    pub fn get_state_counts(&self) -> &HashMap<StateId, usize> {
        &self.counts
    }

    /// Get the probability for a specific next state using Laplace smoothing
    /// 
    /// Computes probability on-demand: P(state) = (count + α) / (total + α * |V|)
    pub fn get_probability(&self, next_state: &str, config: &AnomalyGridConfig) -> f64 {
        let state_id = self.interner.get_or_intern(next_state);
        self.get_probability_by_id(state_id, config)
    }

    /// Get probability for a StateId directly (internal use)
    pub fn get_probability_by_id(&self, state_id: StateId, config: &AnomalyGridConfig) -> f64 {
        if self.total_count == 0 {
            return 1.0 / (self.vocab_size() as f64).max(1.0);
        }

        let count = self.get_count_by_id(state_id) as f64;
        let vocab_size = self.vocab_size() as f64;
        
        (count + config.smoothing_alpha) / 
        (self.total_count as f64 + config.smoothing_alpha * vocab_size)
    }

    /// Calculate Shannon entropy on-demand: H(X) = -∑ P(x) log₂ P(x)
    pub fn calculate_entropy(&self, config: &AnomalyGridConfig) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        self.counts
            .keys()
            .map(|&state_id| {
                let p = self.get_probability_by_id(state_id, config);
                if p > 0.0 { -p * p.log2() } else { 0.0 }
            })
            .sum()
    }

    /// Calculate KL divergence from uniform distribution on-demand
    pub fn calculate_kl_divergence(&self, config: &AnomalyGridConfig) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        let uniform_prob = 1.0 / self.vocab_size() as f64;
        
        self.counts
            .keys()
            .map(|&state_id| {
                let p = self.get_probability_by_id(state_id, config);
                if p > 0.0 {
                    p * (p / uniform_prob).log2()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Get all probabilities as a HashMap (for compatibility with existing code)
    /// 
    /// Note: This creates temporary storage and should be used sparingly
    pub fn get_all_probabilities(&self, config: &AnomalyGridConfig) -> HashMap<String, f64> {
        self.counts
            .keys()
            .filter_map(|&state_id| {
                self.interner.get_string(state_id).map(|state_string| {
                    let prob = self.get_probability_by_id(state_id, config);
                    (state_string, prob)
                })
            })
            .collect()
    }
}

// Note: Default implementation removed as ContextNode now requires StringInterner

/// Context tree for storing variable-order Markov chain contexts
#[derive(Debug, Clone)]
pub struct ContextTree {
    /// Map from context sequences to context nodes
    pub contexts: HashMap<Vec<String>, ContextNode>,
    /// Maximum context order (length)
    pub max_order: usize,
}

impl ContextTree {
    /// Create a new context tree with specified maximum order
    pub fn new(max_order: usize) -> AnomalyGridResult<Self> {
        if max_order == 0 {
            return Err(AnomalyGridError::invalid_max_order(max_order));
        }

        Ok(Self {
            contexts: HashMap::new(),
            max_order,
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
    /// - OPTIMIZED: No redundant probability storage during training
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
                    if self.contexts.len() >= limit {
                        return Err(AnomalyGridError::memory_limit_exceeded(
                            self.contexts.len(),
                            limit,
                        ));
                    }
                }

                let context = window[..window_size].to_vec();
                let next_state = &window[window_size];

                let node = self.contexts.entry(context).or_default();
                node.add_transition(next_state.clone());
            }
        }

        // No need to pre-calculate probabilities - they're computed on-demand
        Ok(())
    }

    /// Get the transition probability for a given context and next state
    pub fn get_transition_probability(&self, context: &[String], next_state: &str) -> Option<f64> {
        self.contexts
            .get(context)
            .map(|node| node.get_probability(next_state, &AnomalyGridConfig::default()))
    }

    /// Get the transition probability with custom config
    pub fn get_transition_probability_with_config(
        &self, 
        context: &[String], 
        next_state: &str,
        config: &AnomalyGridConfig
    ) -> Option<f64> {
        self.contexts
            .get(context)
            .map(|node| node.get_probability(next_state, config))
    }

    /// Get a context node for the given context
    pub fn get_context_node(&self, context: &[String]) -> Option<&ContextNode> {
        self.contexts.get(context)
    }

    /// Get all contexts of a specific order
    pub fn get_contexts_of_order(&self, order: usize) -> Vec<&Vec<String>> {
        self.contexts
            .keys()
            .filter(|context| context.len() == order)
            .collect()
    }

    /// Get the number of contexts stored
    pub fn context_count(&self) -> usize {
        self.contexts.len()
    }
}
