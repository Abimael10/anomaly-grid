//! Context Tree module for variable-order Markov model implementation
//!
//! This module implements context storage and probability estimation for building
//! variable-order Markov models with information-theoretic measures.

use crate::config::AnomalyGridConfig;
use crate::error::{AnomalyGridError, AnomalyGridResult};
use std::collections::HashMap;

/// A node in the context tree that stores transition statistics
#[derive(Debug, Clone)]
pub struct ContextNode {
    /// Raw transition counts for each next state
    pub counts: HashMap<String, usize>,
    /// Normalized transition probabilities
    pub probabilities: HashMap<String, f64>,
    /// Shannon entropy of the transition distribution
    pub entropy: f64,
    /// KL divergence from uniform distribution
    pub kl_divergence: f64,
}

impl ContextNode {
    /// Create a new empty context node
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            probabilities: HashMap::new(),
            entropy: 0.0,
            kl_divergence: 0.0,
        }
    }

    /// Add a transition to this context
    pub fn add_transition(&mut self, next_state: String) {
        *self.counts.entry(next_state).or_insert(0) += 1;
    }

    /// Calculate probabilities and information-theoretic measures
    pub fn calculate_probabilities(&mut self, config: &AnomalyGridConfig) {
        if self.counts.is_empty() {
            return;
        }

        let total_count: usize = self.counts.values().sum();
        let vocab_size = self.counts.len();

        // Calculate probabilities with configurable Laplace smoothing
        self.probabilities.clear();

        for (state, &count) in &self.counts {
            let smoothed_prob = (count as f64 + config.smoothing_alpha)
                / (total_count as f64 + config.smoothing_alpha * vocab_size as f64);
            self.probabilities.insert(state.clone(), smoothed_prob);
        }

        // Calculate Shannon entropy: H(X) = -∑ P(x) log₂ P(x)
        // CRITICAL FIX: Correct formula is -p * log2(p), not -p.log2()
        self.entropy = self
            .probabilities
            .values()
            .map(|&p| if p > 0.0 { -p * p.log2() } else { 0.0 })
            .sum();

        // Calculate KL divergence from uniform distribution
        let uniform_prob = 1.0 / vocab_size as f64;
        self.kl_divergence = self
            .probabilities
            .values()
            .map(|&p| {
                if p > 0.0 {
                    p * (p / uniform_prob).log2()
                } else {
                    0.0
                }
            })
            .sum();
    }
}

impl Default for ContextNode {
    fn default() -> Self {
        Self::new()
    }
}

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

        // Calculate probabilities for all contexts
        for node in self.contexts.values_mut() {
            node.calculate_probabilities(config);
        }

        Ok(())
    }

    /// Get the transition probability for a given context and next state
    pub fn get_transition_probability(&self, context: &[String], next_state: &str) -> Option<f64> {
        self.contexts
            .get(context)
            .and_then(|node| node.probabilities.get(next_state))
            .copied()
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
