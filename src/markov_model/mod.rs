//! Markov Model module for variable-order Markov chain implementation
//!
//! This module provides a variable-order Markov model with hierarchical
//! context selection and robust probability estimation.

use crate::config::AnomalyGridConfig;
use crate::context_tree::ContextTree;
use crate::error::{AnomalyGridError, AnomalyGridResult};
use std::collections::HashMap;

/// Variable-order Markov model for sequence analysis
#[derive(Debug, Clone)]
pub struct MarkovModel {
    /// The underlying context tree
    context_tree: ContextTree,
    /// Mapping from states to unique IDs
    state_mapping: HashMap<String, usize>,
    /// Reverse mapping from IDs to states
    id_to_state: Vec<String>,
    /// Configuration parameters
    config: AnomalyGridConfig,
}

impl MarkovModel {
    /// Create a new Markov model with specified maximum order
    pub fn new(max_order: usize) -> AnomalyGridResult<Self> {
        if max_order == 0 {
            return Err(AnomalyGridError::invalid_max_order(max_order));
        }

        let config = AnomalyGridConfig::default().with_max_order(max_order)?;

        Ok(Self {
            context_tree: ContextTree::new(max_order)?,
            state_mapping: HashMap::new(),
            id_to_state: Vec::new(),
            config,
        })
    }

    /// Create a new Markov model with custom configuration
    pub fn with_config(config: AnomalyGridConfig) -> AnomalyGridResult<Self> {
        config.validate()?;

        Ok(Self {
            context_tree: ContextTree::new(config.max_order)?,
            state_mapping: HashMap::new(),
            id_to_state: Vec::new(),
            config,
        })
    }

    /// Train the model on a sequence
    ///
    /// # Complexity
    /// - Time: O(n × max_order × |alphabet|) where n = sequence length
    /// - Space: O(|alphabet|^max_order) in worst case
    ///
    /// # Performance Guarantees
    /// - Memory usage is bounded by config.memory_limit if set
    /// - Validates sequence length against config.min_sequence_length
    pub fn train(&mut self, sequence: &[String]) -> AnomalyGridResult<()> {
        // Validate sequence length
        if sequence.len() < self.config.min_sequence_length {
            return Err(AnomalyGridError::sequence_too_short(
                self.config.min_sequence_length,
                sequence.len(),
                "model training",
            ));
        }

        // Build state mapping
        self.build_state_mapping(sequence);

        // Build context tree with configuration
        self.context_tree
            .build_from_sequence(sequence, &self.config)
    }

    /// Calculate the likelihood of a sequence under the model
    pub fn calculate_likelihood(&self, sequence: &[String]) -> f64 {
        if sequence.is_empty() {
            return 1.0; // Empty sequences have likelihood 1
        }

        if sequence.len() == 1 {
            // Single-element sequences: return the marginal probability of that element
            return self.get_marginal_probability(&sequence[0]);
        }

        let mut likelihood = 1.0;

        for i in 1..sequence.len() {
            let prob = self.get_best_context_probability_for_position(sequence, i);
            likelihood *= prob;
        }

        likelihood
    }

    /// Get the best context probability using adaptive hierarchical context selection
    pub fn get_best_context_probability(&self, context: &[String], next_state: &str) -> f64 {
        // Check if the state is in the global vocabulary
        if self.state_mapping.contains_key(next_state) {
            // For states in global vocabulary, use normalized probability
            for context_len in (1..=context.len().min(self.context_tree.max_order)).rev() {
                let sub_context = &context[context.len() - context_len..];

                if let Some(prob) = self.context_tree.get_transition_probability_normalized(
                    sub_context,
                    next_state,
                    &self.config,
                    &self.state_mapping,
                ) {
                    // Check if this context has sufficient data for reliable estimation
                    if self.context_has_sufficient_data(sub_context) {
                        return prob;
                    }
                    // If insufficient data, continue to shorter contexts
                }
            }
        }

        // Fallback to background probability for unseen transitions or unknown states
        self.get_background_probability(next_state)
    }

    /// Get the maximum order of the model
    pub fn max_order(&self) -> usize {
        self.config.max_order
    }

    /// Get the configuration
    pub fn config(&self) -> &AnomalyGridConfig {
        &self.config
    }

    /// Get the state mapping
    pub fn state_mapping(&self) -> &HashMap<String, usize> {
        &self.state_mapping
    }

    /// Get the context tree
    pub fn context_tree(&self) -> &ContextTree {
        &self.context_tree
    }

    /// Get mutable access to the context tree (for optimizations)
    pub fn context_tree_mut(&mut self) -> &mut ContextTree {
        &mut self.context_tree
    }

    /// Build state mapping from sequence
    fn build_state_mapping(&mut self, sequence: &[String]) {
        let mut unique_states: std::collections::HashSet<String> =
            sequence.iter().cloned().collect();

        self.state_mapping.clear();
        self.id_to_state.clear();

        for (id, state) in unique_states.drain().enumerate() {
            self.state_mapping.insert(state.clone(), id);
            self.id_to_state.push(state);
        }
    }

    /// Get probability for a specific position in a sequence using adaptive context selection
    fn get_best_context_probability_for_position(
        &self,
        sequence: &[String],
        position: usize,
    ) -> f64 {
        let next_state = &sequence[position];
        let max_context_len = position.min(self.context_tree.max_order);

        // Adaptive context selection: try contexts from longest to shortest,
        // but only use contexts with sufficient data
        for context_len in (1..=max_context_len).rev() {
            let context = &sequence[position - context_len..position];

            if let Some(prob) = self.context_tree.get_transition_probability_normalized(
                context,
                next_state,
                &self.config,
                &self.state_mapping,
            ) {
                // Check if this context has sufficient data for reliable estimation
                if self.context_has_sufficient_data(context) {
                    return prob;
                }
                // If insufficient data, continue to shorter contexts
            }
        }

        // Fallback to background probability
        self.get_background_probability(next_state)
    }

    /// Check if a context has sufficient data for reliable probability estimation
    fn context_has_sufficient_data(&self, context: &[String]) -> bool {
        // Calculate minimum count threshold based on context length
        // Use more lenient thresholds to allow higher orders to work with reasonable data
        let min_count_threshold = match context.len() {
            1 => 1, // Order 1: need at least 1 observation
            2 => 2, // Order 2: need at least 2 observations
            3 => 3, // Order 3: need at least 3 observations
            4 => 4, // Order 4: need at least 4 observations
            _ => 5, // Order 5+: need at least 5 observations
        };

        // Check if context has sufficient total count
        if let Some(context_count) = self.context_tree.get_context_count(context) {
            context_count >= min_count_threshold
        } else {
            false // Context doesn't exist
        }
    }

    /// Get marginal probability of a state from the training data
    pub fn get_marginal_probability(&self, state: &str) -> f64 {
        // Calculate marginal probability by counting occurrences across all contexts
        let mut total_count = 0;
        let mut state_count = 0;

        // Iterate through all contexts in the context tree
        for (_context_states, context_node) in self.context_tree.trie().iter_contexts() {
            let context_total = context_node.total_count();
            total_count += context_total;

            // Count occurrences of our target state in this context
            state_count += context_node.get_count(state);
        }

        if total_count == 0 {
            return self.get_background_probability(state);
        }

        // Apply smoothing
        let vocab_size = self.state_mapping.len() as f64;
        let smoothed_count = state_count as f64 + self.config.smoothing_alpha;
        let smoothed_total = total_count as f64 + self.config.smoothing_alpha * vocab_size;

        (smoothed_count / smoothed_total).max(self.config.min_probability)
    }

    /// Get background probability for unseen transitions (public for adaptive scoring)
    pub fn get_background_probability(&self, state: &str) -> f64 {
        // If state is known, use uniform probability over known states
        if self.state_mapping.contains_key(state) {
            let vocab_size = self.state_mapping.len() as f64;
            (1.0 / (vocab_size + 1.0)).max(self.config.min_probability)
        } else {
            // For completely unknown states, use a reasonable small probability
            // This ensures that unseen states can still be scored as anomalies
            let vocab_size = self.state_mapping.len() as f64;
            let unknown_state_prob = 1.0 / (vocab_size + 2.0); // +2 to account for unknown states
            unknown_state_prob.max(self.config.min_probability)
        }
    }
}
