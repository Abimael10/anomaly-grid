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
        if sequence.len() < 2 {
            return 1.0; // Empty or single-element sequences have likelihood 1
        }

        let mut likelihood = 1.0;

        for i in 1..sequence.len() {
            let prob = self.get_best_context_probability_for_position(sequence, i);
            likelihood *= prob;
        }

        likelihood
    }

    /// Get the best context probability using hierarchical context selection
    pub fn get_best_context_probability(&self, context: &[String], next_state: &str) -> f64 {
        // Try contexts from longest to shortest (hierarchical selection)
        for context_len in (1..=context.len().min(self.context_tree.max_order)).rev() {
            let sub_context = &context[context.len() - context_len..];

            if let Some(prob) = self
                .context_tree
                .get_transition_probability(sub_context, next_state)
            {
                return prob;
            }
        }

        // Fallback to background probability for unseen transitions
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

    /// Get probability for a specific position in a sequence
    fn get_best_context_probability_for_position(
        &self,
        sequence: &[String],
        position: usize,
    ) -> f64 {
        let next_state = &sequence[position];
        let max_context_len = position.min(self.context_tree.max_order);

        // Try contexts from longest to shortest
        for context_len in (1..=max_context_len).rev() {
            let context = &sequence[position - context_len..position];

            if let Some(prob) = self
                .context_tree
                .get_transition_probability(context, next_state)
            {
                return prob;
            }
        }

        // Fallback to background probability
        self.get_background_probability(next_state)
    }

    /// Get background probability for unseen transitions
    fn get_background_probability(&self, state: &str) -> f64 {
        // If state is known, use uniform probability over known states
        if self.state_mapping.contains_key(state) {
            let vocab_size = self.state_mapping.len() as f64;
            (1.0 / (vocab_size + 1.0)).max(self.config.min_probability)
        } else {
            // For completely unknown states, use very small probability
            self.config.min_probability
        }
    }
}
