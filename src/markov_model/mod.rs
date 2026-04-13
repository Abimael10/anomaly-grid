//! Markov Model module for variable-order Markov chain implementation
//!
//! This module provides a variable-order Markov model with hierarchical
//! context selection and robust probability estimation.

use crate::config::AnomalyGridConfig;
use crate::context_tree::ContextTree;
use crate::error::{AnomalyGridError, AnomalyGridResult};
use std::collections::{HashMap, HashSet};

/// Variable-order Markov model for sequence analysis
#[derive(Debug, Clone)]
pub struct MarkovModel {
    /// The underlying context tree
    context_tree: ContextTree,
    /// Mapping from states to unique IDs
    state_mapping: HashMap<String, usize>,
    /// Reverse mapping from IDs to states
    id_to_state: Vec<String>,
    /// Cached marginal counts for fast probability lookups
    state_counts: HashMap<String, usize>,
    /// Total token count across all training data
    total_tokens: usize,
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
            state_counts: HashMap::new(),
            total_tokens: 0,
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
            state_counts: HashMap::new(),
            total_tokens: 0,
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

        // Prepare vocabulary and counts from the provided sequence
        self.prepare_state_mapping(&[sequence.to_vec()]);

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

    /// Calculate likelihood using precomputed StateIds for faster detection
    pub fn calculate_likelihood_ids(
        &self,
        sequence_ids: &[crate::string_interner::StateId],
        sequence: &[String],
    ) -> f64 {
        if sequence_ids.is_empty() {
            return 1.0;
        }

        if sequence_ids.len() == 1 {
            return self.get_marginal_probability(&sequence[0]);
        }

        let mut likelihood = 1.0;

        for i in 1..sequence_ids.len() {
            let prob =
                self.get_best_context_probability_for_position_ids(sequence_ids, sequence, i);
            likelihood *= prob;
        }

        likelihood
    }

    /// Witten-Bell interpolated probability P(next_state | context).
    ///
    /// Recursively blends the ML estimate at each order with the estimate
    /// at the next-shorter context:
    ///
    ///   P_wb(x | c) = λ(c) · P_ml(x | c) + (1 − λ(c)) · P_wb(x | suffix(c))
    ///
    /// where λ(c) = N(c) / (N(c) + T(c)), N = total count, T = distinct types.
    /// Base case (order 0) is the smoothed unigram.
    pub fn get_best_context_probability(&self, context: &[String], next_state: &str) -> f64 {
        let max_ctx = context.len().min(self.context_tree.max_order);

        // Walk from longest context down to order 1, accumulating interpolation
        let mut prob = self.get_marginal_probability(next_state); // order-0 base

        for context_len in 1..=max_ctx {
            let sub_context = &context[context.len() - context_len..];

            if let Some(node) = self.context_tree.get_context_node(sub_context) {
                let n = node.total_count() as f64;
                let t = node.vocab_size() as f64;
                if n > 0.0 {
                    let lambda = n / (n + t);
                    let gv = self.context_tree.global_vocab_size();
                    let p_ml = node.get_probability(next_state, &self.config, gv);
                    prob = lambda * p_ml + (1.0 - lambda) * prob;
                }
            }
        }

        prob.max(self.config.min_probability)
    }

    /// Witten-Bell interpolated probability using StateIds (fast path).
    pub fn get_best_context_probability_ids(
        &self,
        context_ids: &[crate::string_interner::StateId],
        next_state_id: crate::string_interner::StateId,
        next_state: &str,
    ) -> f64 {
        let max_ctx = context_ids.len().min(self.context_tree.max_order);

        let mut prob = self.get_marginal_probability(next_state);

        for context_len in 1..=max_ctx {
            let sub_context = &context_ids[context_ids.len() - context_len..];

            if let Some(node) = self.context_tree.trie().get_context_data(sub_context) {
                let n = node.total_count() as f64;
                let t = node.vocab_size() as f64;
                if n > 0.0 {
                    let lambda = n / (n + t);
                    let gv = self.context_tree.global_vocab_size();
                    let p_ml = node.get_probability_by_id(next_state_id, &self.config, gv);
                    prob = lambda * p_ml + (1.0 - lambda) * prob;
                }
            }
        }

        prob.max(self.config.min_probability)
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
    fn build_state_mapping(&mut self, sequence: &[Vec<String>]) {
        let mut unique_states: HashSet<String> = HashSet::new();

        self.state_counts.clear();
        self.total_tokens = 0;
        self.state_mapping.clear();
        self.id_to_state.clear();

        for seq in sequence {
            for token in seq {
                unique_states.insert(token.clone());
                *self.state_counts.entry(token.clone()).or_insert(0) += 1;
                self.total_tokens += 1;
            }
        }

        for (id, state) in unique_states.drain().enumerate() {
            self.state_mapping.insert(state.clone(), id);
            self.id_to_state.push(state);
        }
    }

    /// Prepare vocabulary and counts across multiple sequences
    pub(crate) fn prepare_state_mapping(&mut self, sequences: &[Vec<String>]) {
        self.build_state_mapping(sequences);
    }

    /// Train using an existing vocabulary prepared from one or more sequences
    pub(crate) fn train_with_existing_vocab(
        &mut self,
        sequence: &[String],
    ) -> AnomalyGridResult<()> {
        if sequence.len() < self.config.min_sequence_length {
            return Err(AnomalyGridError::sequence_too_short(
                self.config.min_sequence_length,
                sequence.len(),
                "model training",
            ));
        }

        self.context_tree
            .build_from_sequence(sequence, &self.config)
    }

    /// Witten-Bell interpolated probability at a specific position.
    fn get_best_context_probability_for_position(
        &self,
        sequence: &[String],
        position: usize,
    ) -> f64 {
        let context = &sequence[..position];
        self.get_best_context_probability(context, &sequence[position])
    }

    /// Witten-Bell interpolated probability at a specific position (StateId fast path).
    fn get_best_context_probability_for_position_ids(
        &self,
        sequence_ids: &[crate::string_interner::StateId],
        sequence: &[String],
        position: usize,
    ) -> f64 {
        let context_ids = &sequence_ids[..position];
        self.get_best_context_probability_ids(
            context_ids,
            sequence_ids[position],
            &sequence[position],
        )
    }

    /// Smoothed unigram probability (order-0 base case).
    ///
    /// P(x) = (count(x) + α) / (N + α·|Σ|)
    pub fn get_marginal_probability(&self, state: &str) -> f64 {
        let gv = self.context_tree.global_vocab_size().max(1) as f64;
        if self.total_tokens == 0 {
            return (1.0 / gv).max(self.config.min_probability);
        }
        let raw_count = self.state_counts.get(state).copied().unwrap_or(0) as f64;
        let smoothed = (raw_count + self.config.smoothing_alpha)
            / (self.total_tokens as f64 + self.config.smoothing_alpha * gv);
        smoothed.max(self.config.min_probability)
    }

    /// Background probability for completely unseen symbols.
    pub fn get_background_probability(&self, _state: &str) -> f64 {
        let gv = self.context_tree.global_vocab_size().max(1) as f64;
        let bg = self.config.smoothing_alpha
            / (self.total_tokens as f64 + self.config.smoothing_alpha * gv);
        bg.max(self.config.min_probability)
    }
}
