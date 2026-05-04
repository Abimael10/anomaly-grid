//! Variable-order Markov model with **Witten-Bell** interpolation.
//!
//! ## Smoothing
//!
//! For a context `c` of length `k`, the conditional probability of the
//! next symbol `x` is the recursive interpolation
//!
//! ```text
//! P_wb(x | c) = λ(c) · P_ml(x | c) + (1 − λ(c)) · P_wb(x | suffix(c))
//! ```
//!
//! where:
//!
//! - `P_ml(x | c) = (count(c, x) + α) / (N(c) + α·|Σ|)` is the Laplace
//!   maximum-likelihood estimate at this order;
//! - `λ(c) = N(c) / (N(c) + T(c))`, with `N(c)` the total observations
//!   in context `c` and `T(c)` the number of distinct continuations seen
//!   — Witten-Bell's classic discount;
//! - The base case is the order-0 unigram
//!   `P(x) = (count(x) + α) / (N + α·|Σ|)`.
//!
//! Unseen contexts contribute nothing to the interpolation but the
//! lower-order estimate still flows through, so backoff is *smooth* and
//! never drops to zero.
//!
//! ## Numerical stability
//!
//! The detection path uses [`MarkovModel::log_likelihood_bits_per_symbol`]
//! which sums `log₂ P(xᵢ | context)` directly. Long sequences cannot
//! underflow because we never multiply many small probabilities.

use crate::config::AnomalyGridConfig;
use crate::context_tree::ContextTree;
use crate::error::{AnomalyGridError, AnomalyGridResult};
use crate::string_interner::StateId;
use std::collections::{HashMap, HashSet};

/// Variable-order Markov model.
#[derive(Debug, Clone)]
pub struct MarkovModel {
    context_tree: ContextTree,
    state_mapping: HashMap<String, usize>,
    state_counts: HashMap<String, usize>,
    total_tokens: usize,
    /// Alphabet size frozen at end of training. Detection paths that
    /// intern previously-unseen symbols would otherwise grow the live
    /// `|Σ|` mid-scoring, which changes Laplace-smoothed probabilities
    /// and breaks parallel determinism (PRs racing to intern fresh
    /// symbols see different orders).
    frozen_vocab_size: usize,
    config: AnomalyGridConfig,
}

impl MarkovModel {
    pub fn new(max_order: usize) -> AnomalyGridResult<Self> {
        if max_order == 0 {
            return Err(AnomalyGridError::invalid_max_order(max_order));
        }
        let config = AnomalyGridConfig::default().with_max_order(max_order)?;
        Ok(Self {
            context_tree: ContextTree::new(max_order)?,
            state_mapping: HashMap::new(),
            state_counts: HashMap::new(),
            total_tokens: 0,
            frozen_vocab_size: 0,
            config,
        })
    }

    pub fn with_config(config: AnomalyGridConfig) -> AnomalyGridResult<Self> {
        config.validate()?;
        Ok(Self {
            context_tree: ContextTree::new(config.max_order)?,
            state_mapping: HashMap::new(),
            state_counts: HashMap::new(),
            total_tokens: 0,
            frozen_vocab_size: 0,
            config,
        })
    }

    pub fn train(&mut self, sequence: &[String]) -> AnomalyGridResult<()> {
        if sequence.len() < self.config.min_sequence_length {
            return Err(AnomalyGridError::sequence_too_short(
                self.config.min_sequence_length,
                sequence.len(),
                "model training",
            ));
        }
        self.prepare_state_mapping(&[sequence.to_vec()]);
        self.context_tree
            .build_from_sequence(sequence, &self.config)?;
        self.frozen_vocab_size = self.context_tree.global_vocab_size();
        Ok(())
    }

    /// Joint chain-rule likelihood `∏ᵢ P_wb(xᵢ | x₁..xᵢ₋₁)`.
    ///
    /// Computed in log space (`exp2(Σ log₂ p)`) so it is stable for
    /// moderately long sequences before exponentiation. Underflows to
    /// zero on very long sequences — callers concerned with long inputs
    /// should use [`MarkovModel::log_likelihood_bits_per_symbol`] which
    /// stays in log space throughout.
    pub fn calculate_likelihood(&self, sequence: &[String]) -> f64 {
        if sequence.is_empty() {
            return 1.0;
        }
        if sequence.len() == 1 {
            return self.get_marginal_probability(&sequence[0]);
        }
        let mut log2_sum = 0.0;
        for i in 1..sequence.len() {
            let p = self.get_best_context_probability(&sequence[..i], &sequence[i]);
            log2_sum += p.log2();
        }
        log2_sum.exp2().clamp(0.0, 1.0)
    }

    /// Same as [`MarkovModel::calculate_likelihood`] but reuses
    /// pre-interned `StateId`s.
    pub fn calculate_likelihood_ids(&self, sequence_ids: &[StateId], sequence: &[String]) -> f64 {
        if sequence_ids.is_empty() {
            return 1.0;
        }
        if sequence_ids.len() == 1 {
            return self.get_marginal_probability(&sequence[0]);
        }
        let mut log2_sum = 0.0;
        for i in 1..sequence_ids.len() {
            let p = self.get_best_context_probability_ids(
                &sequence_ids[..i],
                sequence_ids[i],
                &sequence[i],
            );
            log2_sum += p.log2();
        }
        log2_sum.exp2().clamp(0.0, 1.0)
    }

    /// Average per-symbol surprise in **bits**:
    /// `(−1/(n−1)) · Σ log₂ P_wb(xᵢ | x₁..xᵢ₋₁)`.
    ///
    /// Stable for arbitrarily long sequences — never multiplies probabilities.
    /// Returns `0.0` for sequences of length < 2.
    pub fn log_likelihood_bits_per_symbol(&self, sequence: &[String]) -> f64 {
        if sequence.len() < 2 {
            return 0.0;
        }
        let mut bits = 0.0;
        for i in 1..sequence.len() {
            let p = self.get_best_context_probability(&sequence[..i], &sequence[i]);
            bits += -p.log2();
        }
        bits / (sequence.len() - 1) as f64
    }

    /// `log_likelihood_bits_per_symbol` using pre-interned ids.
    pub fn log_likelihood_bits_per_symbol_ids(
        &self,
        sequence_ids: &[StateId],
        sequence: &[String],
    ) -> f64 {
        if sequence_ids.len() < 2 {
            return 0.0;
        }
        let mut bits = 0.0;
        for i in 1..sequence_ids.len() {
            let p = self.get_best_context_probability_ids(
                &sequence_ids[..i],
                sequence_ids[i],
                &sequence[i],
            );
            bits += -p.log2();
        }
        bits / (sequence_ids.len() - 1) as f64
    }

    /// Vocabulary size used in probability calculations.
    ///
    /// Returns `frozen_vocab_size` when training is complete (so scoring
    /// is deterministic across rayon thread pools), otherwise falls
    /// back to the live count.
    fn vocab_size_for_scoring(&self) -> usize {
        if self.frozen_vocab_size == 0 {
            self.context_tree.global_vocab_size()
        } else {
            self.frozen_vocab_size
        }
    }

    /// Witten-Bell interpolated probability `P(next | context)`.
    pub fn get_best_context_probability(&self, context: &[String], next_state: &str) -> f64 {
        let max_ctx = context.len().min(self.context_tree.max_order);
        let gv = self.vocab_size_for_scoring();

        // Order-0 base case (Laplace unigram).
        let mut prob = self.get_marginal_probability(next_state);

        for context_len in 1..=max_ctx {
            let sub_context = &context[context.len() - context_len..];
            if let Some(node) = self.context_tree.get_context_node(sub_context) {
                let n = node.total_count() as f64;
                let t = node.vocab_size() as f64;
                if n > 0.0 {
                    let lambda = n / (n + t);
                    let p_ml = node.get_probability(next_state, &self.config, gv);
                    prob = lambda.mul_add(p_ml, (1.0 - lambda) * prob);
                }
            }
        }

        let p = prob.max(self.config.min_probability);
        debug_assert!(
            (0.0..=1.0 + 1e-12).contains(&p),
            "Witten-Bell probability {p} out of [0, 1]"
        );
        p
    }

    /// Witten-Bell interpolated probability using `StateId` (fast path).
    pub fn get_best_context_probability_ids(
        &self,
        context_ids: &[StateId],
        next_state_id: StateId,
        next_state: &str,
    ) -> f64 {
        let max_ctx = context_ids.len().min(self.context_tree.max_order);
        let gv = self.vocab_size_for_scoring();
        let mut prob = self.get_marginal_probability(next_state);

        for context_len in 1..=max_ctx {
            let sub_context = &context_ids[context_ids.len() - context_len..];
            if let Some(node) = self.context_tree.trie().get_context_data(sub_context) {
                let n = node.total_count() as f64;
                let t = node.vocab_size() as f64;
                if n > 0.0 {
                    let lambda = n / (n + t);
                    let p_ml = node.get_probability_by_id(next_state_id, &self.config, gv);
                    prob = lambda.mul_add(p_ml, (1.0 - lambda) * prob);
                }
            }
        }

        let p = prob.max(self.config.min_probability);
        debug_assert!(
            (0.0..=1.0 + 1e-12).contains(&p),
            "Witten-Bell probability {p} out of [0, 1]"
        );
        p
    }

    pub fn max_order(&self) -> usize {
        self.config.max_order
    }

    pub fn config(&self) -> &AnomalyGridConfig {
        &self.config
    }

    pub fn state_mapping(&self) -> &HashMap<String, usize> {
        &self.state_mapping
    }

    pub fn context_tree(&self) -> &ContextTree {
        &self.context_tree
    }

    pub fn context_tree_mut(&mut self) -> &mut ContextTree {
        &mut self.context_tree
    }

    /// Smoothed unigram (order-0 base case).
    pub fn get_marginal_probability(&self, state: &str) -> f64 {
        let gv = self.vocab_size_for_scoring().max(1) as f64;
        if self.total_tokens == 0 {
            return (1.0 / gv).max(self.config.min_probability);
        }
        let raw_count = self.state_counts.get(state).copied().unwrap_or(0) as f64;
        let smoothed = (raw_count + self.config.smoothing_alpha)
            / self.config.smoothing_alpha.mul_add(gv, self.total_tokens as f64);
        smoothed.max(self.config.min_probability)
    }

    /// Background probability for completely unseen symbols.
    pub fn get_background_probability(&self) -> f64 {
        let gv = self.vocab_size_for_scoring().max(1) as f64;
        let bg = self.config.smoothing_alpha
            / self.config.smoothing_alpha.mul_add(gv, self.total_tokens as f64);
        bg.max(self.config.min_probability)
    }

    fn build_state_mapping(&mut self, sequences: &[Vec<String>]) {
        let mut unique_states: HashSet<String> = HashSet::new();

        self.state_counts.clear();
        self.total_tokens = 0;
        self.state_mapping.clear();

        for seq in sequences {
            for token in seq {
                unique_states.insert(token.clone());
                *self.state_counts.entry(token.clone()).or_insert(0) += 1;
                self.total_tokens += 1;
            }
        }

        for (id, state) in unique_states.into_iter().enumerate() {
            self.state_mapping.insert(state, id);
        }
    }

    pub(crate) fn prepare_state_mapping(&mut self, sequences: &[Vec<String>]) {
        self.build_state_mapping(sequences);
    }

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
            .build_from_sequence(sequence, &self.config)?;
        self.frozen_vocab_size = self.context_tree.global_vocab_size();
        Ok(())
    }
}
