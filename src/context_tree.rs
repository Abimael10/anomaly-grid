//! Context storage and per-context probability estimation.
//!
//! Each [`ContextNode`] holds the conditional empirical distribution
//! `P̂(· | context)` as integer counts. Probability queries use the
//! **global** alphabet size `|Σ|` so Laplace-smoothed distributions are
//! normalised across the full symbol set, not just symbols seen in this
//! particular context.
//!
//! Memory: contexts share prefixes via [`crate::context_trie::ContextTrie`];
//! transition counts use a [`crate::transition_counts::TransitionCounts`]
//! enum that stays inline (`SmallVec`) for typical alphabets.

use crate::config::AnomalyGridConfig;
use crate::context_trie::ContextTrie;
use crate::error::{AnomalyGridError, AnomalyGridResult};
use crate::string_interner::{StateId, StringInterner};
use crate::transition_counts::TransitionCounts;
use std::collections::HashMap;
use std::sync::Arc;

/// A node in the context tree that stores transition statistics.
#[derive(Debug, Clone)]
pub struct ContextNode {
    counts: TransitionCounts,
    total_count: usize,
    interner: Arc<StringInterner>,
}

impl ContextNode {
    pub(crate) fn new(interner: Arc<StringInterner>) -> Self {
        Self {
            counts: TransitionCounts::new(),
            total_count: 0,
            interner,
        }
    }

    pub(crate) fn add_transition_by_id(&mut self, state_id: StateId) {
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
    /// This is `T(c)` in Witten-Bell smoothing.
    pub fn vocab_size(&self) -> usize {
        self.counts.len()
    }

    pub(crate) fn get_state_counts(&self) -> impl Iterator<Item = (StateId, usize)> + '_ {
        self.counts.iter()
    }

    /// Materialise the empirical distribution as a `String → count` map.
    /// Diagnostic / test use only.
    pub fn counts(&self) -> HashMap<String, usize> {
        self.counts
            .iter()
            .filter_map(|(state_id, count)| self.interner.get_string(state_id).map(|s| (s, count)))
            .collect()
    }

    // ── probability ────────────────────────────────────────────────────

    /// Laplace-smoothed conditional probability over the global alphabet.
    ///
    /// `P(state | context) = (count + α) / (N + α·|Σ|)`
    ///
    /// When no transitions have been observed (`total_count == 0`),
    /// returns `1/|Σ|` (uniform).
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
        let p = (count + config.smoothing_alpha)
            / config.smoothing_alpha.mul_add(v, self.total_count as f64);
        debug_assert!(
            (0.0..=1.0 + 1e-12).contains(&p),
            "ContextNode::get_probability_by_id produced p={p} (out of [0,1])"
        );
        p
    }

    // ── information-theoretic measures ─────────────────────────────────

    /// Shannon entropy `H(X) = −∑ P(x) log₂ P(x)` over observed symbols (bits).
    pub fn compute_entropy(&self, config: &AnomalyGridConfig, global_vocab_size: usize) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }
        self.counts
            .keys()
            .map(|state_id| {
                let p = self.get_probability_by_id(state_id, config, global_vocab_size);
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// `D_KL(P ‖ U)` from the uniform distribution over the global alphabet (bits).
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
                if p > 0.0 {
                    p * (p / uniform_prob).log2()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// All probabilities as a `String → f64` map (diagnostic / test use).
    pub fn get_all_probabilities(
        &self,
        config: &AnomalyGridConfig,
        global_vocab_size: usize,
    ) -> HashMap<String, f64> {
        self.counts
            .keys()
            .filter_map(|state_id| {
                self.interner
                    .get_string(state_id)
                    .map(|s| (s, self.get_probability_by_id(state_id, config, global_vocab_size)))
            })
            .collect()
    }
}

/// Context tree for variable-order Markov chain contexts.
///
/// Trie-based storage shares prefixes between contexts of different
/// orders, so `[A, B]` and `[A, B, C]` reuse the `A → B` path.
#[derive(Debug, Clone)]
pub struct ContextTree {
    trie: ContextTrie,
    pub max_order: usize,
    interner: Arc<StringInterner>,
    pub(crate) last_config: AnomalyGridConfig,
}

impl ContextTree {
    pub fn new(max_order: usize) -> AnomalyGridResult<Self> {
        if max_order == 0 {
            return Err(AnomalyGridError::invalid_max_order(max_order));
        }
        let interner = Arc::new(StringInterner::new());
        let trie = ContextTrie::new(max_order, Arc::clone(&interner));
        Ok(Self {
            trie,
            max_order,
            interner,
            last_config: AnomalyGridConfig::default(),
        })
    }

    /// Build the context tree from a training sequence.
    pub fn build_from_sequence(
        &mut self,
        sequence: &[String],
        config: &AnomalyGridConfig,
    ) -> AnomalyGridResult<()> {
        if sequence.len() < config.min_sequence_length {
            return Err(AnomalyGridError::sequence_too_short(
                config.min_sequence_length,
                sequence.len(),
                "context tree building",
            ));
        }

        // Pre-intern the entire sequence once so the inner loops avoid
        // repeated string lookups.
        let id_seq: Vec<StateId> = sequence
            .iter()
            .map(|s| self.interner.get_or_intern(s))
            .collect();

        for window_size in 1..=self.max_order {
            for window in id_seq.windows(window_size + 1) {
                if let Some(limit) = config.memory_limit {
                    if self.trie.context_count() >= limit {
                        return Err(AnomalyGridError::memory_limit_exceeded(
                            self.trie.context_count(),
                            limit,
                        ));
                    }
                }
                let context_ids = &window[..window_size];
                let next_id = window[window_size];
                let node = self.trie.get_or_create_context_data(context_ids)?;
                node.add_transition_by_id(next_id);
            }
        }

        self.last_config = config.clone();
        Ok(())
    }

    /// Global alphabet size (|Σ|) derived from the shared interner.
    pub fn global_vocab_size(&self) -> usize {
        self.interner.len()
    }

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

    pub fn get_context_node(&self, context: &[String]) -> Option<&ContextNode> {
        let context_state_ids: Vec<StateId> = context
            .iter()
            .map(|s| self.interner.get_or_intern(s))
            .collect();
        self.trie.get_context_data(&context_state_ids)
    }

    pub fn get_context_count(&self, context: &[String]) -> Option<usize> {
        self.get_context_node(context).map(ContextNode::total_count)
    }

    pub fn get_contexts_of_order(&self, order: usize) -> Vec<Vec<String>> {
        self.trie
            .iter_contexts()
            .filter_map(|(state_ids, _)| {
                if state_ids.len() == order {
                    state_ids
                        .iter()
                        .map(|&id| self.interner.get_string(id))
                        .collect::<Option<Vec<_>>>()
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn context_count(&self) -> usize {
        self.trie.context_count()
    }

    pub(crate) fn interner(&self) -> &Arc<StringInterner> {
        &self.interner
    }

    /// Snapshot of the global alphabet observed during training,
    /// in interner-insertion order. Useful for iterating over Σ when
    /// computing per-context probability sums.
    pub fn alphabet(&self) -> Vec<String> {
        self.interner.entries().into_iter().map(|(_, s)| s).collect()
    }

    /// All contexts as a `Vec<String> → ContextNode` map.
    ///
    /// Diagnostic / test use only — clones every node.
    pub fn contexts(&self) -> HashMap<Vec<String>, ContextNode> {
        let mut contexts = HashMap::new();
        for (state_ids, node) in self.trie.iter_contexts() {
            if let Some(strings) = state_ids
                .iter()
                .map(|&id| self.interner.get_string(id))
                .collect::<Option<Vec<_>>>()
            {
                contexts.insert(strings, node.clone());
            }
        }
        contexts
    }

    pub(crate) fn trie(&self) -> &ContextTrie {
        &self.trie
    }

    /// Rebuild the trie keeping only contexts where `keep` returns `true`.
    /// Returns the number of contexts removed. Used by pruning helpers.
    pub(crate) fn rebuild_filtered<F>(&mut self, mut keep: F) -> AnomalyGridResult<usize>
    where
        F: FnMut(&[StateId], &ContextNode) -> bool,
    {
        let original = self.trie.context_count();
        let mut new_trie = ContextTrie::new(self.max_order, Arc::clone(&self.interner));

        for (state_ids, node) in self.trie.iter_contexts() {
            if keep(&state_ids, node) {
                let new_node = new_trie.get_or_create_context_data(&state_ids)?;
                for (state_id, count) in node.get_state_counts() {
                    for _ in 0..count {
                        new_node.add_transition_by_id(state_id);
                    }
                }
            }
        }

        if new_trie.context_count() == 0 {
            Ok(0)
        } else {
            let removed = original.saturating_sub(new_trie.context_count());
            self.trie = new_trie;
            Ok(removed)
        }
    }
}
