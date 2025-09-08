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
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// A node in the context tree that stores transition statistics
///
/// Uses optimized storage for small collections and StateId for memory efficiency
/// Implements lazy computation with caching for entropy and KL divergence
#[derive(Debug, Clone)]
pub struct ContextNode {
    /// Optimized transition counts using SmallVec for small collections
    counts: TransitionCounts,
    /// Cached total count to avoid recomputation
    total_count: usize,
    /// String interner for converting between strings and StateIds
    interner: Arc<StringInterner>,
    /// Cached entropy value (computed lazily)
    cached_entropy: Option<f64>,
    /// Cached KL divergence value (computed lazily)
    cached_kl_divergence: Option<f64>,
    /// Configuration hash for cache invalidation
    cached_config_hash: Option<u64>,
}

impl ContextNode {
    /// Create a new empty context node with string interner
    pub fn new(interner: Arc<StringInterner>) -> Self {
        Self {
            counts: TransitionCounts::new(),
            total_count: 0,
            interner,
            cached_entropy: None,
            cached_kl_divergence: None,
            cached_config_hash: None,
        }
    }

    /// Add a transition to this context using string interning
    pub fn add_transition(&mut self, next_state: &str) {
        let state_id = self.interner.get_or_intern(next_state);
        self.counts.increment(state_id);
        self.total_count += 1;
        self.invalidate_cache();
    }

    /// Add a transition using StateId directly (internal use)
    pub fn add_transition_by_id(&mut self, state_id: StateId) {
        self.counts.increment(state_id);
        self.total_count += 1;
        self.invalidate_cache();
    }

    /// Invalidate cached computations when data changes
    fn invalidate_cache(&mut self) {
        self.cached_entropy = None;
        self.cached_kl_divergence = None;
        self.cached_config_hash = None;
    }

    /// Compute a hash of the configuration for cache validation
    fn compute_config_hash(config: &AnomalyGridConfig) -> u64 {
        let mut hasher = DefaultHasher::new();
        // Hash the relevant configuration parameters that affect entropy/KL divergence
        config.smoothing_alpha.to_bits().hash(&mut hasher);
        hasher.finish()
    }

    /// Check if the cached values are valid for the given configuration
    fn is_cache_valid(&self, config: &AnomalyGridConfig) -> bool {
        if let Some(cached_hash) = self.cached_config_hash {
            cached_hash == Self::compute_config_hash(config)
        } else {
            false
        }
    }

    /// Get the total number of transitions from this context
    pub fn total_count(&self) -> usize {
        self.total_count
    }

    /// Get the count for a specific next state
    pub fn get_count(&self, next_state: &str) -> usize {
        let state_id = self.interner.get_or_intern(next_state);
        self.counts.get(state_id)
    }

    /// Get the count for a StateId directly (internal use)
    pub fn get_count_by_id(&self, state_id: StateId) -> usize {
        self.counts.get(state_id)
    }

    /// Get the number of unique next states
    pub fn vocab_size(&self) -> usize {
        self.counts.len()
    }

    /// Get all state IDs with their counts (internal use)
    pub fn get_state_counts(&self) -> impl Iterator<Item = (StateId, usize)> + '_ {
        self.counts.iter()
    }

    /// Get the sum of all transition counts (for compatibility)
    pub fn total_transitions(&self) -> usize {
        self.total_count
    }

    /// Get all counts as strings for compatibility with performance module
    pub fn get_string_counts(&self) -> HashMap<String, usize> {
        self.counts
            .iter()
            .filter_map(|(state_id, count)| self.interner.get_string(state_id).map(|s| (s, count)))
            .collect()
    }

    /// Get counts for debugging (returns string representation)
    pub fn counts(&self) -> HashMap<String, usize> {
        self.get_string_counts()
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

        (count + config.smoothing_alpha)
            / (self.total_count as f64 + config.smoothing_alpha * vocab_size)
    }

    /// Calculate Shannon entropy with lazy computation and caching: H(X) = -∑ P(x) log₂ P(x)
    pub fn calculate_entropy(&mut self, config: &AnomalyGridConfig) -> f64 {
        // Check if we have a valid cached value
        if self.is_cache_valid(config) {
            if let Some(cached_entropy) = self.cached_entropy {
                return cached_entropy;
            }
        }

        // Compute entropy
        let entropy = if self.total_count == 0 {
            0.0
        } else {
            self.counts
                .keys()
                .map(|state_id| {
                    let p = self.get_probability_by_id(state_id, config);
                    if p > 0.0 {
                        -p * p.log2()
                    } else {
                        0.0
                    }
                })
                .sum()
        };

        // Cache the result
        self.cached_entropy = Some(entropy);
        self.cached_config_hash = Some(Self::compute_config_hash(config));
        
        entropy
    }

    /// Calculate Shannon entropy without caching (for immutable access)
    pub fn compute_entropy(&self, config: &AnomalyGridConfig) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        self.counts
            .keys()
            .map(|state_id| {
                let p = self.get_probability_by_id(state_id, config);
                if p > 0.0 {
                    -p * p.log2()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Calculate KL divergence from uniform distribution with lazy computation and caching
    pub fn calculate_kl_divergence(&mut self, config: &AnomalyGridConfig) -> f64 {
        // Check if we have a valid cached value
        if self.is_cache_valid(config) {
            if let Some(cached_kl_div) = self.cached_kl_divergence {
                return cached_kl_div;
            }
        }

        // Compute KL divergence
        let kl_divergence = if self.total_count == 0 {
            0.0
        } else {
            let uniform_prob = 1.0 / self.vocab_size() as f64;
            
            self.counts
                .keys()
                .map(|state_id| {
                    let p = self.get_probability_by_id(state_id, config);
                    if p > 0.0 {
                        p * (p / uniform_prob).log2()
                    } else {
                        0.0
                    }
                })
                .sum()
        };

        // Cache the result
        self.cached_kl_divergence = Some(kl_divergence);
        self.cached_config_hash = Some(Self::compute_config_hash(config));
        
        kl_divergence
    }

    /// Calculate KL divergence from uniform distribution without caching (for immutable access)
    pub fn compute_kl_divergence(&self, config: &AnomalyGridConfig) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        let uniform_prob = 1.0 / self.vocab_size() as f64;

        self.counts
            .keys()
            .map(|state_id| {
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
            .filter_map(|state_id| {
                self.interner.get_string(state_id).map(|state_string| {
                    let prob = self.get_probability_by_id(state_id, config);
                    (state_string, prob)
                })
            })
            .collect()
    }

    /// Reset the context node for reuse in memory pool
    pub fn reset(&mut self, interner: Arc<StringInterner>) {
        self.counts = TransitionCounts::new();
        self.total_count = 0;
        self.interner = interner;
    }

    /// Clear the context node data for memory pool return
    pub fn clear(&mut self) {
        self.counts = TransitionCounts::new();
        self.total_count = 0;
        // Keep the interner for potential reuse
    }
}

impl Default for ContextNode {
    fn default() -> Self {
        Self::new(Arc::new(StringInterner::new()))
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
}

impl ContextTree {
    /// Create a new context tree with specified maximum order
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

        Ok(Self {
            trie,
            max_order,
            interner,
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

        Ok(())
    }

    /// Get the transition probability for a given context and next state
    pub fn get_transition_probability(&self, context: &[String], next_state: &str) -> Option<f64> {
        // Convert context to StateIds
        let context_state_ids: Vec<StateId> = context
            .iter()
            .map(|s| self.interner.get_or_intern(s))
            .collect();
        
        self.trie
            .get_context_data(&context_state_ids)
            .map(|node| node.get_probability(next_state, &AnomalyGridConfig::default()))
    }

    /// Get the transition probability with custom config
    pub fn get_transition_probability_with_config(
        &self,
        context: &[String],
        next_state: &str,
        config: &AnomalyGridConfig,
    ) -> Option<f64> {
        // Convert context to StateIds
        let context_state_ids: Vec<StateId> = context
            .iter()
            .map(|s| self.interner.get_or_intern(s))
            .collect();
        
        self.trie
            .get_context_data(&context_state_ids)
            .map(|node| node.get_probability(next_state, config))
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
}
