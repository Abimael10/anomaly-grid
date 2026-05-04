//! Optimized transition count storage for small collections
//!
//! This module provides memory-efficient storage for transition counts,
//! optimizing for the common case where contexts have few transitions.

use crate::string_interner::StateId;
use smallvec::{smallvec, SmallVec};
use std::collections::HashMap;

/// Memory-efficient storage for transition counts
///
/// Uses SmallVec for small collections (≤4 transitions) and HashMap for larger ones.
/// Based on analysis showing 100% of typical contexts have ≤4 transitions.
#[derive(Debug, Clone)]
pub enum TransitionCounts {
    /// Inline storage for small collections (≤4 transitions)
    /// Uses stack allocation to avoid heap overhead
    Small(SmallVec<[(StateId, usize); 4]>),

    /// HashMap storage for large collections (>4 transitions)
    /// Falls back to HashMap when small storage is exceeded
    Large(HashMap<StateId, usize>),
}

impl TransitionCounts {
    /// Create a new empty transition counts collection
    pub fn new() -> Self {
        Self::Small(smallvec![])
    }

    /// Get the count for a specific state
    pub fn get(&self, state_id: StateId) -> usize {
        match self {
            Self::Small(vec) => vec
                .iter()
                .find(|(id, _)| *id == state_id)
                .map_or(0, |(_, count)| *count),
            Self::Large(map) => map.get(&state_id).copied().unwrap_or(0),
        }
    }

    /// Insert or update a count for a state
    pub fn insert(&mut self, state_id: StateId, count: usize) {
        match self {
            Self::Small(vec) => {
                // Try to find existing entry
                if let Some((_, existing_count)) = vec.iter_mut().find(|(id, _)| *id == state_id) {
                    *existing_count = count;
                    return;
                }

                // Check if we need to promote to Large
                if vec.len() >= 4 {
                    // Promote to HashMap
                    let mut map = HashMap::new();
                    for (id, c) in vec.iter() {
                        map.insert(*id, *c);
                    }
                    map.insert(state_id, count);
                    *self = Self::Large(map);
                } else {
                    // Add to small vector
                    vec.push((state_id, count));
                }
            }
            Self::Large(map) => {
                map.insert(state_id, count);
            }
        }
    }

    /// Increment the count for a state, inserting if not present
    pub fn increment(&mut self, state_id: StateId) {
        let current = self.get(state_id);
        self.insert(state_id, current + 1);
    }

    /// Get the number of unique states
    pub fn len(&self) -> usize {
        match self {
            Self::Small(vec) => vec.len(),
            Self::Large(map) => map.len(),
        }
    }

    /// Check if the collection is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over all (state_id, count) pairs
    pub fn iter(&self) -> TransitionCountsIter<'_> {
        match self {
            Self::Small(vec) => TransitionCountsIter::Small(vec.iter()),
            Self::Large(map) => TransitionCountsIter::Large(map.iter()),
        }
    }

    /// Get all state IDs
    pub fn keys(&self) -> impl Iterator<Item = StateId> + '_ {
        self.iter().map(|(state_id, _)| state_id)
    }

    /// Get all counts
    pub fn values(&self) -> impl Iterator<Item = usize> + '_ {
        self.iter().map(|(_, count)| count)
    }

    /// Check if the collection is using small storage
    pub fn is_small(&self) -> bool {
        matches!(self, Self::Small(_))
    }

    /// Get memory usage estimate in bytes
    pub fn memory_usage(&self) -> usize {
        match self {
            Self::Small(vec) => {
                // SmallVec overhead + inline storage
                std::mem::size_of::<SmallVec<[(StateId, usize); 4]>>()
                    + if vec.spilled() {
                        vec.capacity() * std::mem::size_of::<(StateId, usize)>()
                    } else {
                        0 // Inline storage already counted
                    }
            }
            Self::Large(map) => {
                // HashMap overhead + entries
                std::mem::size_of::<HashMap<StateId, usize>>()
                    + map.capacity()
                        * (std::mem::size_of::<StateId>() + std::mem::size_of::<usize>())
            }
        }
    }
}

impl<'a> IntoIterator for &'a TransitionCounts {
    type Item = (StateId, usize);
    type IntoIter = TransitionCountsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Default for TransitionCounts {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over transition counts
pub enum TransitionCountsIter<'a> {
    Small(std::slice::Iter<'a, (StateId, usize)>),
    Large(std::collections::hash_map::Iter<'a, StateId, usize>),
}

impl Iterator for TransitionCountsIter<'_> {
    type Item = (StateId, usize);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Small(iter) => iter.next().map(|(id, count)| (*id, *count)),
            Self::Large(iter) => iter.next().map(|(id, count)| (*id, *count)),
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use crate::string_interner::StateId;

    #[test]
    fn test_small_collection_operations() {
        let mut counts = TransitionCounts::new();
        assert!(counts.is_empty());
        assert!(counts.is_small());

        // Add some states
        counts.increment(StateId::new(1));
        counts.increment(StateId::new(2));
        counts.increment(StateId::new(1)); // Increment existing

        assert_eq!(counts.len(), 2);
        assert_eq!(counts.get(StateId::new(1)), 2);
        assert_eq!(counts.get(StateId::new(2)), 1);
        assert_eq!(counts.get(StateId::new(3)), 0);
        assert!(counts.is_small());
    }

    #[test]
    fn test_promotion_to_large() {
        let mut counts = TransitionCounts::new();

        // Add 4 states (still small)
        for i in 1..=4 {
            counts.increment(StateId::new(i));
        }
        assert!(counts.is_small());
        assert_eq!(counts.len(), 4);

        // Add 5th state (should promote to large)
        counts.increment(StateId::new(5));
        assert!(!counts.is_small());
        assert_eq!(counts.len(), 5);

        // Verify all data is preserved
        for i in 1..=5 {
            assert_eq!(counts.get(StateId::new(i)), 1);
        }
    }

    #[test]
    fn test_large_collection_operations() {
        let mut counts = TransitionCounts::new();

        // Force promotion to large
        for i in 1..=10 {
            counts.increment(StateId::new(i));
        }
        assert!(!counts.is_small());
        assert_eq!(counts.len(), 10);

        // Test operations on large collection
        counts.increment(StateId::new(5)); // Should be 2 now
        assert_eq!(counts.get(StateId::new(5)), 2);
        assert_eq!(counts.get(StateId::new(1)), 1);
    }

    #[test]
    fn test_iteration() {
        let mut counts = TransitionCounts::new();
        counts.increment(StateId::new(1));
        counts.increment(StateId::new(2));
        counts.increment(StateId::new(1));

        let collected: Vec<_> = counts.iter().collect();
        assert_eq!(collected.len(), 2);

        // Check that we have the right states (order may vary)
        let state_1_count = collected
            .iter()
            .find(|(id, _)| *id == StateId::new(1))
            .expect("state 1 missing")
            .1;
        let state_2_count = collected
            .iter()
            .find(|(id, _)| *id == StateId::new(2))
            .expect("state 2 missing")
            .1;

        assert_eq!(state_1_count, 2);
        assert_eq!(state_2_count, 1);
    }

    #[test]
    fn test_memory_usage() {
        let small_counts = TransitionCounts::new();
        let small_usage = small_counts.memory_usage();

        let mut large_counts = TransitionCounts::new();
        for i in 1..=10 {
            large_counts.increment(StateId::new(i));
        }
        let large_usage = large_counts.memory_usage();

        // Small should use less memory for small collections
        assert!(small_usage > 0);
        assert!(large_usage > 0);

        // For this test, we just verify the calculation works
        // The actual comparison depends on the specific sizes
        println!("Small usage: {small_usage} bytes");
        println!("Large usage: {large_usage} bytes");
    }

    #[test]
    fn test_keys_and_values() {
        let mut counts = TransitionCounts::new();
        counts.increment(StateId::new(1));
        counts.increment(StateId::new(2));
        counts.increment(StateId::new(1));

        let keys: Vec<_> = counts.keys().collect();
        let values: Vec<_> = counts.values().collect();

        assert_eq!(keys.len(), 2);
        assert_eq!(values.len(), 2);
        assert!(keys.contains(&StateId::new(1)));
        assert!(keys.contains(&StateId::new(2)));
        assert!(values.contains(&1));
        assert!(values.contains(&2));
    }
}
