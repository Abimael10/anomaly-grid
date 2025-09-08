//! Memory pooling for efficient allocation management
//!
//! This module provides object pooling for frequently allocated types
//! to reduce allocation overhead and memory fragmentation during training.

use crate::context_tree::ContextNode;
use crate::context_trie::{TrieNode, NodeId};
use crate::string_interner::{StateId, StringInterner};
use crate::transition_counts::TransitionCounts;
use smallvec::SmallVec;
use std::sync::Arc;

/// Memory pool for managing object allocations efficiently
#[derive(Debug)]
pub struct MemoryPool {
    /// Pool of reusable ContextNode objects
    context_nodes: Vec<ContextNode>,
    /// Indices of free ContextNode objects
    free_context_nodes: Vec<usize>,
    
    /// Pool of reusable TrieNode objects
    trie_nodes: Vec<TrieNode>,
    /// Indices of free TrieNode objects
    free_trie_nodes: Vec<usize>,
    
    /// Pool of reusable SmallVec objects for transition counts
    small_vecs: Vec<SmallVec<[(StateId, usize); 4]>>,
    /// Indices of free SmallVec objects
    free_small_vecs: Vec<usize>,
    
    /// Pool statistics
    stats: PoolStats,
}

/// Statistics for memory pool usage
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total ContextNode allocations requested
    pub context_node_requests: usize,
    /// ContextNode allocations served from pool
    pub context_node_hits: usize,
    
    /// Total TrieNode allocations requested
    pub trie_node_requests: usize,
    /// TrieNode allocations served from pool
    pub trie_node_hits: usize,
    
    /// Total SmallVec allocations requested
    pub small_vec_requests: usize,
    /// SmallVec allocations served from pool
    pub small_vec_hits: usize,
    
    /// Peak pool sizes
    pub peak_context_nodes: usize,
    pub peak_trie_nodes: usize,
    pub peak_small_vecs: usize,
}

impl MemoryPool {
    /// Create a new memory pool with default capacity
    pub fn new() -> Self {
        Self::with_capacity(64, 256, 128)
    }
    
    /// Create a new memory pool with specified initial capacities
    pub fn with_capacity(
        context_nodes: usize,
        trie_nodes: usize,
        small_vecs: usize,
    ) -> Self {
        Self {
            context_nodes: Vec::with_capacity(context_nodes),
            free_context_nodes: Vec::with_capacity(context_nodes),
            trie_nodes: Vec::with_capacity(trie_nodes),
            free_trie_nodes: Vec::with_capacity(trie_nodes),
            small_vecs: Vec::with_capacity(small_vecs),
            free_small_vecs: Vec::with_capacity(small_vecs),
            stats: PoolStats::default(),
        }
    }
    
    /// Get a ContextNode from the pool or create a new one
    pub fn get_context_node(&mut self, interner: Arc<StringInterner>) -> ContextNode {
        self.stats.context_node_requests += 1;
        
        if let Some(index) = self.free_context_nodes.pop() {
            self.stats.context_node_hits += 1;
            
            // Reset the node for reuse
            let mut node = std::mem::replace(
                &mut self.context_nodes[index],
                ContextNode::default()
            );
            node.reset(interner);
            node
        } else {
            // Create new node
            ContextNode::new(interner)
        }
    }
    
    /// Return a ContextNode to the pool for reuse
    pub fn return_context_node(&mut self, mut node: ContextNode) {
        // Clear the node data but keep the allocation
        node.clear();
        
        // Add to pool if we have space
        if self.context_nodes.len() < self.context_nodes.capacity() {
            let index = self.context_nodes.len();
            self.context_nodes.push(node);
            self.free_context_nodes.push(index);
            
            // Update peak statistics
            if self.context_nodes.len() > self.stats.peak_context_nodes {
                self.stats.peak_context_nodes = self.context_nodes.len();
            }
        }
        // Otherwise, let it drop (pool is full)
    }
    
    /// Get a TrieNode from the pool or create a new one
    pub fn get_trie_node(&mut self, parent: Option<NodeId>, state_from_parent: Option<StateId>) -> TrieNode {
        self.stats.trie_node_requests += 1;
        
        if let Some(index) = self.free_trie_nodes.pop() {
            self.stats.trie_node_hits += 1;
            
            // Reset the node for reuse
            let mut node = std::mem::replace(
                &mut self.trie_nodes[index],
                TrieNode::new(None, None)
            );
            node.reset(parent, state_from_parent);
            node
        } else {
            // Create new node
            TrieNode::new(parent, state_from_parent)
        }
    }
    
    /// Return a TrieNode to the pool for reuse
    pub fn return_trie_node(&mut self, mut node: TrieNode) {
        // Clear the node data but keep the allocation
        node.clear();
        
        // Add to pool if we have space
        if self.trie_nodes.len() < self.trie_nodes.capacity() {
            let index = self.trie_nodes.len();
            self.trie_nodes.push(node);
            self.free_trie_nodes.push(index);
            
            // Update peak statistics
            if self.trie_nodes.len() > self.stats.peak_trie_nodes {
                self.stats.peak_trie_nodes = self.trie_nodes.len();
            }
        }
        // Otherwise, let it drop (pool is full)
    }
    
    /// Get a SmallVec from the pool or create a new one
    pub fn get_small_vec(&mut self) -> SmallVec<[(StateId, usize); 4]> {
        self.stats.small_vec_requests += 1;
        
        if let Some(index) = self.free_small_vecs.pop() {
            self.stats.small_vec_hits += 1;
            
            // Take the SmallVec and clear it
            let mut vec = std::mem::take(&mut self.small_vecs[index]);
            vec.clear();
            vec
        } else {
            // Create new SmallVec
            SmallVec::new()
        }
    }
    
    /// Return a SmallVec to the pool for reuse
    pub fn return_small_vec(&mut self, mut vec: SmallVec<[(StateId, usize); 4]>) {
        // Clear the vector but keep the allocation
        vec.clear();
        
        // Add to pool if we have space
        if self.small_vecs.len() < self.small_vecs.capacity() {
            let index = self.small_vecs.len();
            self.small_vecs.push(vec);
            self.free_small_vecs.push(index);
            
            // Update peak statistics
            if self.small_vecs.len() > self.stats.peak_small_vecs {
                self.stats.peak_small_vecs = self.small_vecs.len();
            }
        }
        // Otherwise, let it drop (pool is full)
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }
    
    /// Reset pool statistics
    pub fn reset_stats(&mut self) {
        self.stats = PoolStats::default();
    }
    
    /// Get current pool sizes
    pub fn pool_sizes(&self) -> (usize, usize, usize) {
        (
            self.context_nodes.len(),
            self.trie_nodes.len(),
            self.small_vecs.len(),
        )
    }
    
    /// Calculate hit rates for each pool type
    pub fn hit_rates(&self) -> (f64, f64, f64) {
        let context_hit_rate = if self.stats.context_node_requests > 0 {
            self.stats.context_node_hits as f64 / self.stats.context_node_requests as f64
        } else {
            0.0
        };
        
        let trie_hit_rate = if self.stats.trie_node_requests > 0 {
            self.stats.trie_node_hits as f64 / self.stats.trie_node_requests as f64
        } else {
            0.0
        };
        
        let small_vec_hit_rate = if self.stats.small_vec_requests > 0 {
            self.stats.small_vec_hits as f64 / self.stats.small_vec_requests as f64
        } else {
            0.0
        };
        
        (context_hit_rate, trie_hit_rate, small_vec_hit_rate)
    }
    
    /// Estimate memory usage of the pool
    pub fn memory_usage(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        
        // ContextNode pool
        total += self.context_nodes.capacity() * std::mem::size_of::<ContextNode>();
        total += self.free_context_nodes.capacity() * std::mem::size_of::<usize>();
        
        // TrieNode pool
        total += self.trie_nodes.capacity() * std::mem::size_of::<TrieNode>();
        total += self.free_trie_nodes.capacity() * std::mem::size_of::<usize>();
        
        // SmallVec pool
        total += self.small_vecs.capacity() * std::mem::size_of::<SmallVec<[(StateId, usize); 4]>>();
        total += self.free_small_vecs.capacity() * std::mem::size_of::<usize>();
        
        total
    }
    
    /// Auto-tune pool sizes based on usage patterns
    pub fn auto_tune(&mut self) {
        // Increase pool sizes if hit rates are low
        let (context_hit_rate, trie_hit_rate, small_vec_hit_rate) = self.hit_rates();
        
        // If hit rate is below 80%, consider increasing pool size
        const MIN_HIT_RATE: f64 = 0.8;
        const GROWTH_FACTOR: f64 = 1.5;
        
        if context_hit_rate < MIN_HIT_RATE && self.stats.context_node_requests > 10 {
            let new_capacity = (self.context_nodes.capacity() as f64 * GROWTH_FACTOR) as usize;
            self.context_nodes.reserve(new_capacity - self.context_nodes.capacity());
            self.free_context_nodes.reserve(new_capacity - self.free_context_nodes.capacity());
        }
        
        if trie_hit_rate < MIN_HIT_RATE && self.stats.trie_node_requests > 10 {
            let new_capacity = (self.trie_nodes.capacity() as f64 * GROWTH_FACTOR) as usize;
            self.trie_nodes.reserve(new_capacity - self.trie_nodes.capacity());
            self.free_trie_nodes.reserve(new_capacity - self.free_trie_nodes.capacity());
        }
        
        if small_vec_hit_rate < MIN_HIT_RATE && self.stats.small_vec_requests > 10 {
            let new_capacity = (self.small_vecs.capacity() as f64 * GROWTH_FACTOR) as usize;
            self.small_vecs.reserve(new_capacity - self.small_vecs.capacity());
            self.free_small_vecs.reserve(new_capacity - self.free_small_vecs.capacity());
        }
    }
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl PoolStats {
    /// Calculate overall hit rate across all pool types
    pub fn overall_hit_rate(&self) -> f64 {
        let total_requests = self.context_node_requests + self.trie_node_requests + self.small_vec_requests;
        let total_hits = self.context_node_hits + self.trie_node_hits + self.small_vec_hits;
        
        if total_requests > 0 {
            total_hits as f64 / total_requests as f64
        } else {
            0.0
        }
    }
    
    /// Get a summary string of pool statistics
    pub fn summary(&self) -> String {
        format!(
            "Pool Stats: Overall hit rate: {:.1}%, Context: {}/{} ({:.1}%), Trie: {}/{} ({:.1}%), SmallVec: {}/{} ({:.1}%)",
            self.overall_hit_rate() * 100.0,
            self.context_node_hits, self.context_node_requests,
            if self.context_node_requests > 0 { self.context_node_hits as f64 / self.context_node_requests as f64 * 100.0 } else { 0.0 },
            self.trie_node_hits, self.trie_node_requests,
            if self.trie_node_requests > 0 { self.trie_node_hits as f64 / self.trie_node_requests as f64 * 100.0 } else { 0.0 },
            self.small_vec_hits, self.small_vec_requests,
            if self.small_vec_requests > 0 { self.small_vec_hits as f64 / self.small_vec_requests as f64 * 100.0 } else { 0.0 }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_interner::StringInterner;
    
    #[test]
    fn test_memory_pool_creation() {
        let pool = MemoryPool::new();
        assert_eq!(pool.pool_sizes(), (0, 0, 0));
        
        let pool = MemoryPool::with_capacity(10, 20, 15);
        assert_eq!(pool.pool_sizes(), (0, 0, 0));
        assert!(pool.context_nodes.capacity() >= 10);
        assert!(pool.trie_nodes.capacity() >= 20);
        assert!(pool.small_vecs.capacity() >= 15);
    }
    
    #[test]
    fn test_context_node_pooling() {
        let mut pool = MemoryPool::new();
        let interner = Arc::new(StringInterner::new());
        
        // Get a node from empty pool (should create new)
        let node1 = pool.get_context_node(Arc::clone(&interner));
        assert_eq!(pool.stats().context_node_requests, 1);
        assert_eq!(pool.stats().context_node_hits, 0);
        
        // Return the node
        pool.return_context_node(node1);
        assert_eq!(pool.pool_sizes().0, 1);
        
        // Get another node (should come from pool)
        let _node2 = pool.get_context_node(Arc::clone(&interner));
        assert_eq!(pool.stats().context_node_requests, 2);
        assert_eq!(pool.stats().context_node_hits, 1);
    }
    
    #[test]
    fn test_trie_node_pooling() {
        let mut pool = MemoryPool::new();
        
        // Get a node from empty pool
        let node1 = pool.get_trie_node(None, None);
        assert_eq!(pool.stats().trie_node_requests, 1);
        assert_eq!(pool.stats().trie_node_hits, 0);
        
        // Return the node
        pool.return_trie_node(node1);
        assert_eq!(pool.pool_sizes().1, 1);
        
        // Get another node (should come from pool)
        let _node2 = pool.get_trie_node(Some(NodeId::new(1)), Some(StateId::new(1)));
        assert_eq!(pool.stats().trie_node_requests, 2);
        assert_eq!(pool.stats().trie_node_hits, 1);
    }
    
    #[test]
    fn test_small_vec_pooling() {
        let mut pool = MemoryPool::new();
        
        // Get a SmallVec from empty pool
        let vec1 = pool.get_small_vec();
        assert_eq!(pool.stats().small_vec_requests, 1);
        assert_eq!(pool.stats().small_vec_hits, 0);
        
        // Return the SmallVec
        pool.return_small_vec(vec1);
        assert_eq!(pool.pool_sizes().2, 1);
        
        // Get another SmallVec (should come from pool)
        let _vec2 = pool.get_small_vec();
        assert_eq!(pool.stats().small_vec_requests, 2);
        assert_eq!(pool.stats().small_vec_hits, 1);
    }
    
    #[test]
    fn test_hit_rates() {
        let mut pool = MemoryPool::new();
        let interner = Arc::new(StringInterner::new());
        
        // Initial hit rates should be 0
        let (context_rate, trie_rate, vec_rate) = pool.hit_rates();
        assert_eq!(context_rate, 0.0);
        assert_eq!(trie_rate, 0.0);
        assert_eq!(vec_rate, 0.0);
        
        // Get and return some objects
        let node = pool.get_context_node(Arc::clone(&interner));
        pool.return_context_node(node);
        let _node = pool.get_context_node(Arc::clone(&interner));
        
        // Should have 50% hit rate for context nodes
        let (context_rate, _, _) = pool.hit_rates();
        assert_eq!(context_rate, 0.5);
    }
    
    #[test]
    fn test_memory_usage_calculation() {
        let pool = MemoryPool::new();
        let usage = pool.memory_usage();
        assert!(usage > 0);
        assert!(usage >= std::mem::size_of::<MemoryPool>());
    }
    
    #[test]
    fn test_auto_tuning() {
        let mut pool = MemoryPool::with_capacity(1, 1, 1);
        let interner = Arc::new(StringInterner::new());
        
        // Generate low hit rate scenario
        for _ in 0..20 {
            let node = pool.get_context_node(Arc::clone(&interner));
            // Don't return nodes to keep hit rate low
            std::mem::drop(node);
        }
        
        let initial_capacity = pool.context_nodes.capacity();
        pool.auto_tune();
        
        // Capacity should have increased due to low hit rate
        assert!(pool.context_nodes.capacity() > initial_capacity);
    }
    
    #[test]
    fn test_pool_stats_summary() {
        let mut pool = MemoryPool::new();
        let interner = Arc::new(StringInterner::new());
        
        // Generate some activity
        let node = pool.get_context_node(Arc::clone(&interner));
        pool.return_context_node(node);
        let _node = pool.get_context_node(Arc::clone(&interner));
        
        let summary = pool.stats().summary();
        assert!(summary.contains("Pool Stats"));
        assert!(summary.contains("hit rate"));
    }
}