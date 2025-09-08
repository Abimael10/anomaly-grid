//! Trie-based context storage for memory-efficient prefix sharing
//!
//! This module implements a trie (prefix tree) data structure for storing
//! variable-order Markov chain contexts with significant memory savings
//! through prefix sharing.

use crate::context_tree::ContextNode;
use crate::string_interner::{StateId, StringInterner};
use smallvec::{SmallVec, smallvec};
use std::sync::Arc;

/// Node identifier in the trie
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u32);

impl NodeId {
    /// Create a new NodeId
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    /// Get the raw ID value
    pub fn id(self) -> u32 {
        self.0
    }
}

/// A node in the context trie
#[derive(Debug, Clone)]
pub struct TrieNode {
    /// Children nodes: (StateId, NodeId) pairs
    /// Uses SmallVec for memory efficiency since most nodes have few children
    children: SmallVec<[(StateId, NodeId); 4]>,
    
    /// Context data if this node represents a complete context
    context_data: Option<ContextNode>,
    
    /// Parent node for navigation (None for root)
    parent: Option<NodeId>,
    
    /// The state that led to this node from parent
    state_from_parent: Option<StateId>,
}

impl TrieNode {
    /// Create a new empty trie node
    pub fn new(parent: Option<NodeId>, state_from_parent: Option<StateId>) -> Self {
        Self {
            children: smallvec![],
            context_data: None,
            parent,
            state_from_parent,
        }
    }

    /// Add a child node
    pub fn add_child(&mut self, state: StateId, child_id: NodeId) {
        // Check if child already exists
        for (existing_state, existing_id) in &mut self.children {
            if *existing_state == state {
                *existing_id = child_id;
                return;
            }
        }
        
        // Add new child
        self.children.push((state, child_id));
    }

    /// Get child node ID for a given state
    pub fn get_child(&self, state: StateId) -> Option<NodeId> {
        self.children
            .iter()
            .find(|(s, _)| *s == state)
            .map(|(_, id)| *id)
    }

    /// Get all children
    pub fn children(&self) -> &[(StateId, NodeId)] {
        &self.children
    }

    /// Set context data for this node
    pub fn set_context_data(&mut self, data: ContextNode) {
        self.context_data = Some(data);
    }

    /// Get context data if present
    pub fn context_data(&self) -> Option<&ContextNode> {
        self.context_data.as_ref()
    }

    /// Get mutable context data if present
    pub fn context_data_mut(&mut self) -> Option<&mut ContextNode> {
        self.context_data.as_mut()
    }

    /// Get parent node ID
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    /// Get the state that led to this node from parent
    pub fn state_from_parent(&self) -> Option<StateId> {
        self.state_from_parent
    }

    /// Check if this node has context data
    pub fn has_context_data(&self) -> bool {
        self.context_data.is_some()
    }

    /// Get memory usage estimate for this node
    pub fn memory_usage(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();
        
        // Children storage
        size += self.children.capacity() * std::mem::size_of::<(StateId, NodeId)>();
        
        // Context data if present
        if let Some(ref data) = self.context_data {
            size += std::mem::size_of::<ContextNode>();
            // Add estimated size of transition counts
            size += data.vocab_size() * std::mem::size_of::<(StateId, usize)>();
        }
        
        size
    }
}

/// Trie-based context storage with prefix sharing
#[derive(Debug, Clone)]
pub struct ContextTrie {
    /// All nodes in the trie stored in a vector for cache efficiency
    nodes: Vec<TrieNode>,
    
    /// Root node ID
    root: NodeId,
    
    /// Free node IDs for reuse
    free_nodes: Vec<NodeId>,
    
    /// Maximum context order
    max_order: usize,
    
    /// String interner for state management
    interner: Arc<StringInterner>,
}

impl ContextTrie {
    /// Create a new context trie
    pub fn new(max_order: usize, interner: Arc<StringInterner>) -> Self {
        let mut nodes = Vec::new();
        let root = NodeId::new(0);
        
        // Create root node
        nodes.push(TrieNode::new(None, None));
        
        Self {
            nodes,
            root,
            free_nodes: Vec::new(),
            max_order,
            interner,
        }
    }

    /// Allocate a new node ID
    fn allocate_node_id(&mut self) -> NodeId {
        if let Some(id) = self.free_nodes.pop() {
            id
        } else {
            let id = NodeId::new(self.nodes.len() as u32);
            self.nodes.push(TrieNode::new(None, None));
            id
        }
    }

    /// Get a node by ID
    fn get_node(&self, id: NodeId) -> Option<&TrieNode> {
        self.nodes.get(id.id() as usize)
    }

    /// Get a mutable node by ID
    fn get_node_mut(&mut self, id: NodeId) -> Option<&mut TrieNode> {
        self.nodes.get_mut(id.id() as usize)
    }

    /// Insert a context path and return the node ID for the final context
    pub fn insert_context_path(&mut self, context: &[StateId]) -> NodeId {
        let mut current_id = self.root;
        
        for &state in context {
            let next_id = {
                let current_node = self.get_node(current_id).expect("Invalid node ID");
                current_node.get_child(state)
            };
            
            current_id = if let Some(existing_id) = next_id {
                existing_id
            } else {
                // Create new child node
                let new_id = self.allocate_node_id();
                
                // Set up the new node
                if let Some(new_node) = self.get_node_mut(new_id) {
                    new_node.parent = Some(current_id);
                    new_node.state_from_parent = Some(state);
                }
                
                // Add child to current node
                if let Some(current_node) = self.get_node_mut(current_id) {
                    current_node.add_child(state, new_id);
                }
                
                new_id
            };
        }
        
        current_id
    }

    /// Get the node ID for a context path
    pub fn get_context_node_id(&self, context: &[StateId]) -> Option<NodeId> {
        let mut current_id = self.root;
        
        for &state in context {
            let current_node = self.get_node(current_id)?;
            current_id = current_node.get_child(state)?;
        }
        
        Some(current_id)
    }

    /// Get context data for a given context
    pub fn get_context_data(&self, context: &[StateId]) -> Option<&ContextNode> {
        let node_id = self.get_context_node_id(context)?;
        let node = self.get_node(node_id)?;
        node.context_data()
    }

    /// Get mutable context data for a given context
    pub fn get_context_data_mut(&mut self, context: &[StateId]) -> Option<&mut ContextNode> {
        let node_id = self.get_context_node_id(context)?;
        let node = self.get_node_mut(node_id)?;
        node.context_data_mut()
    }

    /// Set context data for a given context
    pub fn set_context_data(&mut self, context: &[StateId], data: ContextNode) {
        let node_id = self.insert_context_path(context);
        if let Some(node) = self.get_node_mut(node_id) {
            node.set_context_data(data);
        }
    }

    /// Get or create context data for a given context
    pub fn get_or_create_context_data(&mut self, context: &[StateId]) -> &mut ContextNode {
        let node_id = self.insert_context_path(context);
        
        // Check if context data exists
        let needs_creation = {
            let node = self.get_node(node_id).expect("Invalid node ID");
            !node.has_context_data()
        };
        
        if needs_creation {
            let new_data = ContextNode::new(Arc::clone(&self.interner));
            if let Some(node) = self.get_node_mut(node_id) {
                node.set_context_data(new_data);
            }
        }
        
        self.get_node_mut(node_id)
            .expect("Invalid node ID")
            .context_data_mut()
            .expect("Context data should exist")
    }

    /// Iterate over all contexts with data
    pub fn iter_contexts(&self) -> impl Iterator<Item = (Vec<StateId>, &ContextNode)> {
        ContextTrieIterator::new(self)
    }

    /// Get the number of contexts with data
    pub fn context_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| node.has_context_data())
            .count()
    }

    /// Get the total number of nodes in the trie
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get memory usage estimate
    pub fn memory_usage(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        
        // Node storage
        total += self.nodes.capacity() * std::mem::size_of::<TrieNode>();
        
        // Individual node memory usage
        for node in &self.nodes {
            total += node.memory_usage();
        }
        
        // Free nodes vector
        total += self.free_nodes.capacity() * std::mem::size_of::<NodeId>();
        
        total
    }

    /// Get access to the string interner
    pub fn interner(&self) -> &Arc<StringInterner> {
        &self.interner
    }

    /// Get maximum order
    pub fn max_order(&self) -> usize {
        self.max_order
    }
}

/// Iterator over contexts in the trie
pub struct ContextTrieIterator<'a> {
    trie: &'a ContextTrie,
    stack: Vec<(NodeId, Vec<StateId>)>,
}

impl<'a> ContextTrieIterator<'a> {
    fn new(trie: &'a ContextTrie) -> Self {
        let stack = vec![(trie.root, Vec::new())];
        
        Self { trie, stack }
    }
}

impl<'a> Iterator for ContextTrieIterator<'a> {
    type Item = (Vec<StateId>, &'a ContextNode);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node_id, path)) = self.stack.pop() {
            if let Some(node) = self.trie.get_node(node_id) {
                // Add children to stack for further exploration
                for &(state, child_id) in node.children() {
                    let mut child_path = path.clone();
                    child_path.push(state);
                    self.stack.push((child_id, child_path));
                }
                
                // If this node has context data, return it
                if let Some(context_data) = node.context_data() {
                    return Some((path, context_data));
                }
            }
        }
        
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie_basic_operations() {
        let interner = Arc::new(StringInterner::new());
        let mut trie = ContextTrie::new(3, Arc::clone(&interner));
        
        // Create some state IDs
        let state_a = StateId::new(1);
        let state_b = StateId::new(2);
        let _state_c = StateId::new(3);
        
        // Insert a context path
        let context = vec![state_a, state_b];
        let node_id = trie.insert_context_path(&context);
        
        // Verify we can retrieve it
        let retrieved_id = trie.get_context_node_id(&context);
        assert_eq!(Some(node_id), retrieved_id);
        
        // Insert context data
        let context_data = ContextNode::new(Arc::clone(&interner));
        trie.set_context_data(&context, context_data);
        
        // Verify we can retrieve the data
        let retrieved_data = trie.get_context_data(&context);
        assert!(retrieved_data.is_some());
    }

    #[test]
    fn test_trie_prefix_sharing() {
        let interner = Arc::new(StringInterner::new());
        let mut trie = ContextTrie::new(3, Arc::clone(&interner));
        
        let state_a = StateId::new(1);
        let state_b = StateId::new(2);
        let state_c = StateId::new(3);
        
        // Insert contexts that share prefixes
        let context1 = vec![state_a, state_b];
        let context2 = vec![state_a, state_b, state_c];
        let context3 = vec![state_a, state_c];
        
        trie.insert_context_path(&context1);
        trie.insert_context_path(&context2);
        trie.insert_context_path(&context3);
        
        // Should have shared prefix nodes
        let node_count = trie.node_count();
        // Root + A + A->B + A->B->C + A->C = 5 nodes for 3 contexts
        // This demonstrates prefix sharing (A and A->B are shared)
        assert!(node_count <= 6); // Allow some flexibility
        
        // All contexts should be retrievable
        assert!(trie.get_context_node_id(&context1).is_some());
        assert!(trie.get_context_node_id(&context2).is_some());
        assert!(trie.get_context_node_id(&context3).is_some());
    }

    #[test]
    fn test_trie_iteration() {
        let interner = Arc::new(StringInterner::new());
        let mut trie = ContextTrie::new(2, Arc::clone(&interner));
        
        let state_a = StateId::new(1);
        let state_b = StateId::new(2);
        
        // Add some contexts with data
        let context1 = vec![state_a];
        let context2 = vec![state_a, state_b];
        
        let data1 = ContextNode::new(Arc::clone(&interner));
        let data2 = ContextNode::new(Arc::clone(&interner));
        
        trie.set_context_data(&context1, data1);
        trie.set_context_data(&context2, data2);
        
        // Iterate and count
        let contexts: Vec<_> = trie.iter_contexts().collect();
        assert_eq!(contexts.len(), 2);
        
        // Verify context count
        assert_eq!(trie.context_count(), 2);
    }

    #[test]
    fn test_memory_usage_calculation() {
        let interner = Arc::new(StringInterner::new());
        let mut trie = ContextTrie::new(2, Arc::clone(&interner));
        
        let initial_usage = trie.memory_usage();
        assert!(initial_usage > 0);
        
        // Add some data
        let state_a = StateId::new(1);
        let context = vec![state_a];
        let data = ContextNode::new(Arc::clone(&interner));
        trie.set_context_data(&context, data);
        
        let final_usage = trie.memory_usage();
        assert!(final_usage > initial_usage);
    }
}