//! Trie-based context storage for memory-efficient prefix sharing.
//!
//! `pub(crate)` implementation detail: the public API uses
//! [`crate::context_tree::ContextTree`] which wraps this trie. Direct
//! construction is not exposed to library consumers.

use crate::context_tree::ContextNode;
use crate::error::{AnomalyGridError, AnomalyGridResult};
use crate::string_interner::{StateId, StringInterner};
use smallvec::{smallvec, SmallVec};
use std::sync::Arc;

/// Node identifier in the trie.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u32);

impl NodeId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

/// A node in the context trie.
///
/// `children` is a `SmallVec` because typical alphabets keep most nodes
/// well below four continuations — heap allocation is the exception.
#[derive(Debug, Clone, Default)]
pub struct TrieNode {
    children: SmallVec<[(StateId, NodeId); 4]>,
    context_data: Option<ContextNode>,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: smallvec![],
            context_data: None,
        }
    }

    fn add_child(&mut self, state: StateId, child_id: NodeId) {
        for (existing_state, existing_id) in &mut self.children {
            if *existing_state == state {
                *existing_id = child_id;
                return;
            }
        }
        self.children.push((state, child_id));
    }

    fn get_child(&self, state: StateId) -> Option<NodeId> {
        self.children
            .iter()
            .find(|(s, _)| *s == state)
            .map(|(_, id)| *id)
    }

    fn children(&self) -> &[(StateId, NodeId)] {
        &self.children
    }

    fn set_context_data(&mut self, data: ContextNode) {
        self.context_data = Some(data);
    }

    pub fn context_data(&self) -> Option<&ContextNode> {
        self.context_data.as_ref()
    }

    pub fn context_data_mut(&mut self) -> Option<&mut ContextNode> {
        self.context_data.as_mut()
    }

    fn has_context_data(&self) -> bool {
        self.context_data.is_some()
    }

    fn memory_usage(&self) -> usize {
        let mut size = std::mem::size_of::<Self>();
        size += self.children.capacity() * std::mem::size_of::<(StateId, NodeId)>();
        if let Some(ref data) = self.context_data {
            size += std::mem::size_of::<ContextNode>();
            size += data.vocab_size() * std::mem::size_of::<(StateId, usize)>();
        }
        size
    }
}

/// Trie-based context storage with prefix sharing.
#[derive(Debug, Clone)]
pub struct ContextTrie {
    /// All nodes stored in a `Vec` for cache-friendly iteration. The
    /// arena layout avoids per-node heap allocations and keeps `NodeId`
    /// values stable across insertions (we never compact).
    nodes: Vec<TrieNode>,
    root: NodeId,
    interner: Arc<StringInterner>,
}

impl ContextTrie {
    pub fn new(_max_order: usize, interner: Arc<StringInterner>) -> Self {
        Self {
            nodes: vec![TrieNode::new()],
            root: NodeId::new(0),
            interner,
        }
    }

    fn allocate_node_id(&mut self) -> AnomalyGridResult<NodeId> {
        let id = u32::try_from(self.nodes.len())
            .map_err(|_| AnomalyGridError::Internal("trie node arena exceeded u32::MAX"))?;
        self.nodes.push(TrieNode::new());
        Ok(NodeId::new(id))
    }

    fn get_node(&self, id: NodeId) -> Option<&TrieNode> {
        self.nodes.get(id.index())
    }

    fn get_node_mut(&mut self, id: NodeId) -> Option<&mut TrieNode> {
        self.nodes.get_mut(id.index())
    }

    /// Insert a context path and return the node id for the final context.
    ///
    /// Errors only on a violated arena invariant (e.g. id arithmetic overflow);
    /// this should never happen with sane training data and is encoded as
    /// [`AnomalyGridError::Internal`] rather than panicking.
    pub fn insert_context_path(
        &mut self,
        context: &[StateId],
    ) -> AnomalyGridResult<NodeId> {
        let mut current_id = self.root;

        for &state in context {
            let next_id = self
                .get_node(current_id)
                .ok_or(AnomalyGridError::Internal(
                    "trie cursor pointed to missing node",
                ))?
                .get_child(state);

            current_id = if let Some(existing) = next_id {
                existing
            } else {
                let new_id = self.allocate_node_id()?;
                if let Some(parent) = self.get_node_mut(current_id) {
                    parent.add_child(state, new_id);
                }
                new_id
            };
        }
        Ok(current_id)
    }

    fn get_context_node_id(&self, context: &[StateId]) -> Option<NodeId> {
        let mut current_id = self.root;
        for &state in context {
            current_id = self.get_node(current_id)?.get_child(state)?;
        }
        Some(current_id)
    }

    pub fn get_context_data(&self, context: &[StateId]) -> Option<&ContextNode> {
        self.get_node(self.get_context_node_id(context)?)?.context_data()
    }

    pub fn get_or_create_context_data(
        &mut self,
        context: &[StateId],
    ) -> AnomalyGridResult<&mut ContextNode> {
        let node_id = self.insert_context_path(context)?;

        let needs_creation = !self
            .get_node(node_id)
            .ok_or(AnomalyGridError::Internal(
                "freshly inserted node id missing",
            ))?
            .has_context_data();

        if needs_creation {
            let new_data = ContextNode::new(Arc::clone(&self.interner));
            if let Some(node) = self.get_node_mut(node_id) {
                node.set_context_data(new_data);
            }
        }

        self.get_node_mut(node_id)
            .ok_or(AnomalyGridError::Internal(
                "freshly inserted node id missing",
            ))?
            .context_data_mut()
            .ok_or(AnomalyGridError::Internal(
                "context data unset after creation",
            ))
    }

    pub fn iter_contexts(&self) -> ContextTrieIterator<'_> {
        ContextTrieIterator::new(self)
    }

    pub fn context_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.has_context_data()).count()
    }

    pub fn memory_usage(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();
        total += self.nodes.capacity() * std::mem::size_of::<TrieNode>();
        for node in &self.nodes {
            total += node.memory_usage();
        }
        total
    }
}

/// DFS iterator over contexts that carry data. Allocates each path on
/// the fly — only used for diagnostics, pruning, and rebuild, never on
/// the detection hot path.
pub struct ContextTrieIterator<'a> {
    trie: &'a ContextTrie,
    stack: Vec<(NodeId, Vec<StateId>)>,
}

impl<'a> ContextTrieIterator<'a> {
    fn new(trie: &'a ContextTrie) -> Self {
        Self {
            trie,
            stack: vec![(trie.root, Vec::new())],
        }
    }
}

impl<'a> Iterator for ContextTrieIterator<'a> {
    type Item = (Vec<StateId>, &'a ContextNode);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node_id, path)) = self.stack.pop() {
            if let Some(node) = self.trie.get_node(node_id) {
                for &(state, child_id) in node.children() {
                    let mut child_path = path.clone();
                    child_path.push(state);
                    self.stack.push((child_id, child_path));
                }
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
    fn trie_basic_operations() -> AnomalyGridResult<()> {
        let interner = Arc::new(StringInterner::new());
        let mut trie = ContextTrie::new(3, Arc::clone(&interner));

        let state_a = interner.get_or_intern("A");
        let state_b = interner.get_or_intern("B");

        let context = vec![state_a, state_b];
        let node_id = trie.insert_context_path(&context)?;

        assert_eq!(Some(node_id), trie.get_context_node_id(&context));

        let _ = trie.get_or_create_context_data(&context)?;
        assert!(trie.get_context_data(&context).is_some());
        Ok(())
    }

    #[test]
    fn trie_prefix_sharing() -> AnomalyGridResult<()> {
        let interner = Arc::new(StringInterner::new());
        let mut trie = ContextTrie::new(3, Arc::clone(&interner));

        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");
        let c = interner.get_or_intern("C");

        // Three contexts share prefixes: [A, B], [A, B, C], [A, C].
        // Expected node count: root + A + AB + ABC + AC = 5.
        trie.insert_context_path(&[a, b])?;
        trie.insert_context_path(&[a, b, c])?;
        trie.insert_context_path(&[a, c])?;

        assert!(trie.nodes.len() <= 6);
        assert!(trie.get_context_node_id(&[a, b]).is_some());
        assert!(trie.get_context_node_id(&[a, b, c]).is_some());
        assert!(trie.get_context_node_id(&[a, c]).is_some());
        Ok(())
    }

    #[test]
    fn trie_iteration() -> AnomalyGridResult<()> {
        let interner = Arc::new(StringInterner::new());
        let mut trie = ContextTrie::new(2, Arc::clone(&interner));

        let a = interner.get_or_intern("A");
        let b = interner.get_or_intern("B");

        let _ = trie.get_or_create_context_data(&[a])?;
        let _ = trie.get_or_create_context_data(&[a, b])?;

        assert_eq!(trie.iter_contexts().count(), 2);
        assert_eq!(trie.context_count(), 2);
        Ok(())
    }

    #[test]
    fn memory_usage_grows_with_inserts() -> AnomalyGridResult<()> {
        let interner = Arc::new(StringInterner::new());
        let mut trie = ContextTrie::new(2, Arc::clone(&interner));

        let initial = trie.memory_usage();
        assert!(initial > 0);

        let a = interner.get_or_intern("A");
        let _ = trie.get_or_create_context_data(&[a])?;

        assert!(trie.memory_usage() > initial);
        Ok(())
    }
}
