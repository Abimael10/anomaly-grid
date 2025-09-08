//! String Interning System for Memory Optimization
//!
//! This module provides a string interning system that replaces duplicate string
//! storage with compact integer IDs, significantly reducing memory usage.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Compact identifier for interned strings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StateId(u32);

impl StateId {
    /// Create a new StateId (internal use only)
    pub(crate) fn new(id: u32) -> Self {
        Self(id)
    }
    
    /// Get the raw ID value
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for StateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StateId({})", self.0)
    }
}

/// Thread-safe string interning system
#[derive(Debug, Clone)]
pub struct StringInterner {
    inner: Arc<RwLock<InternerInner>>,
}

#[derive(Debug)]
struct InternerInner {
    /// Storage for interned strings
    strings: Vec<String>,
    /// Mapping from string to ID for deduplication
    string_to_id: HashMap<String, StateId>,
}

impl StringInterner {
    /// Create a new string interner
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(InternerInner {
                strings: Vec::new(),
                string_to_id: HashMap::new(),
            })),
        }
    }
    
    /// Intern a string and return its ID
    /// 
    /// If the string is already interned, returns the existing ID.
    /// Otherwise, creates a new ID and stores the string.
    pub fn get_or_intern(&self, s: &str) -> StateId {
        // Try read-only access first (common case)
        {
            let inner = self.inner.read().unwrap();
            if let Some(&id) = inner.string_to_id.get(s) {
                return id;
            }
        }
        
        // Need write access to intern new string
        let mut inner = self.inner.write().unwrap();
        
        // Double-check in case another thread interned it
        if let Some(&id) = inner.string_to_id.get(s) {
            return id;
        }
        
        // Create new ID and intern the string
        let id = StateId::new(inner.strings.len() as u32);
        inner.strings.push(s.to_string());
        inner.string_to_id.insert(s.to_string(), id);
        
        id
    }
    
    /// Get the string for a given ID
    /// 
    /// Returns None if the ID is invalid.
    pub fn get_string(&self, id: StateId) -> Option<String> {
        let inner = self.inner.read().unwrap();
        inner.strings.get(id.0 as usize).cloned()
    }
    
    /// Get the number of interned strings
    pub fn len(&self) -> usize {
        let inner = self.inner.read().unwrap();
        inner.strings.len()
    }
    
    /// Check if the interner is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get all interned strings with their IDs
    pub fn iter(&self) -> Vec<(StateId, String)> {
        let inner = self.inner.read().unwrap();
        inner.strings
            .iter()
            .enumerate()
            .map(|(i, s)| (StateId::new(i as u32), s.clone()))
            .collect()
    }
    
    /// Estimate memory usage of the interner
    pub fn estimate_memory_usage(&self) -> usize {
        let inner = self.inner.read().unwrap();
        let strings_memory: usize = inner.strings.iter().map(|s| s.capacity()).sum();
        let hashmap_memory = inner.string_to_id.capacity() * 
            (std::mem::size_of::<String>() + std::mem::size_of::<StateId>());
        let vec_memory = inner.strings.capacity() * std::mem::size_of::<String>();
        
        strings_memory + hashmap_memory + vec_memory
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper trait for converting between strings and StateIds
pub trait StateIdConversion {
    /// Convert a string slice to a StateId using the interner
    fn to_state_id(&self, interner: &StringInterner) -> StateId;
    
    /// Convert a StateId back to a string using the interner
    fn from_state_id(id: StateId, interner: &StringInterner) -> Option<String>;
}

impl StateIdConversion for str {
    fn to_state_id(&self, interner: &StringInterner) -> StateId {
        interner.get_or_intern(self)
    }
    
    fn from_state_id(id: StateId, interner: &StringInterner) -> Option<String> {
        interner.get_string(id)
    }
}

impl StateIdConversion for String {
    fn to_state_id(&self, interner: &StringInterner) -> StateId {
        interner.get_or_intern(self)
    }
    
    fn from_state_id(id: StateId, interner: &StringInterner) -> Option<String> {
        interner.get_string(id)
    }
}

/// Convert a vector of strings to StateIds
pub fn strings_to_state_ids(strings: &[String], interner: &StringInterner) -> Vec<StateId> {
    strings.iter().map(|s| interner.get_or_intern(s)).collect()
}

/// Convert a vector of StateIds back to strings
pub fn state_ids_to_strings(ids: &[StateId], interner: &StringInterner) -> Option<Vec<String>> {
    ids.iter()
        .map(|&id| interner.get_string(id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_interning() {
        let interner = StringInterner::new();
        
        let id1 = interner.get_or_intern("hello");
        let id2 = interner.get_or_intern("world");
        let id3 = interner.get_or_intern("hello"); // Should reuse id1
        
        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        
        assert_eq!(interner.get_string(id1), Some("hello".to_string()));
        assert_eq!(interner.get_string(id2), Some("world".to_string()));
        
        assert_eq!(interner.len(), 2);
    }
    
    #[test]
    fn test_thread_safety() {
        use std::thread;
        
        let interner = StringInterner::new();
        let interner_clone = interner.clone();
        
        let handle = thread::spawn(move || {
            interner_clone.get_or_intern("thread_string")
        });
        
        let id1 = interner.get_or_intern("main_string");
        let id2 = handle.join().unwrap();
        
        assert_ne!(id1, id2);
        assert_eq!(interner.len(), 2);
    }
    
    #[test]
    fn test_memory_estimation() {
        let interner = StringInterner::new();
        
        let initial_memory = interner.estimate_memory_usage();
        
        interner.get_or_intern("test_string_1");
        interner.get_or_intern("test_string_2");
        
        let after_memory = interner.estimate_memory_usage();
        
        assert!(after_memory > initial_memory);
    }
    
    #[test]
    fn test_conversion_helpers() {
        let interner = StringInterner::new();
        
        let strings = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let ids = strings_to_state_ids(&strings, &interner);
        let recovered = state_ids_to_strings(&ids, &interner).unwrap();
        
        assert_eq!(strings, recovered);
    }
    
    #[test]
    fn test_state_id_conversion_trait() {
        let interner = StringInterner::new();
        
        let id = "test".to_state_id(&interner);
        let recovered = String::from_state_id(id, &interner).unwrap();
        
        assert_eq!(recovered, "test");
    }
    
    #[test]
    fn test_iterator() {
        let interner = StringInterner::new();
        
        interner.get_or_intern("first");
        interner.get_or_intern("second");
        interner.get_or_intern("third");
        
        let items = interner.iter();
        assert_eq!(items.len(), 3);
        
        // Check that all strings are present
        let strings: Vec<String> = items.into_iter().map(|(_, s)| s).collect();
        assert!(strings.contains(&"first".to_string()));
        assert!(strings.contains(&"second".to_string()));
        assert!(strings.contains(&"third".to_string()));
    }
    
    #[test]
    fn test_invalid_state_id() {
        let interner = StringInterner::new();
        
        let invalid_id = StateId::new(999);
        assert_eq!(interner.get_string(invalid_id), None);
    }
    
    #[test]
    fn test_memory_efficiency() {
        let interner = StringInterner::new();
        
        // Intern the same string multiple times
        let test_string = "repeated_string";
        let mut ids = Vec::new();
        
        for _ in 0..1000 {
            ids.push(interner.get_or_intern(test_string));
        }
        
        // Should only have one unique string stored
        assert_eq!(interner.len(), 1);
        
        // All IDs should be the same
        let first_id = ids[0];
        assert!(ids.iter().all(|&id| id == first_id));
    }
}