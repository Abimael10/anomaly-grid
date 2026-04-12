//! Single-ownership string interning.
//!
//! The interner stores each unique string **once** on the heap as an
//! [`Arc<str>`]. Lookup and reverse-lookup share that allocation, so the
//! byte payload is never duplicated. Interior mutability is via
//! [`RwLock`] so that [`AnomalyDetector`] can remain `Send + Sync`
//! without any unsafe code. The interner never panics while holding the
//! lock, so poisoning is recovered from in place instead of propagating.
//!
//! [`AnomalyDetector`]: crate::anomaly_detector::AnomalyDetector

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::error::{AnomalyGridError, AnomalyGridResult};

/// Compact identifier for an interned string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StateId(u32);

impl StateId {
    pub(crate) fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }

    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StateId({})", self.0)
    }
}

#[derive(Debug, Default, Clone)]
struct InternerInner {
    storage: Vec<Arc<str>>,
    lookup: HashMap<Arc<str>, StateId>,
}

/// String interner with amortised O(1) insert and deterministic IDs.
#[derive(Debug, Clone, Default)]
pub struct StringInterner {
    inner: Arc<RwLock<InternerInner>>,
}

impl StringInterner {
    pub fn new() -> Self {
        Self::default()
    }

    fn read_inner(&self) -> RwLockReadGuard<'_, InternerInner> {
        self.inner.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write_inner(&self) -> RwLockWriteGuard<'_, InternerInner> {
        self.inner.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// Intern a string, returning its [`StateId`]. If the string is
    /// already interned the existing id is returned.
    pub fn get_or_intern(&self, s: &str) -> StateId {
        if let Some(id) = self.read_inner().lookup.get(s).copied() {
            return id;
        }

        let mut inner = self.write_inner();
        if let Some(&id) = inner.lookup.get(s) {
            return id;
        }

        let id = StateId::new(u32::try_from(inner.storage.len()).unwrap_or(u32::MAX));
        let key: Arc<str> = Arc::from(s);
        inner.storage.push(Arc::clone(&key));
        inner.lookup.insert(key, id);
        id
    }

    /// Intern a string, erroring if the alphabet would overflow `u32`.
    pub fn try_intern(&self, s: &str) -> AnomalyGridResult<StateId> {
        if let Some(id) = self.read_inner().lookup.get(s).copied() {
            return Ok(id);
        }

        let mut inner = self.write_inner();
        if let Some(&id) = inner.lookup.get(s) {
            return Ok(id);
        }

        let next = u32::try_from(inner.storage.len())
            .map_err(|_| AnomalyGridError::Internal("alphabet exceeded u32::MAX"))?;
        let id = StateId::new(next);
        let key: Arc<str> = Arc::from(s);
        inner.storage.push(Arc::clone(&key));
        inner.lookup.insert(key, id);
        Ok(id)
    }

    /// Resolve an id back to an owned `String`. Returns `None` if the id
    /// was not produced by this interner.
    pub fn get_string(&self, id: StateId) -> Option<String> {
        self.read_inner()
            .storage
            .get(id.index())
            .map(|s| s.to_string())
    }

    /// Resolve an id back to a shared `Arc<str>`.
    pub fn get_arc(&self, id: StateId) -> Option<Arc<str>> {
        self.read_inner().storage.get(id.index()).map(Arc::clone)
    }

    pub fn len(&self) -> usize {
        self.read_inner().storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.read_inner().storage.is_empty()
    }

    /// Snapshot of all (id, string) pairs. Order matches insertion order.
    pub fn iter(&self) -> Vec<(StateId, String)> {
        let inner = self.read_inner();
        inner
            .storage
            .iter()
            .enumerate()
            .map(|(i, s)| (StateId::new(i as u32), s.to_string()))
            .collect()
    }

    /// Bytes held by the interner, including [`Arc`] headers.
    pub fn estimate_memory_usage(&self) -> usize {
        let inner = self.read_inner();
        // Arc<str> header (strong + weak counts + len) ~= 3 usize; payload = bytes.
        let arc_header_bytes = 3 * std::mem::size_of::<usize>();
        let payload: usize = inner.storage.iter().map(|s| s.len() + arc_header_bytes).sum();
        let storage_vec = inner.storage.capacity() * std::mem::size_of::<Arc<str>>();
        let lookup_map = inner.lookup.capacity()
            * (std::mem::size_of::<Arc<str>>() + std::mem::size_of::<StateId>());
        payload + storage_vec + lookup_map
    }
}

/// Helper trait for converting strings and ids through an interner.
pub trait StateIdConversion {
    fn to_state_id(&self, interner: &StringInterner) -> StateId;
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

pub fn strings_to_state_ids(strings: &[String], interner: &StringInterner) -> Vec<StateId> {
    strings.iter().map(|s| interner.get_or_intern(s)).collect()
}

pub fn state_ids_to_strings(ids: &[StateId], interner: &StringInterner) -> Option<Vec<String>> {
    ids.iter().map(|&id| interner.get_string(id)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_is_idempotent() {
        let interner = StringInterner::new();
        let id1 = interner.get_or_intern("hello");
        let id2 = interner.get_or_intern("world");
        let id3 = interner.get_or_intern("hello");
        assert_eq!(id1, id3);
        assert_ne!(id1, id2);
        assert_eq!(interner.len(), 2);
    }

    #[test]
    fn ids_are_deterministic_and_sequential() {
        let interner = StringInterner::new();
        assert_eq!(interner.get_or_intern("a").as_u32(), 0);
        assert_eq!(interner.get_or_intern("b").as_u32(), 1);
        assert_eq!(interner.get_or_intern("c").as_u32(), 2);
        assert_eq!(interner.get_or_intern("a").as_u32(), 0);
    }

    #[test]
    fn reverse_lookup_roundtrips() {
        let interner = StringInterner::new();
        let id = interner.get_or_intern("hello");
        assert_eq!(interner.get_string(id).as_deref(), Some("hello"));
    }

    #[test]
    fn unknown_id_is_none() {
        let interner = StringInterner::new();
        assert!(interner.get_string(StateId::new(999)).is_none());
    }

    #[test]
    fn dedup_stores_payload_once() {
        let interner = StringInterner::new();
        for _ in 0..1000 {
            let _ = interner.get_or_intern("repeated");
        }
        assert_eq!(interner.len(), 1);
    }

    #[test]
    fn conversion_helpers() {
        let interner = StringInterner::new();
        let strings = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let ids = strings_to_state_ids(&strings, &interner);
        let back = state_ids_to_strings(&ids, &interner).expect("roundtrip");
        assert_eq!(strings, back);
    }

    #[test]
    fn interner_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StringInterner>();
    }
}
