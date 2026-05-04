//! Crate-internal string interner.
//!
//! Each unique string is stored **once** as an `Arc<str>` shared
//! between the forward `Vec<Arc<str>>` and reverse `HashMap<Arc<str>,
//! StateId>` maps, so the byte payload is never duplicated. Interior
//! mutability is via `RwLock` so [`crate::AnomalyDetector`] stays
//! `Send + Sync` without any `unsafe`. The interner never panics while
//! holding the lock, so poisoning is recovered in place rather than
//! propagated.

#![allow(clippy::expect_used)] // SAFETY: poison recovery is the idiomatic pattern below.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

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
        // SAFETY: poison recovery — interner state is consistent post-panic.
        self.inner.read().unwrap_or_else(PoisonError::into_inner)
    }

    fn write_inner(&self) -> RwLockWriteGuard<'_, InternerInner> {
        // SAFETY: poison recovery — interner state is consistent post-panic.
        self.inner.write().unwrap_or_else(PoisonError::into_inner)
    }

    /// Intern a string, returning its [`StateId`]. If the string is
    /// already interned the existing id is returned.
    ///
    /// Saturating semantics: if the alphabet would exceed `u32::MAX`,
    /// returns `StateId(u32::MAX)`. The detection path ignores ids it
    /// cannot resolve, so this degrades gracefully rather than panicking.
    pub fn get_or_intern(&self, s: &str) -> StateId {
        {
            let guard = self.read_inner();
            if let Some(id) = guard.lookup.get(s).copied() {
                return id;
            }
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

    /// Resolve an id back to an owned `String`. Returns `None` if the id
    /// was not produced by this interner.
    pub fn get_string(&self, id: StateId) -> Option<String> {
        self.read_inner()
            .storage
            .get(id.index())
            .map(ToString::to_string)
    }

    pub fn len(&self) -> usize {
        self.read_inner().storage.len()
    }

    /// Snapshot of all `(id, string)` pairs in insertion order.
    pub fn entries(&self) -> Vec<(StateId, String)> {
        self.read_inner()
            .storage
            .iter()
            .enumerate()
            .map(|(i, s)| (StateId::new(i as u32), s.to_string()))
            .collect()
    }
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
    fn interner_is_send_and_sync() {
        const fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StringInterner>();
    }
}
