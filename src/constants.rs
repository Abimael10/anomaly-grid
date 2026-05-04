//! Compile-time bounds shared between modules.
//!
//! Defaults for [`crate::AnomalyGridConfig`] live as inline literals on
//! the `Default` impl in `config.rs` — duplicating them here just rots.

/// Validation bounds used at API entry points.
pub mod validation {
    pub const MIN_THRESHOLD: f64 = 0.0;
    pub const MAX_THRESHOLD: f64 = 1.0;
}
