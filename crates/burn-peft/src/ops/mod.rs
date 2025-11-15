//! Core tensor operations for PEFT methods.
//!
//! This module provides specialized operations needed for parameter-efficient fine-tuning:
//! - Column-wise L2 normalization (critical for DoRA/QDoRA)
//! - Merge operations for combining base weights with adapters
//! - Numerical stability utilities

mod col_norm;
pub use col_norm::*;

mod merge;
pub use merge::*;
