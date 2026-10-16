#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Measure where a model's device time goes, component by component.
//!
//! A component declares itself at the seam this crate defines: entering it opens a tag carrying
//! a [`Declaration`] (its label, the execution that served it, where the entry is written, and
//! optionally the [`Bounds`] that are its floor), and leaving it closes the tag. Nesting follows
//! composition, so a forward that calls its children's forwards is their parent.
//!
//! A tag is not a device-time window. Fusion registers operations as the program reaches them
//! and launches kernels later, a kernel can carry out several components' work, and a window
//! drawn around a host call holds whatever happened to launch inside it. A tag only says which
//! component registered the work; what a kernel took, and which component it belongs to, is the
//! [`Collector`]'s to work out from the launches its runtime reports.
//!
//! With no collector installed, [`armed`] is one thread-local read and an annotated method calls
//! its body directly, so nothing here is behind a feature. Without `std` there is nowhere to
//! install one, and [`armed`] is always `None`.
//!
//! # Feature flags
//!
//! - `std` (default): installing a collector.

extern crate alloc;

mod bounds;
mod collector;
mod label;

#[cfg(all(test, feature = "std"))]
mod recording;

pub use bounds::*;
pub use collector::*;
pub use label::*;
