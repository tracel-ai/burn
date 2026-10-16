#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Measure where a model's device time goes, component by component.
//!
//! A component declares itself with [`compute`], an attribute on the method that runs it. Every
//! call then opens a tag carrying a [`Declaration`] (its label, the execution that served it,
//! where the entry is written, and optionally the [`Bounds`] that are its floor), and returning
//! closes it. Nesting follows composition, so a forward that calls its children's forwards is
//! their parent.
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
//! # Collecting a pass
//!
//! With the `cubecl` feature, [`SharedCollector`] installs itself as the collector and as the
//! cubecl runtime's launch observer, runs a pass, and returns its [`Breakdown`]: the component
//! tree, each component's kernels, and what the device spent on them. Nothing is drained at a
//! component boundary, so the collected pass runs the kernels a clean pass runs. Each launch is
//! recorded under the component that registered its work, on whichever thread issues it; on a
//! fused device, a kernel that carried out several components' work is recorded on the closest
//! component containing all of them and names the ones it fused.
//!
//! ```rust,ignore
//! let breakdown = SharedCollector::new(&device).collect(|| {
//!     model.forward(input);
//! });
//! ```
//!
//! # Feature flags
//!
//! - `std` (default): installing a collector.
//! - `cubecl`: the collector, which reads a cubecl runtime's kernel launches and fusion blocks.

extern crate alloc;

mod bounds;
mod collector;
mod label;

#[cfg(feature = "cubecl")]
mod breakdown;
#[cfg(feature = "cubecl")]
mod cursor;
#[cfg(feature = "cubecl")]
mod observer;
#[cfg(feature = "cubecl")]
mod row_path;
#[cfg(feature = "cubecl")]
mod tree;

#[cfg(all(test, feature = "std"))]
mod recording;
#[cfg(all(test, feature = "std"))]
mod tests;

pub use bounds::*;
pub use collector::*;
pub use label::*;

#[cfg(feature = "cubecl")]
pub use breakdown::{Breakdown, Kernel, Node, Unplaced, Variant};
#[cfg(feature = "cubecl")]
pub use row_path::RowPath;
#[cfg(feature = "cubecl")]
pub use tree::SharedCollector;

pub use burn_instrument_macros::compute;
