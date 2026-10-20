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
//! component registered the work; what a kernel took, and which component it belongs to, is for
//! the [`ComponentObserver`] to work out from the launches its runtime reports.
//!
//! With no observer installed, [`observing`] is one thread-local read and an annotated method calls
//! its body directly, so nothing here is behind a feature. Without `std` there is nowhere to
//! install one, and [`observing`] is always `None`.
//!
//! # Collecting a pass
//!
//! With the `cubecl` feature, [`PassCollector`] installs one recorder as the component observer,
//! the cubecl runtime's launch observer and the fusion server's observer, runs a pass, and returns its [`Breakdown`]: the component
//! tree, each component's kernels, and what the device spent on them. Nothing is drained at a
//! component boundary, so the collected pass runs the kernels a clean pass runs. Each launch is
//! recorded under the component that registered its work, on whichever thread issues it; on a
//! fused device, a kernel that carried out several components' work is recorded on the closest
//! component containing all of them and names the ones it fused.
//!
//! Collect a pass the device has already run once. A first pass also autotunes: the candidates
//! the tuner launches are recorded as untagged kernels, and the stall while it measures them as
//! the wait of the component that ran next.
//!
//! A pass is one thread's work. On a fused device, the operations every thread registers go
//! through one fusion server, which follows the collecting thread's components, so another
//! thread's work on the device during the pass is recorded under whichever component the
//! collecting thread had open.
//!
//! ```rust,ignore
//! let breakdown = PassCollector::new(&device).collect(|| {
//!     model.forward(input);
//! });
//! breakdown.write(std::fs::File::create("forward.json")?)?;
//! ```
//!
//! A [`Breakdown`] is data, and [`Breakdown::write`] saves it as a versioned JSON file
//! ([`BREAKDOWN_FILE_VERSION`]) that [`Breakdown::read`] reads back on any machine, with or
//! without a runtime.
//!
//! # Feature flags
//!
//! - `std` (default): installing an observer, and reading and writing breakdowns.
//! - `cubecl`: [`PassCollector`], which reads a cubecl runtime's kernel launches.
//! - `fusion` (default): the collector also reads the fusion server's blocks, which it needs to
//!   collect a device that fuses.
//!
// Where the enabled features leave an item out, its links go to this version's docs.rs page.
#![cfg_attr(not(feature = "cubecl"), doc = concat!("[`PassCollector`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.PassCollector.html")))]
#![cfg_attr(not(feature = "std"), doc = concat!("[`Breakdown`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Breakdown.html")))]
#![cfg_attr(not(feature = "std"), doc = concat!("[`Breakdown::write`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Breakdown.html#method.write")))]
#![cfg_attr(not(feature = "std"), doc = concat!("[`Breakdown::read`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Breakdown.html#method.read")))]
#![cfg_attr(not(feature = "std"), doc = concat!("[`BREAKDOWN_FILE_VERSION`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/constant.BREAKDOWN_FILE_VERSION.html")))]

extern crate alloc;

mod bounds;
mod label;
mod observer;

#[cfg(feature = "std")]
mod breakdown;
#[cfg(feature = "std")]
mod component_path;
#[cfg(feature = "cubecl")]
mod pass;

#[cfg(all(test, feature = "std"))]
mod observed;
#[cfg(all(test, feature = "std"))]
mod tests;

pub use bounds::*;
pub use label::*;
pub use observer::*;

#[cfg(feature = "std")]
pub use breakdown::{
    BREAKDOWN_FILE_VERSION, Breakdown, BreakdownFileError, Component, Kernel, Unplaced, Variant,
};
#[cfg(feature = "std")]
pub use component_path::ComponentPath;
#[cfg(feature = "cubecl")]
pub use pass::PassCollector;

pub use burn_instrument_macros::compute;
