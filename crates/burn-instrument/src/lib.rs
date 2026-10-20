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
//! With the `cubecl` feature, [`Instrument`] installs one recorder as the component observer,
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
//! let measurement = Instrument::new(&device).measure(|| model.forward(input.clone()));
//! measurement.write(std::fs::File::create("model.json")?)?;
//! ```
//!
//! [`Instrument::measure`] warms the pass up, measures the device's [`Peaks`] before anything
//! is timed (the probes are saturating kernels), times uncollected runs for the clean wall, and
//! collects one more run. [`Instrument::breakdown`] collects a pass alone, for a caller
//! that warms it up itself.
//!
//! A [`Breakdown`] is data, and a [`Measurement`] carries it beside the device's [`Peaks`] and
//! the wall time of the same pass run uncollected. [`Measurement::write`] saves it as a
//! versioned JSON file ([`MEASUREMENT_FILE_VERSION`]) that [`Measurement::read`] reads back on any
//! machine, with or without a runtime, to be scored there.
//!
//! # Feature flags
//!
//! - `std` (default): installing an observer, and reading and writing breakdowns.
//! - `cubecl`: [`Instrument`], which reads a cubecl runtime's kernel launches, and
//!   [`Peaks::new`].
//! - `fusion` (default): the collector also reads the fusion server's blocks, which it needs to
//!   collect a device that fuses.
//!
// Where the enabled features leave an item out, its links go to this version's docs.rs page.
#![cfg_attr(not(feature = "cubecl"), doc = concat!("[`Instrument`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Instrument.html")))]
#![cfg_attr(not(feature = "cubecl"), doc = concat!("[`Instrument::measure`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Instrument.html#method.measure")))]
#![cfg_attr(not(feature = "cubecl"), doc = concat!("[`Instrument::breakdown`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Instrument.html#method.breakdown")))]
#![cfg_attr(not(feature = "cubecl"), doc = concat!("[`Peaks::new`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Peaks.html#method.new")))]
#![cfg_attr(not(feature = "std"), doc = concat!("[`Breakdown`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Breakdown.html")))]
#![cfg_attr(not(feature = "std"), doc = concat!("[`Measurement`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Measurement.html")))]
#![cfg_attr(not(feature = "std"), doc = concat!("[`Measurement::write`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Measurement.html#method.write")))]
#![cfg_attr(not(feature = "std"), doc = concat!("[`Measurement::read`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Measurement.html#method.read")))]
#![cfg_attr(not(feature = "std"), doc = concat!("[`Peaks`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Peaks.html")))]
#![cfg_attr(not(feature = "std"), doc = concat!("[`MEASUREMENT_FILE_VERSION`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/constant.MEASUREMENT_FILE_VERSION.html")))]

extern crate alloc;

mod bounds;
mod label;
mod observer;

#[cfg(feature = "std")]
mod breakdown;
#[cfg(feature = "std")]
mod component_path;
#[cfg(feature = "cubecl")]
mod instrument;
#[cfg(feature = "std")]
mod peaks;

#[cfg(all(test, feature = "std"))]
mod observed;
#[cfg(all(test, feature = "std"))]
mod tests;

pub use bounds::*;
pub use label::*;
pub use observer::*;

#[cfg(feature = "std")]
pub use breakdown::{
    Breakdown, Component, FusedElsewhere, Kernel, MEASUREMENT_FILE_VERSION, Measurement,
    MeasurementFileError, Unplaced, Variant,
};
#[cfg(feature = "std")]
pub use component_path::ComponentPath;
#[cfg(feature = "cubecl")]
pub use instrument::Instrument;
#[cfg(feature = "std")]
pub use peaks::{ComputePeak, PeakFinding, Peaks};

pub use burn_instrument_macros::compute;
