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
//! Observing is the `instrument` feature. Without it there is nowhere to install an observer,
//! [`observing`] is always `None`, and an annotated method compiles to its body: a library can
//! annotate its components and cost a build that never measures nothing. With it and no observer
//! installed, [`observing`] is one thread-local read.
//!
//! # Measuring a model
//!
//! With the `cubecl` and `instrument` features, [`Instrument`] installs one recorder as the
//! component observer, the cubecl runtime's launch observer and the fusion server's observer,
//! runs the model, and returns its [`Breakdown`]: the component tree, each component's kernels,
//! and what the device spent on them. Nothing is drained at a component boundary, so an
//! instrumented run launches the kernels a clean run launches. Each launch is
//! recorded under the component that registered its work, on whichever thread issues it; on a
//! fused device, a kernel that carried out several components' work is recorded on the closest
//! component containing all of them and names the ones it fused.
//!
//! Instrument a model the device has already run once. A first run also autotunes: the candidates
//! the tuner launches are recorded as untagged kernels, and the stall while it measures them as
//! the wait of the component that ran next.
//!
//! A run is one thread's work. On a fused device, the operations every thread registers go
//! through one fusion server, which follows the instrumented thread's components, so another
//! thread's work on the device during the run is recorded under whichever component the
//! instrumented thread had open.
//!
//! ```rust,ignore
//! let measurement = Instrument::new(&device).measure(|| model.forward(input.clone()));
//! measurement.write(std::fs::File::create("model.json")?)?;
//! ```
//!
//! [`Instrument::measure`] warms the model up, measures the device's [`Peaks`] before anything
//! is timed (the probes are saturating kernels), times clean runs for the clean wall, and
//! instruments one more run. [`Instrument::breakdown`] instruments a run alone, for a caller
//! that warms the model up itself. Either prices the model first, in a run it does not time:
//! the [`Bounds`] are evaluated there, where reading values back costs the measurement nothing,
//! and the instrumented run's entries take them by position.
//! Both runs must replay the same components and inputs relevant to their bounds; device
//! seeding does not restore model state, and mismatches are not checked. See
//! [`Instrument::breakdown`] for the replay requirements.
//!
//! A [`Breakdown`] is data, and a [`Measurement`] carries it beside the device's [`Peaks`] and
//! the wall time of a clean run. [`Measurement::write`] saves it as a
//! versioned JSON file ([`MEASUREMENT_FILE_VERSION`]) that [`Measurement::read`] reads back on any
//! machine, with or without a runtime, to be scored there.
//!
//! # Feature flags
//!
//! - `std` (default): reading and writing breakdowns.
//! - `instrument` (default): installing an observer, which arms every annotated method. Off, the
//!   annotations compile away.
//! - `cubecl`: [`Peaks::new`], and with `instrument` [`Instrument`], which reads a cubecl runtime's
//!   kernel launches.
//! - `fusion` (default): the instrument also reads the fusion server's blocks, which it needs to
//!   instrument a device that fuses.
//!
// Where the enabled features leave an item out, its links go to this version's docs.rs page.
#![cfg_attr(not(all(feature = "cubecl", feature = "instrument")), doc = concat!("[`Instrument`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Instrument.html")))]
#![cfg_attr(not(all(feature = "cubecl", feature = "instrument")), doc = concat!("[`Instrument::measure`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Instrument.html#method.measure")))]
#![cfg_attr(not(all(feature = "cubecl", feature = "instrument")), doc = concat!("[`Instrument::breakdown`]: ", concat!("https://docs.rs/burn-instrument/", env!("CARGO_PKG_VERSION"), "/burn_instrument/struct.Instrument.html#method.breakdown")))]
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
#[cfg(all(feature = "cubecl", feature = "instrument"))]
mod instrument;
#[cfg(feature = "std")]
mod peaks;

#[cfg(all(test, feature = "instrument"))]
mod observed;
#[cfg(all(test, feature = "instrument"))]
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
#[cfg(all(feature = "cubecl", feature = "instrument"))]
pub use instrument::Instrument;
#[cfg(feature = "std")]
pub use peaks::{ComputePeak, PeakFinding, Peaks};

pub use burn_instrument_macros::compute;
