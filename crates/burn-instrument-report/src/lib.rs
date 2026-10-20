#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Read a pass [`burn_instrument`] collected, and report where its device time went.
//!
//! A [`CollectedPass`](burn_instrument::CollectedPass) is data: a component tree, each
//! component's kernels and their device time, and the device's peaks and clean-pass wall time
//! measured beside it. This crate turns one into the report a reader acts on, on any machine:
//! each component scored against the wall it is bound by ([`roofline`]), in a [`Report`]
//! that renders as a table, and a pass held against a saved one ([`Report::against`]) in a
//! [`Comparison`].

mod comparison;
mod millis;
mod report;
pub mod roofline;
mod served_by;

pub use comparison::Comparison;
pub use report::Report;
