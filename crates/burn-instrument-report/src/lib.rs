#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Read a measurement [`burn_instrument`] took, and report where its device time went.
//!
//! A [`Measurement`](burn_instrument::Measurement) is data: a component tree, each
//! component's kernels and their device time, and the device's peaks and clean-run wall time
//! measured beside it. This crate turns one into the report a reader acts on, on any machine:
//! each component scored against the wall it is bound by ([`roofline`]), in a [`Report`]
//! that renders as a table, and a measurement held against a saved one ([`Report::against`]) in a
//! [`Comparison`].

mod comparison;
mod millis;
mod reading;
mod report;
pub mod roofline;
mod served_by;
mod share;

pub use comparison::Comparison;
pub use report::Report;
