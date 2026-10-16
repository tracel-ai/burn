#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! A backend that records Burn operation graphs instead of executing them.
//!
//! Tensors moved to a [`CaptureDevice`] are recorded rather than computed. Each
//! [`CaptureDevice::capture_scope`] call delimits one capture: the closure runs ordinary tensor
//! code, declares the graph's inputs and outputs through [`CaptureScope::complete`], and the
//! scope returns a [`CapturedGraph`]: the recorded `burn_ir::GraphIr` plus the recorded tensor
//! data, such as model weights. This is for tools that need a model's computation as a graph,
//! such as exporters, rather than its results.
//!
//! Applications enable the `capture` feature of `burn` and use `Device::capture()` and
//! `Device::capture_scope`. Inputs and outputs are declared by [`TensorId`]; with the
//! `extension` feature, `tensor.clone().try_into_primitive::<CaptureBackend>()?.id()` gives a
//! tensor's ID. A runtime input must come from outside the scope, for example a tensor moved
//! onto the capture device; tensors created inside the scope are recorded as constants.

extern crate alloc;

mod capture;

pub use burn_ir::TensorId;
pub use capture::*;
