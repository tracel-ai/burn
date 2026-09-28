#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Burn optimizers.

#[macro_use]
extern crate derive_new;

extern crate alloc;

/// Optimizer module.
mod optim;
pub use optim::*;

/// Gradient clipping module.
pub mod grad_clipping;

/// Learning rate scheduler module.
#[cfg(feature = "std")]
pub mod lr_scheduler;

/// A learning rate computed on the host.
///
/// [Learning rate schedulers](crate::lr_scheduler::LrScheduler) work with host values: a schedule
/// is plain arithmetic on the step count, with no reason to touch the device. It also implements
/// [`LrScheduler`](crate::lr_scheduler::LrScheduler) itself, for a constant learning rate.
///
/// An optimizer step takes a [`LearningRate`], which a host value converts into; see its graph
/// capture section for a schedule in a captured step.
pub type HostLr = f64;
