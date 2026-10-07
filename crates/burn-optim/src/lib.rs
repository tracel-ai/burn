#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! Optimizers and learning rate schedulers for Burn.
//!
//! Applications use these through `burn::optim`, `burn::lr_scheduler` and `burn::grad_clipping`.
//! An optimizer is built from its config and applied to a module with gradients from a backward pass:
//!
//! ```rust,ignore
//! let mut optimizer = AdamConfig::new().init();
//! let grads = GradientsParams::from_grads(loss.backward(), &model);
//! model = optimizer.step(learning_rate, model, grads);
//! ```
//!
//! - Optimizers: SGD, Adam, AdamW, Adagrad, Adafactor, Adan, LAMB, L-BFGS, Lion, Muon and RMSprop,
//!   with momentum, weight decay and gradient accumulation helpers.
//! - [`lr_scheduler`]: constant, step, exponential, linear, cosine, Noam, and sequential or
//!   composed schedules.
//! - [`grad_clipping`]: clipping by value or by norm.
//!
//! # Feature flags
//!
//! - `std` (default): standard library support. Without it the crate is `no_std` with `alloc`.
//! - `tracing`: instrument operations with the `tracing` crate.

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
