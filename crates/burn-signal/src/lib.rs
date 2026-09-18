#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(feature = "tch", allow(deprecated))]

//! Signal processing operations for Burn tensors.
//!
//! No execution backend is enabled by default. Select `flex`, `wgpu`, or another
//! backend feature on this crate. When configuring through `burn`, enable both
//! `signal` and the backend feature, for example `features = ["signal", "flex"]`.
//! Enabling only `burn/flex` does not enable this crate's FFT backend implementation.
//!
//! # Migration
//!
//! Signal processing previously lived at `burn_tensor::signal` (and therefore
//! `burn_core::tensor::signal`). Use this crate directly or enable Burn's `signal`
//! feature for `burn::signal` and the compatibility path `burn::tensor::signal`.
//! FFTs are backend extensions; windows and STFT/ISTFT compose
//! tensor operations. Enable `autodiff` to differentiate through FFTs.
//!
//! FFT kernels are available for Flex, CubeCL backends, and LibTorch.
//! The backend features enable native FFT kernels through `burn-flex/fft` or
//! `burn-cubecl/fft`; these kernels are excluded from the backends' default builds.
//!
//! # Remote execution and capture
//!
//! FFTs are recorded as custom operations named `signal::rfft` and `signal::irfft`.
//! Remote servers and interpreters replaying captured graphs must enable `router`
//! and install `register_fft_ops`
//! in their custom-operation registry before execution. Pass that registry to
//! `RemoteServerBuilder::custom_ops` or `TensorInterpreter::with_custom_ops`.
//! Enable `remote` on the client and the concrete backend feature on the server.

extern crate alloc;

mod backends;
#[cfg(any(feature = "fusion", feature = "router"))]
mod custom;
mod functions;
mod ops;

#[cfg(feature = "router")]
pub use backends::router::register_fft_ops;
pub use functions::*;
pub use ops::SignalOps;
