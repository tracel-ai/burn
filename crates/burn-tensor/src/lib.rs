#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

//! This library provides the core abstractions required to run tensor operations with Burn.
//! `Tensor`s are generic over the backend to allow users to perform operations using different `Backend` implementations.
//! Burn's tensors also support auto-differentiation thanks to the `AutodiffBackend` trait.

#[macro_use]
extern crate derive_new;

extern crate alloc;

mod tensor;

pub(crate) use tensor::check::macros::check;
pub use tensor::*;

// Re-exported types
pub use burn_backend::{AllocationProperty, Bytes, StreamId, bf16, f16, read_sync, try_read_sync};






enum Device {
    Cuda(CudaDevice),
    Autodiff(Box<Self>),
}

enum FloatPrimitive {
    Cuda(CudaDevice),
    Autodiff(Box<Self>),
}

fn manui() {
    let device= Device::Cuda(0);
    let param = Tensor::new(device.autodiff()).require_grad();
    let signal = Tensor::new(device);
    let loss = param*signal;
    let stuff = loss.backward();
}


impl FloatOps {
    fn add(lhs, rhs) {
        match (lhs, rhs) {
            (FloatPrimitive::Cuda(lhs), FloatPrimitive::Cuda(rhs)) => FloatPrimitive::Cuda(CudaBackend::float_add(lhs, rhs)),
            (FloatPrimitive::Autodiff(FloatPrimitive::Cuda(lhs)), FloatPrimitive::Cuda(rhs)) => FloatPrimitive::Autodiff(AutodiffBackend::float_add(lhs, from_inner(rhs))),
            _ => panic!("Not the same device"),
        }
    }
}
