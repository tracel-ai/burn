pub(crate) mod stats;

mod api;

pub use api::*;

// Re-exported types
pub use burn_backend::{
    DataError, TensorData, TensorMetadata, TensorPrimitive, Tolerance,
    distribution::*,
    element::*,
    ops::TransactionPrimitive,
    tensor::{Bool, Float, Int, TensorKind},
};
pub use burn_std::{
    DType, FloatDType, IntDType, s,
    tensor::{indexing::*, shape::*, slice::*},
};

/// The activation module.
pub mod activation;

/// The backend module.
pub mod backend {
    pub use burn_backend::backend::*;
}

/// The container module.
pub mod container {
    pub use burn_backend::tensor::TensorContainer;
}

/// The grid module.
pub mod grid;

/// The linalg module.
pub mod linalg;

/// The loss module.
pub mod loss;

/// The neural network module.
pub mod module;

/// Operations on tensors module.
pub mod ops {
    pub use burn_backend::backend::ops::*;
    pub use burn_backend::tensor::{
        BoolTensor, Device, FloatElem, FloatTensor, IntElem, IntTensor, QuantizedTensor,
    };
}

/// Tensor quantization module.
pub mod quantization;

#[cfg(feature = "std")]
pub use report::*;

#[cfg(feature = "std")]
mod report;

#[cfg(feature = "experimental-named-tensor")]
mod named;

#[cfg(feature = "experimental-named-tensor")]
pub use named::*;

pub use ops::Device; // Re-export device so that it's available from `burn_tensor::Device`.
