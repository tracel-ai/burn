pub(crate) mod stats;

mod api;

pub use api::*;

// Re-exported types
pub use burn_std::{
    BoolDType, BoolStore, DType, DataError, FloatDType, IndexingUpdateOp, IntDType, TensorData,
    Tolerance, distribution::*, element::*, indexing::*, s, shape::*, slice::*,
};

/// The tensor kind module.
pub mod kind;
pub use kind::{Bool, Float, Int};

/// The activation module.
pub mod activation;

/// The container module.
pub mod container {
    pub use burn_std::tensor::container::TensorContainer;
}

/// The grid module.
pub mod grid;

/// The linalg module.
pub mod linalg;

/// The loss module.
pub mod loss;

/// The neural network module.
pub mod module;

/// The signal processing module.
pub mod signal;

/// Operations on tensors module.
pub mod ops {
    pub(crate) use crate::bridge::*;
    pub use burn_std::ops::*;
}

/// Tensor quantization module.
pub mod quantization;

#[cfg(feature = "std")]
pub use report::*;

#[cfg(feature = "std")]
mod report;
