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
pub use kind::{Bool, Float, Int, TensorKind};

/// The activation module.
pub mod activation;

#[cfg(feature = "extension")]
/// The backend module — escape hatch for backend extension authors.
///
/// Most users should not need this. It re-exports the backend implementer API
/// (`Backend` trait, ops traits, tensor type aliases) under one path.
pub mod backend {
    pub use burn_backend::backend::*;
    pub use burn_backend::tensor::{
        BoolElem, BoolTensor, Device, FloatElem, FloatTensor, IntElem, IntTensor, QuantizedTensor,
    };

    pub use crate::bridge::{BasicOps, Numeric, Ordered};

    /// The backend extension module.
    pub mod extension {
        pub use burn_backend_extension::backend_extension;
        pub use burn_dispatch::*;
    }
}

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
    pub use burn_std::ops::*;
}

/// Tensor quantization module.
pub mod quantization;

#[cfg(feature = "std")]
pub use report::*;

#[cfg(feature = "std")]
mod report;
