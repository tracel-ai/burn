pub(crate) mod stats;

mod api;

pub use api::*;

// Re-exported types
pub use burn_backend::{
    BoolDType, BoolStore, DType, DataError, FloatDType, IntDType, TensorData, TensorMetadata,
    TensorPrimitive, Tolerance, distribution::*, element::*, indexing::*,
    ops::TransactionPrimitive, shape::*, slice::*,
};

/// The tensor kind module.
pub mod kind;
pub use kind::{Bool, Float, Int, TensorKind};

/// The activation module.
pub mod activation;

/// The backend module.
pub mod backend {
    #[cfg(not(feature = "extension"))]
    pub use burn_backend::backend::ExecutionError;
    #[cfg(feature = "distributed")]
    pub use burn_backend::distributed;

    #[cfg(feature = "extension")]
    pub use burn_backend::backend::*;

    #[cfg(feature = "extension")]
    pub use burn_backend::tensor::{BasicOps, Numeric, Ordered};

    #[cfg(feature = "extension")]
    /// The backend extension module.
    pub mod extension {
        pub use burn_backend_extension::backend_extension;
        pub use burn_dispatch::*;
    }
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

/// The signal processing module.
pub mod signal;

/// Operations on tensors module.
pub mod ops {
    // Re-export explicit types without exposing the backend-level traits
    // TODO: group them in burn-backend module instead, and re-export ::*
    // pub use burn_backend::backend::ops::{
    //     AttentionModuleOptions, ConvOptions, ConvTransposeOptions, DeformConvOptions,
    //     GridSampleOptions, GridSamplePaddingMode, InterpolateMode, InterpolateOptions, PadMode,
    //     PaddedConvOptions, TransactionPrimitive, UnfoldOptions,
    // };
    pub use burn_backend::backend::ops::*;
    pub use burn_backend::tensor::{
        BoolElem, BoolTensor, Device, FloatElem, FloatTensor, IntElem, IntTensor, QuantizedTensor,
    };
}

/// Tensor quantization module.
pub mod quantization;

#[cfg(feature = "std")]
pub use report::*;

#[cfg(feature = "std")]
mod report;
