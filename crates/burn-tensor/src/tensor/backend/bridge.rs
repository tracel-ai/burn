use crate::{ops::FloatTensor, Device};

use super::Backend;

/// Allows tensors to be transferred between backends efficiently.
pub trait BackendBridge<Origin: Backend>: Send + Sync + core::fmt::Debug {
    /// The target backend
    type Target: Backend;

    /// Transfer the tensor to the target backend.
    fn into_target(
        tensor: FloatTensor<Origin>,
        device: Option<Device<Self::Target>>,
    ) -> FloatTensor<Self::Target>;

    /// Transfer the tensor from the target backend.
    fn from_target(
        tensor: FloatTensor<Self::Target>,
        device: Option<Device<Origin>>,
    ) -> FloatTensor<Origin>;
}
