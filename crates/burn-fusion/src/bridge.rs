use burn_tensor::backend::BackendBridge;

use crate::{Fusion, FusionBackend};

#[derive(Debug)]
/// Fusion bridge.
pub struct PrecisionBridge;

impl<B: FusionBackend> BackendBridge<Fusion<B>> for PrecisionBridge {
    type Target = Fusion<B>;

    fn into_target<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Fusion<B>, D>,
        _device: Option<burn_tensor::Device<Self::Target>>,
    ) -> burn_tensor::ops::FloatTensor<Self::Target, D> {
        tensor
    }

    fn from_target<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self::Target, D>,
        _device: Option<burn_tensor::Device<Fusion<B>>>,
    ) -> burn_tensor::ops::FloatTensor<Fusion<B>, D> {
        tensor
    }
}
