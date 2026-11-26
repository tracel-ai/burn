use crate::RecordState;
use burn_core::tensor::{Device, FloatDType, Tensor};

/// Encodes one floating-point optimizer statistic between updates.
///
/// The codec receives the tensor in the optimizer's compute dtype. Its record owns all
/// per-parameter decoding data, such as block scales. Encoded leaves may have other shapes,
/// ranks, and dtypes. Decoding must recover the original logical shape on the same device.
/// Use a [`FloatDType`] directly for ordinary floating-point storage or implement this
/// trait for a custom format. [`crate::AdamMomentStorage`] accepts one codec per moment.
pub trait StatisticCodec: Clone + Send + Sync + 'static {
    /// Checkpoint representation of a statistic of rank `D`.
    type Encoded<const D: usize>: RecordState + Clone;

    /// Encode the newly updated statistic on its device.
    fn encode<const D: usize>(&self, tensor: Tensor<D>) -> Self::Encoded<D>;

    /// Decode the stored statistic directly into the requested compute dtype.
    fn decode<const D: usize>(&self, stored: Self::Encoded<D>, compute: FloatDType) -> Tensor<D>;

    /// Move all leaves to the update device in their encoded representation.
    fn to_device<const D: usize>(stored: Self::Encoded<D>, device: &Device) -> Self::Encoded<D>;
}

impl StatisticCodec for FloatDType {
    type Encoded<const D: usize> = Tensor<D>;

    fn encode<const D: usize>(&self, tensor: Tensor<D>) -> Tensor<D> {
        tensor.cast(*self)
    }

    fn decode<const D: usize>(&self, stored: Tensor<D>, compute: FloatDType) -> Tensor<D> {
        stored.cast(compute)
    }

    fn to_device<const D: usize>(stored: Tensor<D>, device: &Device) -> Tensor<D> {
        stored.to_device(device)
    }
}
