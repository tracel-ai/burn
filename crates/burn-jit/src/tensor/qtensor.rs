use burn_tensor::quantization::{QTensorPrimitive, QuantizationScheme, QuantizationStrategy};

use crate::JitRuntime;

use super::JitTensor;

/// A quantized tensor primitive.
#[derive(Debug)]
pub struct QJitTensor<R: JitRuntime, const D: usize> {
    /// The quantized tensor.
    // TODO: implement `JitElement` / `CubeElement` for quantized type
    pub qtensor: JitTensor<R, u32, D>,
    /// The quantization scheme.
    pub scheme: QuantizationScheme,
}

impl<R: JitRuntime, const D: usize> QTensorPrimitive for QJitTensor<R, D> {
    fn scheme(&self) -> &QuantizationScheme {
        &self.scheme
    }

    fn strategy(&self) -> QuantizationStrategy {
        todo!()
    }
}

impl<R: JitRuntime, const D: usize> Clone for QJitTensor<R, D> {
    fn clone(&self) -> Self {
        Self {
            qtensor: self.qtensor.clone(),
            scheme: self.scheme.clone(),
        }
    }
}
