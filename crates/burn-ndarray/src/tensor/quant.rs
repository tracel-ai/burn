use crate::{NdArrayTensor, QuantElement};
use burn_tensor::quantization::{
    QParams, QTensorPrimitive, QuantInputType, QuantLevel, QuantMode, QuantScheme,
    QuantizationStrategy, SymmetricQuantization,
};
use burn_tensor::{DType, Shape, TensorMetadata};

/// A quantized tensor for the ndarray backend.
#[derive(Clone, Debug)]
pub struct NdArrayQTensor<Q: QuantElement> {
    /// The quantized tensor.
    pub qtensor: NdArrayTensor<Q>,
    /// The quantization scheme.
    pub scheme: QuantScheme,
    /// The quantization parameters.
    pub qparams: Vec<QParams<f32, Q>>,
}

impl<Q: QuantElement> NdArrayQTensor<Q> {
    /// Returns the quantization strategy, including quantization parameters, for the given tensor.
    pub fn strategy(&self) -> QuantizationStrategy {
        match self.scheme {
            QuantScheme {
                level: QuantLevel::Tensor,
                mode: QuantMode::Symmetric,
                q_type: QuantInputType::QInt8,
                ..
            } => QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(
                self.qparams[0].scale,
            )),
        }
    }
}

impl<Q: QuantElement> QTensorPrimitive for NdArrayQTensor<Q> {
    fn scheme(&self) -> &QuantScheme {
        &self.scheme
    }
}

impl<Q: QuantElement> TensorMetadata for NdArrayQTensor<Q> {
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> Shape {
        self.qtensor.shape()
    }
}
