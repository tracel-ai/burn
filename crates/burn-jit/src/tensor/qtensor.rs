use burn_tensor::{
    quantization::{
        AffineQuantization, QTensorPrimitive, QuantizationParametersPrimitive, QuantizationScheme,
        QuantizationStrategy, QuantizationType, SymmetricQuantization,
    },
    read_sync, TensorData,
};

use crate::{ops::into_data, FloatElement, IntElement, JitBackend, JitRuntime};

use super::JitTensor;

/// A quantized tensor primitive.
#[derive(Debug)]
pub struct QJitTensor<R: JitRuntime, F: FloatElement, I: IntElement> {
    /// The quantized tensor.
    /// Values are stored as multiple packed quantized values in u32.
    pub qtensor: JitTensor<R, u32>,
    /// The quantization scheme.
    pub scheme: QuantizationScheme,
    /// The quantization parameters.
    pub qparams: JitQuantizationParameters<R, F, I>,
}

impl<R: JitRuntime, F: FloatElement, I: IntElement> QTensorPrimitive for QJitTensor<R, F, I> {
    fn scheme(&self) -> &QuantizationScheme {
        &self.scheme
    }

    fn strategy(&self) -> QuantizationStrategy {
        match &self.scheme {
            QuantizationScheme::PerTensorAffine(dtype) => match dtype {
                QuantizationType::QInt8 => {
                    let scale = read_sync(into_data(self.qparams.scale.clone()))
                        .iter()
                        .next()
                        .unwrap();
                    let offset = read_sync(into_data(self.qparams.offset.clone().unwrap()))
                        .iter()
                        .next()
                        .unwrap();
                    QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(
                        scale, offset,
                    ))
                }
            },
            QuantizationScheme::PerTensorSymmetric(dtype) => match dtype {
                QuantizationType::QInt8 => {
                    let scale = read_sync(into_data(self.qparams.scale.clone()))
                        .iter()
                        .next()
                        .unwrap();
                    QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(scale))
                }
            },
        }
    }
}

impl<R: JitRuntime, F: FloatElement, I: IntElement> Clone for QJitTensor<R, F, I> {
    fn clone(&self) -> Self {
        Self {
            qtensor: self.qtensor.clone(),
            scheme: self.scheme.clone(),
            qparams: self.qparams.clone(),
        }
    }
}

/// The quantization parameters.
#[derive(Debug)]
pub struct JitQuantizationParameters<R: JitRuntime, F: FloatElement, I: IntElement> {
    /// The scaling factor.
    pub scale: JitTensor<R, F>,
    /// The zero-point offset.
    pub offset: Option<JitTensor<R, I>>,
}

impl<R: JitRuntime, F: FloatElement, I: IntElement> Clone for JitQuantizationParameters<R, F, I> {
    fn clone(&self) -> Self {
        Self {
            scale: self.scale.clone(),
            offset: self.offset.clone(),
        }
    }
}

impl<R: JitRuntime, F: FloatElement, I: IntElement>
    From<QuantizationParametersPrimitive<JitBackend<R, F, I>>>
    for JitQuantizationParameters<R, F, I>
{
    fn from(value: QuantizationParametersPrimitive<JitBackend<R, F, I>>) -> Self {
        JitQuantizationParameters {
            scale: value.scale,
            offset: value.offset,
        }
    }
}

impl<R: JitRuntime, F: FloatElement, I: IntElement> JitQuantizationParameters<R, F, I> {
    pub fn new(scale: F, offset: Option<I>, device: &R::Device) -> Self {
        Self {
            scale: crate::ops::from_data(TensorData::new(vec![scale], [1]), device),
            offset: offset.map(|o| crate::ops::from_data(TensorData::new(vec![o], [1]), device)),
        }
    }
}
