use burn_tensor::{
    quantization::{
        AffineQuantization, QTensorPrimitive, QuantizationParametersPrimitive, QuantizationScheme,
        QuantizationStrategy, QuantizationType, SymmetricQuantization,
    },
    read_sync, DType, TensorData, TensorMetadata,
};

use crate::{
    element::BoolElement, ops::into_data, FloatElement, IntElement, JitBackend, JitRuntime,
};

use super::JitTensor;

/// A quantized tensor primitive.
#[derive(Debug)]
pub struct QJitTensor<R: JitRuntime> {
    /// The quantized tensor.
    /// Values are stored as multiple packed quantized values in u32.
    pub qtensor: JitTensor<R>,
    /// The quantization scheme.
    pub scheme: QuantizationScheme,
    /// The quantization parameters.
    pub qparams: JitQuantizationParameters<R>,
}

impl<R: JitRuntime> QTensorPrimitive for QJitTensor<R> {
    fn scheme(&self) -> &QuantizationScheme {
        &self.scheme
    }

    fn strategy(&self) -> QuantizationStrategy {
        match &self.scheme {
            QuantizationScheme::PerTensorAffine(dtype) => match dtype {
                QuantizationType::QInt8 => {
                    let scale = read_sync(into_data::<R, f32>(self.qparams.scale.clone()))
                        .iter()
                        .next()
                        .unwrap();
                    let offset =
                        read_sync(into_data::<R, i8>(self.qparams.offset.clone().unwrap()))
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
                    let scale = read_sync(into_data::<R, f32>(self.qparams.scale.clone()))
                        .iter()
                        .next()
                        .unwrap();
                    QuantizationStrategy::PerTensorSymmetricInt8(SymmetricQuantization::init(scale))
                }
            },
        }
    }
}

impl<R: JitRuntime> Clone for QJitTensor<R> {
    fn clone(&self) -> Self {
        Self {
            qtensor: self.qtensor.clone(),
            scheme: self.scheme,
            qparams: self.qparams.clone(),
        }
    }
}

impl<R: JitRuntime> TensorMetadata for QJitTensor<R> {
    fn dtype(&self) -> DType {
        DType::QFloat(self.scheme)
    }

    fn shape(&self) -> burn_tensor::Shape {
        self.qtensor.shape()
    }
}

/// The quantization parameters.
#[derive(Debug)]
pub struct JitQuantizationParameters<R: JitRuntime> {
    /// The scaling factor.
    pub scale: JitTensor<R>,
    /// The zero-point offset.
    pub offset: Option<JitTensor<R>>,
}

impl<R: JitRuntime> Clone for JitQuantizationParameters<R> {
    fn clone(&self) -> Self {
        Self {
            scale: self.scale.clone(),
            offset: self.offset.clone(),
        }
    }
}

impl<R: JitRuntime, F: FloatElement, I: IntElement, BT: BoolElement>
    From<QuantizationParametersPrimitive<JitBackend<R, F, I, BT>>>
    for JitQuantizationParameters<R>
{
    fn from(value: QuantizationParametersPrimitive<JitBackend<R, F, I, BT>>) -> Self {
        JitQuantizationParameters {
            scale: value.scale,
            offset: value.offset,
        }
    }
}

impl<R: JitRuntime> JitQuantizationParameters<R> {
    pub fn new<F: FloatElement, I: IntElement>(
        scale: F,
        offset: Option<I>,
        device: &R::Device,
    ) -> Self {
        Self {
            scale: crate::ops::from_data::<R, F>(TensorData::new(vec![scale], [1]), device),
            offset: offset
                .map(|o| crate::ops::from_data::<R, I>(TensorData::new(vec![o], [1]), device)),
        }
    }
}
