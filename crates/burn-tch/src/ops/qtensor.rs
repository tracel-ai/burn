use std::ops::Range;

use burn_tensor::{
    ops::{FloatTensor, IntTensor, QTensorOps, QuantizedTensor},
    quantization::{
        QTensorPrimitive, Quantization, QuantizationParametersPrimitive, QuantizationScheme,
        QuantizationStrategy, QuantizationType,
    },
    DType, Shape, TensorData,
};

use crate::{LibTorch, LibTorchDevice, QuantElement, TchElement, TchQTensor, TchShape, TchTensor};

use super::TchOps;

impl<E: TchElement, Q: QuantElement> QTensorOps<Self> for LibTorch<E, Q> {
    fn q_from_data<const D: usize>(
        data: TensorData,
        device: &LibTorchDevice,
    ) -> QuantizedTensor<Self, D> {
        let shape_tch = TchShape::<D>::from(data.shape.as_slice());
        let device = (*device).into();

        // NOTE: tch-rs doesn't have `from_blob_quantized_*` APIs
        // https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/quantized/Quantizer.cpp#L322
        // So for now we have to load the dequantized values to quantize them back since the dequantization
        // methods take the values provided when quantizing.
        let (tensor, scheme) = match data.dtype {
            DType::QFloat(strategy) => match strategy {
                QuantizationStrategy::PerTensorAffineInt8(q) => {
                    let values = q.dequantize(&data.iter::<i8>().collect::<Vec<_>>());
                    let tensor = tch::Tensor::from_slice(&values).to(device);
                    let tensor = TchOps::<E>::quantize::<D, i8>(
                        TchTensor::new(tensor.reshape(shape_tch.dims)),
                        &strategy,
                    )
                    .tensor;
                    (tensor, strategy.scheme())
                }
                QuantizationStrategy::PerTensorSymmetricInt8(q) => {
                    let values = q.dequantize(&data.iter::<i8>().collect::<Vec<_>>());
                    let tensor = tch::Tensor::from_slice(&values).to(device);
                    let tensor = TchOps::<E>::quantize::<D, i8>(
                        TchTensor::new(tensor.reshape(shape_tch.dims)),
                        &strategy,
                    )
                    .tensor;
                    (tensor, strategy.scheme())
                }
            },
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            ),
        };
        TchQTensor {
            qtensor: TchTensor::new(tensor),
            scheme,
        }
    }

    fn quantize<const D: usize>(
        tensor: FloatTensor<Self, D>,
        scheme: &QuantizationScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self, D> {
        let mut tensor = tensor;
        // Quantize only works on Float Tensor
        if E::dtype() == DType::F16 {
            tensor.tensor = tensor.tensor.to_kind(tch::Kind::Float);
        }

        let qtensor = match scheme {
            QuantizationScheme::PerTensorAffine(dtype) => match dtype {
                QuantizationType::QInt8 => tensor.tensor.quantize_per_tensor_tensor_qparams(
                    &qparams.scale.tensor,
                    &qparams.offset.unwrap().tensor,
                    tch::Kind::QInt8,
                ),
            },
            QuantizationScheme::PerTensorSymmetric(_) => {
                tensor.tensor.quantize_per_tensor_tensor_qparams(
                    &qparams.scale.tensor,
                    &tch::Tensor::zeros_like(&qparams.scale.tensor),
                    tch::Kind::QInt8,
                )
            }
        };

        TchQTensor {
            qtensor: TchTensor::new(qtensor),
            scheme: scheme.clone(),
        }
    }

    fn quantize_dynamic<const D: usize>(
        tensor: FloatTensor<Self, D>,
        scheme: &QuantizationScheme,
    ) -> QuantizedTensor<Self, D> {
        let qtensor = match &scheme {
            QuantizationScheme::PerTensorAffine(dtype) => match dtype {
                // Notes on `reduce_range`:
                // https://github.com/pytorch/pytorch/issues/93140
                // https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html#data-type-selection
                QuantizationType::QInt8 => tensor
                    .tensor
                    .quantize_per_tensor_dynamic(tch::Kind::QInt8, /*reduce_range*/ false),
            },
            QuantizationScheme::PerTensorSymmetric(_) => {
                panic!("LibTorch backend does not support symmetric quantize_dynamic")
            }
        };

        TchQTensor {
            qtensor: TchTensor::new(qtensor),
            scheme: scheme.clone(),
        }
    }

    fn dequantize<const D: usize>(tensor: QuantizedTensor<Self, D>) -> FloatTensor<Self, D> {
        TchTensor::new(tensor.qtensor.tensor.dequantize().to_kind(E::KIND))
    }

    fn q_shape<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> Shape<D> {
        tensor.qtensor.shape()
    }

    fn q_device<const D: usize>(tensor: &QuantizedTensor<Self, D>) -> LibTorchDevice {
        tensor.qtensor.tensor.device().into()
    }

    fn q_to_device<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        device: &burn_tensor::Device<Self>,
    ) -> QuantizedTensor<Self, D> {
        let mut tensor = tensor;
        tensor.qtensor = TchOps::to_device(tensor.qtensor, device);
        tensor
    }

    fn q_reshape<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<Self, D2> {
        TchQTensor {
            qtensor: TchOps::reshape(tensor.qtensor, shape),
            scheme: tensor.scheme,
        }
    }

    async fn q_into_data<const D: usize>(tensor: QuantizedTensor<Self, D>) -> TensorData {
        let shape = Self::q_shape(&tensor);
        let tensor = Self::q_reshape(tensor.clone(), Shape::new([shape.num_elements()]));
        let strategy = tensor.strategy();

        // To get the integer values we have to call `int_repr()`
        let values: Result<Vec<i8>, tch::TchError> = tensor.qtensor.tensor.int_repr().try_into();

        TensorData::quantized(values.unwrap(), shape, strategy)
    }

    fn q_swap_dims<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self, D> {
        // NOTE: with per-channel quantization (future), the channel axis could be impacted by this op
        let mut tensor = tensor;
        tensor.qtensor = TchOps::swap_dims(tensor.qtensor, dim1, dim2);
        tensor
    }

    fn q_permute<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        axes: [usize; D],
    ) -> QuantizedTensor<Self, D> {
        // NOTE: with per-channel quantization (future), the channel axis could be impacted by this op
        let mut tensor = tensor;
        tensor.qtensor = TchOps::permute(tensor.qtensor, axes);
        tensor
    }

    fn q_flip<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        axes: &[usize],
    ) -> QuantizedTensor<Self, D> {
        let mut tensor = tensor;
        tensor.qtensor = TchOps::flip(tensor.qtensor, axes);
        tensor
    }

    fn q_gather<const D: usize>(
        dim: usize,
        tensor: QuantizedTensor<Self, D>,
        indices: IntTensor<Self, D>,
    ) -> QuantizedTensor<Self, D> {
        let mut tensor = tensor;
        tensor.qtensor = TchOps::gather(dim, tensor.qtensor, indices);
        tensor
    }

    fn q_select<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
        indices: IntTensor<Self, 1>,
    ) -> QuantizedTensor<Self, D> {
        let mut tensor = tensor;
        tensor.qtensor = TchOps::index_select_dim(tensor.qtensor, dim, indices);
        tensor
    }

    fn q_slice<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<Self, D1>,
        ranges: [Range<usize>; D2],
    ) -> QuantizedTensor<Self, D1> {
        let mut tensor = tensor;
        tensor.qtensor = TchOps::slice(tensor.qtensor, ranges);
        tensor
    }

    fn q_argmax<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        TchOps::argmax(
            TchTensor::<Q, D>::new(tensor.qtensor.tensor.int_repr()),
            dim,
        )
    }

    fn q_argmin<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
    ) -> IntTensor<Self, D> {
        TchOps::argmin(
            TchTensor::<Q, D>::new(tensor.qtensor.tensor.int_repr()),
            dim,
        )
    }

    fn q_max_dim<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
    ) -> QuantizedTensor<Self, D> {
        TchQTensor {
            qtensor: TchOps::max_dim(tensor.qtensor, dim),
            scheme: tensor.scheme,
        }
    }

    fn q_min_dim<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
    ) -> QuantizedTensor<Self, D> {
        TchQTensor {
            qtensor: TchOps::min_dim(tensor.qtensor, dim),
            scheme: tensor.scheme,
        }
    }

    fn q_narrow<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
        start: usize,
        length: usize,
    ) -> QuantizedTensor<Self, D> {
        TchQTensor {
            qtensor: TchOps::narrow(tensor.qtensor, dim, start, length),
            scheme: tensor.scheme,
        }
    }

    fn q_chunk<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        chunks: usize,
        dim: usize,
    ) -> Vec<QuantizedTensor<Self, D>> {
        TchOps::chunk(tensor.qtensor, chunks, dim)
            .into_iter()
            .map(|x| TchQTensor {
                qtensor: x,
                scheme: tensor.scheme.clone(),
            })
            .collect()
    }

    fn q_expand<const D1: usize, const D2: usize>(
        tensor: QuantizedTensor<Self, D1>,
        shape: Shape<D2>,
    ) -> QuantizedTensor<Self, D2> {
        // NOTE: with per-channel quantization (future), the channel axis could be impacted by this op
        TchQTensor {
            qtensor: TchOps::expand(tensor.qtensor, shape),
            scheme: tensor.scheme,
        }
    }

    fn q_sort<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
        descending: bool,
    ) -> QuantizedTensor<Self, D> {
        TchQTensor {
            qtensor: TchOps::sort(tensor.qtensor, dim, descending),
            scheme: tensor.scheme,
        }
    }

    fn q_sort_with_indices<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
        descending: bool,
    ) -> (QuantizedTensor<Self, D>, IntTensor<Self, D>) {
        let (qtensor, indices) = TchOps::sort_with_indices(tensor.qtensor, dim, descending);
        let tensor = TchQTensor {
            qtensor,
            scheme: tensor.scheme,
        };
        (tensor, indices)
    }

    fn q_argsort<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        dim: usize,
        descending: bool,
    ) -> IntTensor<Self, D> {
        TchOps::argsort(tensor.qtensor, dim, descending)
    }
}
