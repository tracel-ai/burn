use burn_tensor::{
    ops::{FloatTensor, QTensorOps, QuantizedTensor},
    Quantization, QuantizationStrategy, TensorData,
};

use crate::{element::NdArrayElement, FloatNdArrayElement, NdArray, NdArrayTensor};

fn into_data<E: NdArrayElement, const D: usize>(tensor: NdArrayTensor<E, D>) -> TensorData {
    let shape = tensor.shape();
    let values = tensor.array.into_iter().collect();
    TensorData::new(values, shape)
}

impl<E: FloatNdArrayElement> QTensorOps<Self> for NdArray<E> {
    fn quantize<const D: usize>(
        tensor: FloatTensor<Self, D>,
        strategy: &QuantizationStrategy,
    ) -> QuantizedTensor<Self, D> {
        let data = into_data(tensor).with_quantization(*strategy);
        NdArrayTensor::<i8, D>::from_data(data)
    }

    fn dequantize<const D: usize>(
        tensor: QuantizedTensor<Self, D>,
        strategy: &QuantizationStrategy,
    ) -> FloatTensor<Self, D> {
        let data = into_data(tensor);
        let values = match strategy {
            QuantizationStrategy::PerTensorAffineInt8(s) => s.dequantize(data.as_slice().unwrap()),
            QuantizationStrategy::PerTensorSymmetricInt8(s) => {
                s.dequantize(data.as_slice().unwrap())
            }
        };
        NdArrayTensor::<E, D>::from_data(TensorData::new(values, data.shape))
    }
}
