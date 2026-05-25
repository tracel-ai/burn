use crate::{NdArray, execute_with_numeric_dtype, ops::NdArrayMathOps};
use burn_backend::{ElementConversion, TensorMetadata, ops::ActivationOps, tensor::FloatTensor};

impl ActivationOps<Self> for NdArray {
    fn relu(tensor: FloatTensor<Self>) -> FloatTensor<Self> {
        execute_with_numeric_dtype!(tensor, |array| NdArrayMathOps::clamp_min(array, 0.elem()))
    }
}
