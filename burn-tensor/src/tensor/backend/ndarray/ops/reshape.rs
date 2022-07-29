use crate::{
    tensor::{
        backend::ndarray::{NdArrayTensor, NdArrayTensorBackend},
        ops::*,
        Element, Shape, Tensor,
    },
    to_nd_array_tensor,
};
use ndarray::{Dim, LinalgScalar, ScalarOperand};
use rand::distributions::{uniform::SampleUniform, Standard};

impl<P, const D1: usize> TensorOpsReshape<P, D1, NdArrayTensorBackend<P>> for NdArrayTensor<P, D1>
where
    P: Element + ScalarOperand + LinalgScalar + SampleUniform,
    Standard: rand::distributions::Distribution<P>,
{
    fn reshape<const D2: usize>(&self, shape: Shape<D2>) -> Tensor<D2, NdArrayTensorBackend<P>> {
        match D2 {
            1 => to_nd_array_tensor!(1, shape, self.array),
            2 => to_nd_array_tensor!(2, shape, self.array),
            3 => to_nd_array_tensor!(3, shape, self.array),
            4 => to_nd_array_tensor!(4, shape, self.array),
            5 => to_nd_array_tensor!(5, shape, self.array),
            6 => to_nd_array_tensor!(6, shape, self.array),
            _ => panic!("NdArrayTensor support only 6 dimensions."),
        }
    }
}
