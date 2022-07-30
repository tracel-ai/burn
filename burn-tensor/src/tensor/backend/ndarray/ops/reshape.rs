use crate::{
    tensor::{backend::ndarray::NdArrayTensor, ops::*, Element, Shape},
    to_nd_array_tensor,
};
use ndarray::Dim;

impl<P, const D1: usize, const D2: usize> TensorOpsReshape<P, D1, D2, NdArrayTensor<P, D2>>
    for NdArrayTensor<P, D1>
where
    P: Element,
{
    fn reshape(&self, shape: Shape<D2>) -> NdArrayTensor<P, D2> {
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
