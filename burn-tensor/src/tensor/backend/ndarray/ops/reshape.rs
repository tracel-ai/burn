use crate::{
    tensor::{
        backend::ndarray::{NdArrayBackend, NdArrayTensor},
        ops::*,
        NdArrayElement, Shape,
    },
    to_nd_array_tensor,
};
use ndarray::Dim;
use rand::distributions::Standard;

impl<E, const D: usize> TensorOpsReshape<NdArrayBackend<E>, D> for NdArrayTensor<E, D>
where
    E: NdArrayElement,
    Standard: rand::distributions::Distribution<E>,
{
    fn reshape<const D2: usize>(&self, shape: Shape<D2>) -> NdArrayTensor<E, D2> {
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
