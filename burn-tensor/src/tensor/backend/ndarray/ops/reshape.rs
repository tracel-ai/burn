use crate::{backend::ndarray::NdArrayTensor, Shape, TensorOpsReshape};
use ndarray::{Dim, Dimension};

macro_rules! define_impl {
    (
        $n:expr
    ) => {
        impl<P, const D1: usize> TensorOpsReshape<P, D1, $n, NdArrayTensor<P, $n>>
            for NdArrayTensor<P, D1>
        where
            P: Clone + Default + std::fmt::Debug,
            Dim<[usize; $n]>: Dimension,
        {
            fn reshape(&self, shape: Shape<$n>) -> NdArrayTensor<P, $n> {
                let dim: Dim<[usize; $n]> = shape.clone().into();
                let array = self.array.reshape(dim).into_dyn();

                NdArrayTensor { array, shape }
            }
        }
    };
}

define_impl!(1);
define_impl!(2);
define_impl!(3);
define_impl!(4);
define_impl!(5);
define_impl!(6);
