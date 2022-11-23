use crate::backend::Backend;
use crate::{Dim, NamedDims, NamedTensor, Tensor};

pub trait Permut<N, const D1: usize, const D2: usize> {
    fn permut(&self) -> N;
}

impl<B: Backend, const D: usize, ND> NamedTensor<B, ND>
where
    ND: NamedDims<B, Tensor = Tensor<B, D>>,
{
    /// Permut two dimensions.
    pub fn permut<ND2, const D1: usize, const D2: usize>(&self) -> NamedTensor<B, ND2>
    where
        ND2: NamedDims<B, Tensor = Tensor<B, D>>,
        Self: Permut<NamedTensor<B, ND2>, D1, D2>,
    {
        Permut::permut(self)
    }
}

macro_rules! generate_permut {
    (2 => $output:ty, ($dim1:expr, $dim2:expr)) => {
        impl<B: Backend, D1: Dim, D2: Dim> Permut<NamedTensor<B, $output>, $dim1, $dim2>
            for NamedTensor<B, (D1, D2)>
        {
            fn permut(&self) -> NamedTensor<B, $output> {
                NamedTensor::from_tensor(self.tensor.swap_dims($dim1, $dim2))
            }
        }
    };

    (3 => $output:ty, ($dim1:expr, $dim2:expr)) => {
        impl<B: Backend, D1: Dim, D2: Dim, D3: Dim> Permut<NamedTensor<B, $output>, $dim1, $dim2>
            for NamedTensor<B, (D1, D2, D3)>
        {
            fn permut(&self) -> NamedTensor<B, $output> {
                NamedTensor::from_tensor(self.tensor.swap_dims($dim1, $dim2))
            }
        }
    };

    (4 => $output:ty, ($dim1:expr, $dim2:expr)) => {
        impl<B: Backend, D1: Dim, D2: Dim, D3: Dim, D4: Dim>
            Permut<NamedTensor<B, $output>, $dim1, $dim2> for NamedTensor<B, (D1, D2, D3, D4)>
        {
            fn permut(&self) -> NamedTensor<B, $output> {
                NamedTensor::from_tensor(self.tensor.swap_dims($dim1, $dim2))
            }
        }
    };
}

generate_permut!(2 => (D2, D1), (0, 1));
generate_permut!(3 => (D2, D1, D3), (0, 1));
generate_permut!(3 => (D3, D2, D1), (0, 2));
generate_permut!(3 => (D1, D3, D2), (1, 2));
generate_permut!(4 => (D2, D1, D3, D4), (0, 1));
generate_permut!(4 => (D3, D2, D1, D4), (0, 2));
generate_permut!(4 => (D4, D2, D3, D1), (0, 3));
generate_permut!(4 => (D1, D3, D2, D4), (1, 2));
generate_permut!(4 => (D1, D4, D3, D2), (1, 3));
