use crate::backend::Backend;
use crate::{Dim, NamedDims, NamedTensor, Tensor};

pub trait SwapDims<N, const D1: usize, const D2: usize> {
    fn swap_dims(self) -> N;
}

impl<B: Backend, const D: usize, ND> NamedTensor<B, ND>
where
    ND: NamedDims<B, Tensor = Tensor<B, D>>,
{
    /// Swap two dimensions.
    pub fn swap_dims<ND2, const D1: usize, const D2: usize>(self) -> NamedTensor<B, ND2>
    where
        ND2: NamedDims<B, Tensor = Tensor<B, D>>,
        Self: SwapDims<NamedTensor<B, ND2>, D1, D2>,
    {
        SwapDims::swap_dims(self)
    }
}

macro_rules! generate_permute {
    (2 => $output:ty, ($dim1:expr, $dim2:expr)) => {
        impl<B: Backend, D1: Dim, D2: Dim> SwapDims<NamedTensor<B, $output>, $dim1, $dim2>
            for NamedTensor<B, (D1, D2)>
        {
            fn swap_dims(self) -> NamedTensor<B, $output> {
                NamedTensor::from_tensor(self.tensor.swap_dims($dim1, $dim2))
            }
        }
    };

    (3 => $output:ty, ($dim1:expr, $dim2:expr)) => {
        impl<B: Backend, D1: Dim, D2: Dim, D3: Dim> SwapDims<NamedTensor<B, $output>, $dim1, $dim2>
            for NamedTensor<B, (D1, D2, D3)>
        {
            fn swap_dims(self) -> NamedTensor<B, $output> {
                NamedTensor::from_tensor(self.tensor.swap_dims($dim1, $dim2))
            }
        }
    };

    (4 => $output:ty, ($dim1:expr, $dim2:expr)) => {
        impl<B: Backend, D1: Dim, D2: Dim, D3: Dim, D4: Dim>
            SwapDims<NamedTensor<B, $output>, $dim1, $dim2> for NamedTensor<B, (D1, D2, D3, D4)>
        {
            fn swap_dims(self) -> NamedTensor<B, $output> {
                NamedTensor::from_tensor(self.tensor.swap_dims($dim1, $dim2))
            }
        }
    };
}

generate_permute!(2 => (D2, D1), (0, 1));
generate_permute!(3 => (D2, D1, D3), (0, 1));
generate_permute!(3 => (D3, D2, D1), (0, 2));
generate_permute!(3 => (D1, D3, D2), (1, 2));
generate_permute!(4 => (D2, D1, D3, D4), (0, 1));
generate_permute!(4 => (D3, D2, D1, D4), (0, 2));
generate_permute!(4 => (D4, D2, D3, D1), (0, 3));
generate_permute!(4 => (D1, D3, D2, D4), (1, 2));
generate_permute!(4 => (D1, D4, D3, D2), (1, 3));
