use crate::NdArrayTensor;
use burn_tensor::{ops::*, Data};

impl<P, const D: usize> Zeros for NdArrayTensor<P, D>
where
    P: Default + Clone + Zeros + core::fmt::Debug,
{
    fn zeros(&self) -> NdArrayTensor<P, D> {
        let data = Data::<P, D>::zeros(self.shape());
        Self::from_data(data)
    }
}

impl<P, const D: usize> Ones for NdArrayTensor<P, D>
where
    P: Default + Clone + Ones + core::fmt::Debug,
{
    fn ones(&self) -> NdArrayTensor<P, D> {
        let data = Data::<P, D>::ones(self.shape());
        Self::from_data(data)
    }
}
