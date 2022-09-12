use crate::tensor::{backend::ndarray::NdArrayTensor, ops::*, Data};

impl<P, const D: usize> Zeros<NdArrayTensor<P, D>> for NdArrayTensor<P, D>
where
    P: Default + Clone + Zeros<P> + std::fmt::Debug,
{
    fn zeros(&self) -> NdArrayTensor<P, D> {
        let data = Data::<P, D>::zeros(self.shape);
        Self::from_data(data)
    }
}

impl<P, const D: usize> Ones<NdArrayTensor<P, D>> for NdArrayTensor<P, D>
where
    P: Default + Clone + Ones<P> + std::fmt::Debug,
{
    fn ones(&self) -> NdArrayTensor<P, D> {
        let data = Data::<P, D>::ones(self.shape);
        Self::from_data(data)
    }
}
