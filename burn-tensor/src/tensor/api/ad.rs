use super::Tensor;
use crate::graph::grad::Gradients;
use crate::tensor::backend::autodiff::ADTensor;
use crate::tensor::backend::ADBackend;
use rand::distributions::Standard;

impl<const D: usize, B: ADBackend> Tensor<D, B> {
    pub fn backward(&self) -> Gradients {
        B::backward::<D>(&self.value)
    }

    pub fn grad(&self, grads: &Gradients) -> Option<Tensor<D, B::InnerBackend>> {
        B::grad(&self.value, grads).map(|value| Tensor::new(value))
    }

    pub fn inner(&self) -> Tensor<D, B::InnerBackend> {
        Tensor::new(B::inner(&self.value))
    }

    pub fn update(&mut self, other_inner: Tensor<D, B::InnerBackend>) {
        self.value = B::from_inner(other_inner.value);
    }
}

#[cfg(feature = "ndarray")]
mod ndarray {
    use super::*;
    use crate::tensor::backend::autodiff::ADBackendNdArray;
    use crate::tensor::backend::ndarray::NdArrayBackend;

    impl<E: crate::NdArrayElement, const D: usize> Tensor<D, NdArrayBackend<E>>
    where
        Standard: rand::distributions::Distribution<E>,
    {
        pub fn with_grad(self) -> Tensor<D, ADBackendNdArray<E>> {
            let tensor = ADTensor::from_tensor(self.value);
            Tensor::new(tensor)
        }
    }
}

#[cfg(feature = "tch")]
mod tch {
    use super::*;
    use crate::tensor::backend::autodiff::ADBackendTch;
    use crate::tensor::backend::tch::TchBackend;
    use crate::TchElement;

    impl<E: TchElement, const D: usize> Tensor<D, TchBackend<E>>
    where
        Standard: rand::distributions::Distribution<E>,
    {
        pub fn with_grad(self) -> Tensor<D, ADBackendTch<E>> {
            let tensor = ADTensor::from_tensor(self.value);
            Tensor::new(tensor)
        }
    }
}
