use super::Tensor;
use crate::graph::grad::Gradients;
use crate::tensor::backend::autodiff::ADTensor;
use crate::tensor::backend::ADBackend;

impl<const D: usize, B: ADBackend> Tensor<B, D> {
    pub fn backward(&self) -> Gradients {
        B::backward::<D>(&self.value)
    }

    pub fn grad(&self, grads: &Gradients) -> Option<Tensor<B::InnerBackend, D>> {
        B::grad(&self.value, grads).map(|value| Tensor::new(value))
    }

    pub fn inner(&self) -> Tensor<B::InnerBackend, D> {
        Tensor::new(B::inner(&self.value))
    }

    pub fn update(&mut self, other_inner: Tensor<B::InnerBackend, D>) {
        self.value = B::from_inner(other_inner.value);
    }

    pub fn from_inner(inner: Tensor<B::InnerBackend, D>) -> Self {
        Self::new(B::from_inner(inner.value))
    }
}

#[cfg(feature = "ndarray")]
mod ndarray {
    use super::*;
    use crate::tensor::backend::autodiff::ADBackendNdArray;
    use crate::tensor::backend::ndarray::NdArrayBackend;

    impl<E: crate::NdArrayElement, const D: usize> Tensor<NdArrayBackend<E>, D> {
        pub fn with_grad(self) -> Tensor<ADBackendNdArray<E>, D> {
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

    impl<E: TchElement, const D: usize> Tensor<TchBackend<E>, D> {
        pub fn with_grad(self) -> Tensor<ADBackendTch<E>, D> {
            let tensor = ADTensor::from_tensor(self.value);
            Tensor::new(tensor)
        }
    }
}
