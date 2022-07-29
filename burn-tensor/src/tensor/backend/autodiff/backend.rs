use crate::tensor::{Backend, Data, Element, Tensor, TensorType};
use ndarray::{LinalgScalar, ScalarOperand};
use rand::distributions::{uniform::SampleUniform, Standard};

use super::ADTensor;

#[derive(Debug)]
pub struct ADTensorBackend<E, B> {
    _b: B,
    _e: E,
}

impl<E: Default, B: Backend> Default for ADTensorBackend<E, B> {
    fn default() -> Self {
        Self {
            _b: B::default(),
            _e: E::default(),
        }
    }
}

macro_rules! define_impl {
    ($b:ty) => {
        impl<E> Backend for ADTensorBackend<E, $b>
        where
            E: Element + ScalarOperand + LinalgScalar + SampleUniform,
            E: tch::kind::Element + Into<f64>,
            Standard: rand::distributions::Distribution<E>,
        {
            type E = E;

            fn from_data<const D: usize>(data: Data<E, D>) -> <Self as TensorType<D, Self>>::T
            where
                Self: TensorType<D, Self>,
            {
                <Self as TensorType<D, Self>>::from_data(data)
            }
        }

        impl<E, const D: usize> TensorType<D, Self> for ADTensorBackend<E, $b>
        where
            E: Element + ScalarOperand + LinalgScalar + SampleUniform,
            E: tch::kind::Element + Into<f64>,
            Standard: rand::distributions::Distribution<E>,
        {
            type T = ADTensor<E, D, Tensor<D, $b>>;

            fn from_data(data: Data<E, D>) -> Self::T {
                let tensor = <$b as TensorType<D, $b>>::from_data(data);
                let tensor = ADTensor::from_tensor(tensor);
                tensor
            }
        }
    };
}

define_impl!(crate::tensor::backend::ndarray::NdArrayTensorBackend<E>);
define_impl!(crate::tensor::backend::tch::TchTensorCPUBackend<E>);
