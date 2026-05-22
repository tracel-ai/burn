use burn_tensor::{Element, FloatDType, IntDType};
use ctor::ctor;

// Re-export
use super::{FloatElem, IntElem};

#[ctor]
fn init_device_settings() {
    let mut device = burn_tensor::Device::default();
    device
        .configure((
            FloatDType::from(<FloatElem as Element>::dtype()),
            IntDType::from(<IntElem as Element>::dtype()),
        ))
        .unwrap();
}

/// Collection of types used across tests
#[allow(unused)]
pub mod prelude {
    pub use burn_tensor::Tensor;

    use super::*;
    pub type TestTensor<const D: usize> = Tensor<D>;
    pub type TestTensorInt<const D: usize> = Tensor<D, burn_tensor::Int>;
    pub type TestTensorBool<const D: usize> = Tensor<D, burn_tensor::Bool>;
}

#[allow(unused)]
pub use prelude::*;
