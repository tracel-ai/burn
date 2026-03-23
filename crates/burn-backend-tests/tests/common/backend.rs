use ctor::ctor;

// Re-export
use super::{FloatElem, IntElem};

// Default
pub type TestBackend = burn_dispatch::Dispatch;

#[ctor]
fn init_device_settings() {
    let device = burn_dispatch::DispatchDevice::default();
    burn_tensor::set_default_dtypes::<TestBackend>(
        &device,
        <FloatElem as burn_tensor::Element>::dtype(),
        <IntElem as burn_tensor::Element>::dtype(),
    )
    .unwrap();
}

/// Collection of types used across tests
#[allow(unused)]
pub mod prelude {
    pub use burn_tensor::Tensor;

    use super::*;
    pub type TestTensor<const D: usize> = Tensor<TestBackend, D>;
    pub type TestTensorInt<const D: usize> = Tensor<TestBackend, D, burn_tensor::Int>;
    pub type TestTensorBool<const D: usize> = Tensor<TestBackend, D, burn_tensor::Bool>;
}

#[allow(unused)]
pub use prelude::*;
