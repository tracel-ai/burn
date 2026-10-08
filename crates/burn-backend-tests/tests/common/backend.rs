use burn_tensor::Element;
use ctor::ctor;

// Re-export
use super::{FloatElem, IntElem};

#[ctor(unsafe)]
fn init_device_settings() {
    let _ = std::panic::catch_unwind(|| {
        let mut device = burn_tensor::Device::default();
        let _ = device.configure(
            burn_tensor::DeviceConfig::default()
                .float_dtype(<FloatElem as Element>::dtype())
                .int_dtype(<IntElem as Element>::dtype()),
        );
    });
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
