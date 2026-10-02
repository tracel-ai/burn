use burn_tensor::Element;
use ctor::ctor;

// Re-export
use super::{FloatElem, IntElem};

#[ctor]
fn init_device_settings() {
    // Existing exceptional-value assertions exercise propagation. Run native
    // policy coverage in a separate process with BURN_TEST_NAN_POLICY=native.
    #[cfg(feature = "cube")]
    {
        use burn_std::config::{BurnConfig, NanPolicy, RuntimeConfig};
        let policy = match std::env::var("BURN_TEST_NAN_POLICY").as_deref() {
            Ok("native") => NanPolicy::Native,
            Ok("propagate") | Err(_) => NanPolicy::Propagate,
            Ok(value) => panic!("invalid BURN_TEST_NAN_POLICY: {value}"),
        };
        BurnConfig::set(BurnConfig::default().with_nan_policy(policy));
    }
    let mut device = burn_tensor::Device::default();
    device
        .configure(
            burn_tensor::DeviceConfig::default()
                .float_dtype(<FloatElem as Element>::dtype())
                .int_dtype(<IntElem as Element>::dtype()),
        )
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
