use burn_core::tensor::{DeviceConfig, Element, Tensor};

pub(crate) type FloatElem = f32;
pub(crate) type IntElem = i32;
pub(crate) type TestTensor<const D: usize> = Tensor<D>;

#[ctor::ctor]
fn init_device_settings() {
    let mut device = burn_core::tensor::Device::default();
    device
        .configure(
            DeviceConfig::default()
                .float_dtype(<FloatElem as Element>::dtype())
                .int_dtype(<IntElem as Element>::dtype()),
        )
        .unwrap();
}

mod blackman_window;
#[cfg(not(feature = "capture"))]
mod fft;
mod hamming_window;
mod hann_window;
#[cfg(not(feature = "capture"))]
mod stft;
#[cfg(all(feature = "autodiff", not(feature = "capture")))]
mod autodiff {
    use super::*;
    struct AutodiffDevice;
    impl AutodiffDevice {
        #[allow(clippy::new_ret_no_self)]
        fn new() -> burn_core::tensor::Device {
            burn_core::tensor::Device::default().autodiff()
        }
    }
    mod rfft {
        include!("rfft.rs");
    }

    mod checkpointing {
        use super::super::*;
        struct AutodiffDevice;
        impl AutodiffDevice {
            #[allow(clippy::new_ret_no_self)]
            fn new() -> burn_core::tensor::Device {
                burn_core::tensor::Device::default()
                    .autodiff()
                    .gradient_checkpointing()
            }
        }
        mod rfft {
            include!("rfft.rs");
        }
    }
}
