use burn_core::tensor::{DeviceConfig, Element, Int, Tensor};

pub(crate) type FloatElem = f32;
pub(crate) type IntElem = i32;
pub(crate) type TestTensor<const D: usize> = Tensor<D>;
pub(crate) type TestTensorInt<const D: usize> = Tensor<D, Int>;

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

pub(crate) mod cosine_similarity;
pub(crate) mod det;
pub(crate) mod diag;
pub(crate) mod lu;
pub(crate) mod matvec;
pub(crate) mod outer;
pub(crate) mod qr;
pub(crate) mod svd;
pub(crate) mod trace;
pub(crate) mod vector_norm;
