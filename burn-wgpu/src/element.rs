use burn_tensor::Element;
use bytemuck::Pod;

pub trait WGPUElement: Element + core::fmt::Debug + Pod {
    fn type_name() -> &'static str;
}

pub trait FloatElement: WGPUElement {}

pub trait IntElement: WGPUElement {}

impl WGPUElement for i32 {
    fn type_name() -> &'static str {
        "i32"
    }
}

impl WGPUElement for i64 {
    fn type_name() -> &'static str {
        "i64"
    }
}

impl WGPUElement for f32 {
    fn type_name() -> &'static str {
        "f32"
    }
}

impl FloatElement for f32 {}
impl IntElement for i32 {}
impl IntElement for i64 {}
