use burn_tensor::Element;
use bytemuck::Pod;

pub trait FloatElement: Element + core::fmt::Debug + Pod {
    fn type_name() -> &'static str;
}

pub trait IntElement: Element + core::fmt::Debug + Pod {
    fn type_name() -> &'static str;
}

impl IntElement for i32 {
    fn type_name() -> &'static str {
        "i32"
    }
}

impl FloatElement for f32 {
    fn type_name() -> &'static str {
        "f32"
    }
}

impl IntElement for i64 {
    fn type_name() -> &'static str {
        "i64"
    }
}
