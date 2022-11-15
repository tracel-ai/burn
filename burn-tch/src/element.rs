use burn_tensor::Element;
use half::f16;

pub trait TchElement: Element + tch::kind::Element {}

impl TchElement for f64 {}
impl TchElement for f32 {}
impl TchElement for f16 {}

impl TchElement for i64 {}
impl TchElement for i32 {}
impl TchElement for i16 {}

impl TchElement for u8 {}
