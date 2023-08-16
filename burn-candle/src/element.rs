use burn_tensor::Element;
use candle_core::WithDType;
use half::{bf16, f16};

pub trait CandleElement: Element + WithDType {}

impl CandleElement for f64 {}
impl CandleElement for f32 {}
impl CandleElement for f16 {}
impl CandleElement for bf16 {}

impl CandleElement for u8 {}
impl CandleElement for u32 {}
// impl CandleElement for i32 {}
