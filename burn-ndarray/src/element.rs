use burn_tensor::Element;
use libm::{exp, log, log1p, pow, sqrt};
use libm::{expf, log1pf, logf, powf, sqrtf};

pub(crate) trait NdArrayElement:
    Element
    + ndarray::LinalgScalar
    + ndarray::ScalarOperand
    + ExpElement
    + num_traits::FromPrimitive
    + core::cmp::PartialEq
    + core::cmp::PartialOrd<Self>
{
}

pub(crate) trait ExpElement {
    fn exp_elem(self) -> Self;
    fn log_elem(self) -> Self;
    fn log1p_elem(self) -> Self;
    fn pow_elem(self, value: f32) -> Self;
    fn sqrt_elem(self) -> Self;
}

impl NdArrayElement for f64 {}
impl ExpElement for f64 {
    fn exp_elem(self) -> Self {
        exp(self)
    }

    fn log_elem(self) -> Self {
        log(self)
    }

    fn log1p_elem(self) -> Self {
        log1p(self)
    }

    fn pow_elem(self, value: f32) -> Self {
        pow(self, value.into())
    }

    fn sqrt_elem(self) -> Self {
        sqrt(self)
    }
}

impl NdArrayElement for f32 {}
impl ExpElement for f32 {
    fn exp_elem(self) -> Self {
        expf(self)
    }

    fn log_elem(self) -> Self {
        logf(self)
    }

    fn log1p_elem(self) -> Self {
        log1pf(self)
    }

    fn pow_elem(self, value: f32) -> Self {
        powf(self, value)
    }

    fn sqrt_elem(self) -> Self {
        sqrtf(self)
    }
}

impl NdArrayElement for i64 {}
impl ExpElement for i64 {
    fn exp_elem(self) -> Self {
        exp(self as f64) as i64
    }

    fn log_elem(self) -> Self {
        log(self as f64) as i64
    }

    fn log1p_elem(self) -> Self {
        log1p(self as f64) as i64
    }

    fn pow_elem(self, value: f32) -> Self {
        pow(self as f64, value.into()) as i64
    }

    fn sqrt_elem(self) -> Self {
        sqrt(self as f64) as i64
    }
}

impl NdArrayElement for i32 {}
impl ExpElement for i32 {
    fn exp_elem(self) -> Self {
        expf(self as f32) as i32
    }

    fn log_elem(self) -> Self {
        logf(self as f32) as i32
    }

    fn log1p_elem(self) -> Self {
        log1pf(self as f32) as i32
    }

    fn pow_elem(self, value: f32) -> Self {
        powf(self as f32, value) as i32
    }

    fn sqrt_elem(self) -> Self {
        sqrtf(self as f32) as i32
    }
}

impl NdArrayElement for i16 {}
impl ExpElement for i16 {
    fn exp_elem(self) -> Self {
        expf(self as f32) as i16
    }

    fn log_elem(self) -> Self {
        logf(self as f32) as i16
    }

    fn log1p_elem(self) -> Self {
        log1pf(self as f32) as i16
    }

    fn pow_elem(self, value: f32) -> Self {
        powf(self as f32, value) as i16
    }

    fn sqrt_elem(self) -> Self {
        sqrtf(self as f32) as i16
    }
}

impl NdArrayElement for u8 {}
impl ExpElement for u8 {
    fn exp_elem(self) -> Self {
        expf(self as f32) as u8
    }

    fn log_elem(self) -> Self {
        logf(self as f32) as u8
    }

    fn log1p_elem(self) -> Self {
        log1pf(self as f32) as u8
    }

    fn pow_elem(self, value: f32) -> Self {
        powf(self as f32, value) as u8
    }

    fn sqrt_elem(self) -> Self {
        sqrtf(self as f32) as u8
    }
}
