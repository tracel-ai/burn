use crate::Element;

pub(crate) trait NdArrayElement:
    Element + ndarray::LinalgScalar + ndarray::ScalarOperand + ExpElement + num_traits::FromPrimitive
{
}

pub(crate) trait ExpElement {
    fn exp_elem(self) -> Self;
    fn log_elem(self) -> Self;
    fn pow_elem(self, value: f32) -> Self;
}

macro_rules! impl_exp_elem {
    ($elem:ident) => {
        impl ExpElement for $elem {
            fn exp_elem(self) -> Self {
                $elem::exp(self)
            }
            fn log_elem(self) -> Self {
                $elem::ln(self)
            }
            fn pow_elem(self, value: f32) -> Self {
                $elem::powf(self, value.into())
            }
        }
    };
    ($elem:ident, $tmp:ident) => {
        impl ExpElement for $elem {
            fn exp_elem(self) -> Self {
                let tmp = $tmp::exp(self as $tmp);
                tmp as $elem
            }
            fn log_elem(self) -> Self {
                let tmp = $tmp::ln(self as $tmp);
                tmp as $elem
            }
            fn pow_elem(self, value: f32) -> Self {
                let tmp = $tmp::powf(self as $tmp, value as $tmp);
                tmp as $elem
            }
        }
    };
}

impl NdArrayElement for f64 {}
impl_exp_elem!(f64);

impl NdArrayElement for f32 {}
impl_exp_elem!(f32);

impl NdArrayElement for i64 {}
impl_exp_elem!(i64, f64);

impl NdArrayElement for i32 {}
impl_exp_elem!(i32, f32);

impl NdArrayElement for i16 {}
impl_exp_elem!(i16, f32);

impl NdArrayElement for u8 {}
impl_exp_elem!(u8, f32);
