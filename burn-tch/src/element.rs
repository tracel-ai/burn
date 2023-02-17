use burn_tensor::Element;
use half::f16;

pub trait IsInt {
    fn is_int(&self) -> bool;
}
pub trait TchElement: Element + tch::kind::Element + IsInt {}

macro_rules! make_element {
    (
        $ty:ident,
        $bool:expr

    ) => {
        impl IsInt for $ty {
            fn is_int(&self) -> bool {
                $bool
            }
        }
    };
}

impl TchElement for f64 {}
impl TchElement for f32 {}
impl TchElement for f16 {}

impl TchElement for i64 {}
impl TchElement for i32 {}
impl TchElement for i16 {}

impl TchElement for u8 {}

make_element!(f64, false);
make_element!(f32, false);
make_element!(f16, false);
make_element!(i64, true);
make_element!(i32, true);
make_element!(i16, true);
make_element!(u8, false);
