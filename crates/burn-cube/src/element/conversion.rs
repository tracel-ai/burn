use crate::{Bool, Float, Int, PrimitiveVariable, UInt, BF16, F16, F32, F64, I32, I64};

// Enable elegant casting from any to any primitive variable

macro_rules! impl_to_float {
    ($to:ident, $from1:ident) => {
        impl From<$from1> for $to {
            fn from(value: $from1) -> Self {
                Self::from_primitive(value.val() as f64)
            }
        }
    };
}

macro_rules! impl_to_float_from_bool {
    ($to:ident, $from1:ident) => {
        impl From<$from1> for $to {
            fn from(value: $from1) -> Self {
                Self::from_primitive(match value.val() {
                    true => 1.,
                    false => 0.,
                })
            }
        }
    };
}

impl_to_float!(F16, BF16);
impl_to_float!(F16, F32);
impl_to_float!(F16, F64);
impl_to_float!(F16, I32);
impl_to_float!(F16, I64);
impl_to_float!(F16, UInt);
impl_to_float_from_bool!(F16, Bool);

impl_to_float!(BF16, F16);
impl_to_float!(BF16, F32);
impl_to_float!(BF16, F64);
impl_to_float!(BF16, I32);
impl_to_float!(BF16, I64);
impl_to_float!(BF16, UInt);
impl_to_float_from_bool!(BF16, Bool);

impl_to_float!(F32, F16);
impl_to_float!(F32, BF16);
impl_to_float!(F32, F64);
impl_to_float!(F32, I32);
impl_to_float!(F32, I64);
impl_to_float!(F32, UInt);
impl_to_float_from_bool!(F32, Bool);

impl_to_float!(F64, F16);
impl_to_float!(F64, BF16);
impl_to_float!(F64, F32);
impl_to_float!(F64, I32);
impl_to_float!(F64, I64);
impl_to_float!(F64, UInt);
impl_to_float_from_bool!(F64, Bool);

macro_rules! impl_to_int {
    ($to:ident, $from1:ident) => {
        impl From<$from1> for $to {
            fn from(value: $from1) -> Self {
                Self::from_primitive(value.val() as i64)
            }
        }
    };
}

macro_rules! impl_to_int_from_bool {
    ($to:ident, $from1:ident) => {
        impl From<$from1> for $to {
            fn from(value: $from1) -> Self {
                Self::from_primitive(match value.val() {
                    true => 1,
                    false => 0,
                })
            }
        }
    };
}

impl_to_int!(I32, F16);
impl_to_int!(I32, BF16);
impl_to_int!(I32, F32);
impl_to_int!(I32, F64);
impl_to_int!(I32, I64);
impl_to_int!(I32, UInt);
impl_to_int_from_bool!(I32, Bool);

impl_to_int!(I64, F16);
impl_to_int!(I64, BF16);
impl_to_int!(I64, F32);
impl_to_int!(I64, F64);
impl_to_int!(I64, I32);
impl_to_int!(I64, UInt);
impl_to_int_from_bool!(I64, Bool);

impl_to_int!(UInt, F16);
impl_to_int!(UInt, BF16);
impl_to_int!(UInt, F32);
impl_to_int!(UInt, F64);
impl_to_int!(UInt, I32);
impl_to_int!(UInt, I64);
impl_to_int_from_bool!(UInt, Bool);

macro_rules! impl_to_bool_from_float {
    ($to:ident, $from1:ident) => {
        impl From<$from1> for $to {
            fn from(value: $from1) -> Self {
                Self::from_primitive(value.val() > 0.)
            }
        }
    };
}

macro_rules! impl_to_bool_from_int {
    ($to:ident, $from1:ident) => {
        impl From<$from1> for $to {
            fn from(value: $from1) -> Self {
                Self::from_primitive(value.val() > 0)
            }
        }
    };
}

impl_to_bool_from_float!(Bool, F16);
impl_to_bool_from_float!(Bool, BF16);
impl_to_bool_from_float!(Bool, F32);
impl_to_bool_from_float!(Bool, F64);
impl_to_bool_from_int!(Bool, I32);
impl_to_bool_from_int!(Bool, I64);
impl_to_bool_from_int!(Bool, UInt);
