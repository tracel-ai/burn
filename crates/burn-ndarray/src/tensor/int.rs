use crate::NdArrayTensor;
use burn_tensor::{DType, Shape, TensorMetadata};

/// Int tensor primitive.
#[derive(Debug, Clone)]
pub enum NdArrayTensorInt {
    I64(NdArrayTensor<i64>),
    I32(NdArrayTensor<i32>),
    I16(NdArrayTensor<i16>),
    I8(NdArrayTensor<i8>),
    U64(NdArrayTensor<u64>),
    U32(NdArrayTensor<u32>),
    U16(NdArrayTensor<u16>),
    U8(NdArrayTensor<u8>),
}

impl From<NdArrayTensor<i64>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<i64>) -> Self {
        NdArrayTensorInt::I64(value)
    }
}
impl From<NdArrayTensor<i32>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<i32>) -> Self {
        NdArrayTensorInt::I32(value)
    }
}

impl From<NdArrayTensor<i16>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<i16>) -> Self {
        NdArrayTensorInt::I16(value)
    }
}

impl From<NdArrayTensor<i8>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<i8>) -> Self {
        NdArrayTensorInt::I8(value)
    }
}
impl From<NdArrayTensor<u64>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<u64>) -> Self {
        NdArrayTensorInt::U64(value)
    }
}

impl From<NdArrayTensor<u32>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<u32>) -> Self {
        NdArrayTensorInt::U32(value)
    }
}

impl From<NdArrayTensor<u16>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<u16>) -> Self {
        NdArrayTensorInt::U16(value)
    }
}

impl From<NdArrayTensor<u8>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<u8>) -> Self {
        NdArrayTensorInt::U8(value)
    }
}

impl TensorMetadata for NdArrayTensorInt {
    fn dtype(&self) -> DType {
        match self {
            NdArrayTensorInt::I64(tensor) => tensor.dtype(),
            NdArrayTensorInt::I32(tensor) => tensor.dtype(),
            NdArrayTensorInt::I16(tensor) => tensor.dtype(),
            NdArrayTensorInt::I8(tensor) => tensor.dtype(),
            NdArrayTensorInt::U64(tensor) => tensor.dtype(),
            NdArrayTensorInt::U32(tensor) => tensor.dtype(),
            NdArrayTensorInt::U16(tensor) => tensor.dtype(),
            NdArrayTensorInt::U8(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            NdArrayTensorInt::I64(tensor) => tensor.shape(),
            NdArrayTensorInt::I32(tensor) => tensor.shape(),
            NdArrayTensorInt::I16(tensor) => tensor.shape(),
            NdArrayTensorInt::I8(tensor) => tensor.shape(),
            NdArrayTensorInt::U64(tensor) => tensor.shape(),
            NdArrayTensorInt::U32(tensor) => tensor.shape(),
            NdArrayTensorInt::U16(tensor) => tensor.shape(),
            NdArrayTensorInt::U8(tensor) => tensor.shape(),
        }
    }
}

/// Macro to create a new [int tensor](NdArrayIntTensor) based on the element type.
#[macro_export]
macro_rules! new_tensor_int {
    // Op executed with default dtype
    ($tensor:expr) => {{
        match E::dtype() {
            burn_tensor::DType::I64 => $crate::NdArrayTensorFloat::I64($tensor),
            burn_tensor::DType::I32 => $crate::NdArrayTensorFloat::I32($tensor),
            burn_tensor::DType::I16 => $crate::NdArrayTensorFloat::I16($tensor),
            burn_tensor::DType::I8 => $crate::NdArrayTensorFloat::I8($tensor),
            burn_tensor::DType::U64 => $crate::NdArrayTensorFloat::U64($tensor),
            burn_tensor::DType::U32 => $crate::NdArrayTensorFloat::U32($tensor),
            burn_tensor::DType::U16 => $crate::NdArrayTensorFloat::U16($tensor),
            burn_tensor::DType::U8 => $crate::NdArrayTensorFloat::U8($tensor),
            _ => unimplemented!("Unsupported dtype"),
        }
    }};
}

/// Macro to execute an operation a given element type.
///
/// # Panics
/// Since there is no automatic type cast at this time, binary operations for different
/// floating point precision data types will panic with a data type mismatch.
#[macro_export]
macro_rules! execute_with_int_dtype {
    // Binary op: type automatically inferred by the compiler
    (($lhs:expr, $rhs:expr), $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorInt::I64(lhs), $crate::NdArrayTensorInt::I64(rhs)) => {
                $crate::NdArrayTensorInt::I64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::I32(lhs), $crate::NdArrayTensorInt::I32(rhs)) => {
                $crate::NdArrayTensorInt::I32($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::I16(lhs), $crate::NdArrayTensorInt::I16(rhs)) => {
                $crate::NdArrayTensorInt::I16($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::I8(lhs), $crate::NdArrayTensorInt::I8(rhs)) => {
                $crate::NdArrayTensorInt::I8($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U64(lhs), $crate::NdArrayTensorInt::U64(rhs)) => {
                $crate::NdArrayTensorInt::U64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U32(lhs), $crate::NdArrayTensorInt::U32(rhs)) => {
                $crate::NdArrayTensorInt::U32($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U16(lhs), $crate::NdArrayTensorInt::U16(rhs)) => {
                $crate::NdArrayTensorInt::U16($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U8(lhs), $crate::NdArrayTensorInt::U8(rhs)) => {
                $crate::NdArrayTensorInt::U8($op(lhs, rhs))
            }
            _ => panic!(
                "Data type mismatch (lhs: {:?}, rhs: {:?})",
                lhs_dtype, rhs_dtype
            ),
        }
    }};

    // Binary op: generic type cannot be inferred for an operation
    (($lhs:expr, $rhs:expr), $element:ident, $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorInt::I64(lhs), $crate::NdArrayTensorInt::I64(rhs)) => {
                type $element = i64;
                $crate::NdArrayTensorInt::I64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::I32(lhs), $crate::NdArrayTensorInt::I32(rhs)) => {
                type $element = i32;
                $crate::NdArrayTensorInt::I32($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::I16(lhs), $crate::NdArrayTensorInt::I16(rhs)) => {
                type $element = i16;
                $crate::NdArrayTensorInt::I16($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::I8(lhs), $crate::NdArrayTensorInt::I8(rhs)) => {
                type $element = i8;
                $crate::NdArrayTensorInt::I8($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U64(lhs), $crate::NdArrayTensorInt::U64(rhs)) => {
                type $element = u64;
                $crate::NdArrayTensorInt::U64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U32(lhs), $crate::NdArrayTensorInt::U32(rhs)) => {
                type $element = u32;
                  $crate::NdArrayTensorInt::U32($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U16(lhs), $crate::NdArrayTensorInt::U16(rhs)) => {
                type $element = u16;
                $crate::NdArrayTensorInt::U16($op(lhs, rhs))
            }
            ($crate::NdArrayTensorInt::U8(lhs), $crate::NdArrayTensorInt::U8(rhs)) => {
                type $element = u8;
                $crate::NdArrayTensorInt::U8($op(lhs, rhs))
            }
            _ => panic!(
                "Data type mismatch (lhs: {:?}, rhs: {:?})",
                lhs_dtype, rhs_dtype
            ),
        }
    }};

    // Binary op: type automatically inferred by the compiler but return type is not a float tensor
    (($lhs:expr, $rhs:expr) => $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorInt::I64(lhs), $crate::NdArrayTensorInt::I64(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::I32(lhs), $crate::NdArrayTensorInt::I32(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::I16(lhs), $crate::NdArrayTensorInt::I16(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::I8(lhs), $crate::NdArrayTensorInt::I8(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::U64(lhs), $crate::NdArrayTensorInt::U64(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::U32(lhs), $crate::NdArrayTensorInt::U32(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::U16(lhs), $crate::NdArrayTensorInt::U16(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::U8(lhs), $crate::NdArrayTensorInt::U8(rhs)) {
                $op(lhs, rhs)
            }
            _ => panic!(
                "Data type mismatch (lhs: {:?}, rhs: {:?})",
                lhs_dtype, rhs_dtype
            ),
        }
    }};

    // Unary op: type automatically inferred by the compiler
    ($tensor:expr, $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorInt::I64(tensor) => $crate::NdArrayTensorInt::I64($op(tensor)),
            $crate::NdArrayTensorInt::I32(tensor) => $crate::NdArrayTensorInt::I32($op(tensor)),
            $crate::NdArrayTensorInt::I16(tensor) => $crate::NdArrayTensorInt::I16($op(tensor)),
            $crate::NdArrayTensorInt::I8(tensor) => $crate::NdArrayTensorInt::I8($op(tensor)),
            $crate::NdArrayTensorInt::U64(tensor) => $crate::NdArrayTensorInt::U64($op(tensor)),
            $crate::NdArrayTensorInt::U32(tensor) => $crate::NdArrayTensorInt::U32($op(tensor)),
            $crate::NdArrayTensorInt::U16(tensor) => $crate::NdArrayTensorInt::U16($op(tensor)),
            $crate::NdArrayTensorInt::U8(tensor) => $crate::NdArrayTensorInt::U8($op(tensor)),
        }
    }};

    // Unary op: generic type cannot be inferred for an operation
    ($tensor:expr, $element:ident, $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorFloat::I64(tensor) => {
                type $element = i64;
                $crate::NdArrayTensorFloat::I64($op(tensor))
            }
            $crate::NdArrayTensorFloat::I32(tensor) => {
                type $element = i32;
                $crate::NdArrayTensorFloat::I32($op(tensor))
            }
            $crate::NdArrayTensorFloat::I16(tensor) => {
                type $element = i16;
                $crate::NdArrayTensorFloat::I16($op(tensor))
            }
            $crate::NdArrayTensorFloat::I8(tensor) => {
                type $element = i8;
                $crate::NdArrayTensorFloat::I8($op(tensor))
            }
            $crate::NdArrayTensorFloat::U64(tensor) => {
                type $element = u64;
                $crate::NdArrayTensorFloat::U64($op(tensor))
            }
            $crate::NdArrayTensorFloat::U32(tensor) => {
                type $element = u32;
                $crate::NdArrayTensorFloat::U32($op(tensor))
            }
            $crate::NdArrayTensorFloat::U16(tensor) => {
                type $element = u16;
                $crate::NdArrayTensorFloat::U16($op(tensor))
            }
            $crate::NdArrayTensorFloat::U8(tensor) => {
                type $element = u8;
                $crate::NdArrayTensorFloat::U8($op(tensor))
            }
        }
    }};

    // Unary op: type automatically inferred by the compiler but return type is not a float tensor
    ($tensor:expr => $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorInt::I64(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::I32(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::I16(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::I8(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::U64(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::U32(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::U16(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::U8(tensor) => $op(tensor),
        }
    }};

    // Unary op: generic type cannot be inferred for an operation and return type is not a float tensor
    ($tensor:expr, $element:ident => $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorInt::I64(tensor) => {
                type $element = i64;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::I32(tensor) => {
                type $element = i32;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::I16(tensor) => {
                type $element = i16;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::I8(tensor) => {
                type $element = i8;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::U64(tensor) => {
                type $element = u64;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::U32(tensor) => {
                type $element = u32;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::U16(tensor) => {
                type $element = u16;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::U8(tensor) => {
                type $element = u8;
                $op(tensor)
            }
        }

    }};
}
