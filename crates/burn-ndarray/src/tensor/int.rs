use crate::NdArrayTensor;
use burn_tensor::{DType, Shape, TensorMetadata};

/// Int tensor primitive.
#[derive(Debug, Clone)]
pub enum NdArrayTensorInt {
    I64(NdArrayTensor<i64>),
    U8(NdArrayTensor<u8>),
}

impl From<NdArrayTensor<i64>> for NdArrayTensorInt {
    fn from(value: NdArrayTensor<i64>) -> Self {
        NdArrayTensorInt::I64(value)
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
            NdArrayTensorInt::U8(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            NdArrayTensorInt::I64(tensor) => tensor.shape(),
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
            burn_tensor::DType::I64 => $crate::NdArrayTensorInt::I64($tensor),
            burn_tensor::DType::U8 => $crate::NdArrayTensorInt::U8($tensor),
            _ => unimplemented!("Unsupported dtype"),
        }
    }};
}

/// Macro to dispatch a float-to-int cast.
/// It takes a float tensor and the generic integer type,
/// and returns the correct integer enum variant.
#[macro_export]
macro_rules! dispatch_int_to_float_cast {
    ($tensor:expr, $int_ty:ty) => {{
        match <$int_ty as burn_tensor::Element>::dtype() {
            burn_tensor::DType::F32 => {
                let array = $tensor.array.mapv(|a| a.elem::<f32>()).into_shared();
                NdArrayTensor::<f32>::new(array).into()
            }
            burn_tensor::DType::F64 => {
                let array = $tensor.array.mapv(|a| a.elem::<f64>()).into_shared();
                NdArrayTensor::<f64>::new(array).into()
            }
            dtype => panic!("Unsupported integer dtype for cast: {:?}", dtype),
        }
    }};
}

/// Macro to execute an operation a given element type.
///
/// # Panics
/// Since there is no automatic type cast at this time, binary operations for different
/// integer point precision data types will panic with a data type mismatch.
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

    // Binary op: type automatically inferred by the compiler but return type is not a int tensor
    (($lhs:expr, $rhs:expr) => $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorInt::I64(lhs), $crate::NdArrayTensorInt::I64(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorInt::U8(lhs), $crate::NdArrayTensorInt::U8(rhs)) => $op(lhs, rhs),
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
            $crate::NdArrayTensorInt::U8(tensor) => $crate::NdArrayTensorInt::U8($op(tensor)),
        }
    }};

    // Unary op: generic type cannot be inferred for an operation
    ($tensor:expr, $element:ident, $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorInt::I64(tensor) => {
                type $element = i64;
                $crate::NdArrayTensorInt::I64($op(tensor))
            }
            $crate::NdArrayTensorInt::U8(tensor) => {
                type $element = u8;
                $crate::NdArrayTensorInt::U8($op(tensor))
            }
        }
    }};

    // Unary op: type automatically inferred by the compiler but return type is not a int tensor
    ($tensor:expr => $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorInt::I64(tensor) => $op(tensor),
            $crate::NdArrayTensorInt::U8(tensor) => $op(tensor),
        }
    }};

    // Unary op: generic type cannot be inferred for an operation and return type is not a int tensor
    ($tensor:expr, $element:ident => $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorInt::I64(tensor) => {
                type $element = i64;
                $op(tensor)
            }
            $crate::NdArrayTensorInt::U8(tensor) => {
                type $element = u8;
                $op(tensor)
            }
        }
    }};
}
