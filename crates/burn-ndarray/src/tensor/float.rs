use crate::NdArrayTensor;
use burn_tensor::{DType, Shape, TensorMetadata};

/// Float tensor primitive.
#[derive(Debug, Clone)]
pub enum NdArrayTensorFloat {
    /// 32-bit float.
    F32(NdArrayTensor<f32>),
    /// 64-bit float.
    F64(NdArrayTensor<f64>),
}

impl From<NdArrayTensor<f32>> for NdArrayTensorFloat {
    fn from(value: NdArrayTensor<f32>) -> Self {
        NdArrayTensorFloat::F32(value)
    }
}

impl From<NdArrayTensor<f64>> for NdArrayTensorFloat {
    fn from(value: NdArrayTensor<f64>) -> Self {
        NdArrayTensorFloat::F64(value)
    }
}

impl TensorMetadata for NdArrayTensorFloat {
    fn dtype(&self) -> DType {
        match self {
            NdArrayTensorFloat::F32(tensor) => tensor.dtype(),
            NdArrayTensorFloat::F64(tensor) => tensor.dtype(),
        }
    }

    fn shape(&self) -> Shape {
        match self {
            NdArrayTensorFloat::F32(tensor) => tensor.shape(),
            NdArrayTensorFloat::F64(tensor) => tensor.shape(),
        }
    }
}

/// Macro to create a new [float tensor](NdArrayTensorFloat) based on the element type.
#[macro_export]
macro_rules! new_tensor_float {
    // Op executed with default dtype
    ($tensor:expr) => {{
        match E::dtype() {
            burn_tensor::DType::F64 => $crate::NdArrayTensorFloat::F64($tensor),
            burn_tensor::DType::F32 => $crate::NdArrayTensorFloat::F32($tensor),
            // FloatNdArrayElement only implemented for f64 and f32
            _ => unimplemented!("Unsupported dtype"),
        }
    }};
}

/// Macro to dispatch a float-to-int cast.
/// It takes a float tensor and the generic integer type,
/// and returns the correct integer enum variant.
#[macro_export]
macro_rules! dispatch_float_to_int_cast {
    ($tensor:expr, $int_ty:ty) => {{
        match <$int_ty as burn_tensor::Element>::dtype() {
            burn_tensor::DType::I64 => {
                let array = $tensor.array.mapv(|a| a.elem::<i64>()).into_shared();
                NdArrayTensor::<i64>::new(array).into()
            }
            burn_tensor::DType::U8 => {
                let array = $tensor.array.mapv(|a| a.elem::<u8>()).into_shared();
                NdArrayTensor::<u8>::new(array).into()
            }
            dtype => panic!("Unsupported integer dtype for cast: {:?}", dtype),
        }
    }};
}

/// Macro to execute an operation a given element type.
///
/// # Panics
/// Since there is no automatic type cast at this time, binary operations for different
/// floating point precision data types will panic with a data type mismatch.
#[macro_export]
macro_rules! execute_with_float_dtype {
    // Binary op: type automatically inferred by the compiler
    (($lhs:expr, $rhs:expr), $op:expr) => {{
        let lhs_dtype = burn_tensor::TensorMetadata::dtype(&$lhs);
        let rhs_dtype = burn_tensor::TensorMetadata::dtype(&$rhs);
        match ($lhs, $rhs) {
            ($crate::NdArrayTensorFloat::F64(lhs), $crate::NdArrayTensorFloat::F64(rhs)) => {
                $crate::NdArrayTensorFloat::F64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorFloat::F32(lhs), $crate::NdArrayTensorFloat::F32(rhs)) => {
                $crate::NdArrayTensorFloat::F32($op(lhs, rhs))
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
            ($crate::NdArrayTensorFloat::F64(lhs), $crate::NdArrayTensorFloat::F64(rhs)) => {
                type $element = f64;
                $crate::NdArrayTensorFloat::F64($op(lhs, rhs))
            }
            ($crate::NdArrayTensorFloat::F32(lhs), $crate::NdArrayTensorFloat::F32(rhs)) => {
                type $element = f32;
                $crate::NdArrayTensorFloat::F32($op(lhs, rhs))
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
            ($crate::NdArrayTensorFloat::F64(lhs), $crate::NdArrayTensorFloat::F64(rhs)) => {
                $op(lhs, rhs)
            }
            ($crate::NdArrayTensorFloat::F32(lhs), $crate::NdArrayTensorFloat::F32(rhs)) => {
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
            $crate::NdArrayTensorFloat::F64(tensor) => $crate::NdArrayTensorFloat::F64($op(tensor)),
            $crate::NdArrayTensorFloat::F32(tensor) => $crate::NdArrayTensorFloat::F32($op(tensor)),
        }
    }};

    // Unary op: generic type cannot be inferred for an operation
    ($tensor:expr, $element:ident, $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorFloat::F64(tensor) => {
                type $element = f64;
                $crate::NdArrayTensorFloat::F64($op(tensor))
            }
            $crate::NdArrayTensorFloat::F32(tensor) => {
                type $element = f32;
                $crate::NdArrayTensorFloat::F32($op(tensor))
            }
        }
    }};

    // Unary op: type automatically inferred by the compiler but return type is not a float tensor
    ($tensor:expr => $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorFloat::F64(tensor) => $op(tensor),
            $crate::NdArrayTensorFloat::F32(tensor) => $op(tensor),
        }
    }};

    // Unary op: generic type cannot be inferred for an operation and return type is not a float tensor
    ($tensor:expr, $element:ident => $op:expr) => {{
        match $tensor {
            $crate::NdArrayTensorFloat::F64(tensor) => {
                type $element = f64;
                $op(tensor)
            }
            $crate::NdArrayTensorFloat::F32(tensor) => {
                type $element = f32;
                $op(tensor)
            }
        }
    }};
}
