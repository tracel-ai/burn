//! Extension trait for burn_tensor::TensorData
//!
//! This module provides ONNX-specific helper methods for working with TensorData,
//! including type conversions and scalar extraction.

use burn_tensor::DType;

/// Re-export burn-tensor's TensorData for direct use
pub use burn_tensor::TensorData;

/// Extension trait providing ONNX-specific helper methods for TensorData
pub trait TensorDataExt {
    /// Alias for dtype for backward compatibility with ONNX naming
    fn elem_type(&self) -> DType;

    /// Extract the first element as f64, converting from any numeric type
    ///
    /// Useful for extracting scalar parameters from constant nodes.
    fn scalar_f64(&self) -> Result<f64, burn_tensor::DataError>;

    /// Extract the first element as f32, converting from any numeric type
    fn scalar_f32(&self) -> Result<f32, burn_tensor::DataError>;

    /// Extract the first element as i64, converting from any numeric type
    fn scalar_i64(&self) -> Result<i64, burn_tensor::DataError>;

    /// Convert to `Vec<i64>`, handling Int32 and Int64 types
    ///
    /// Useful for extracting indices, shapes, or other integer arrays that need to be i64.
    fn to_i64_vec(&self) -> Result<Vec<i64>, burn_tensor::DataError>;

    /// Convert to `Vec<f32>`, handling all numeric types with automatic conversion
    ///
    /// Useful for extracting numeric arrays that need to be f32.
    fn to_f32_vec(&self) -> Result<Vec<f32>, burn_tensor::DataError>;
}

impl TensorDataExt for burn_tensor::TensorData {
    fn elem_type(&self) -> DType {
        self.dtype
    }

    fn scalar_f64(&self) -> Result<f64, burn_tensor::DataError> {
        use burn_tensor::DType;
        match self.dtype {
            DType::F16 => {
                let val = self.as_slice::<half::f16>()?[0];
                Ok(f32::from(val) as f64)
            }
            DType::F32 => Ok(self.as_slice::<f32>()?[0] as f64),
            DType::F64 => Ok(self.as_slice::<f64>()?[0]),
            DType::I32 => Ok(self.as_slice::<i32>()?[0] as f64),
            DType::I64 => Ok(self.as_slice::<i64>()?[0] as f64),
            DType::I8 => Ok(self.as_slice::<i8>()?[0] as f64),
            DType::U8 => Ok(self.as_slice::<u8>()?[0] as f64),
            DType::U16 => Ok(self.as_slice::<u16>()?[0] as f64),
            other => Err(burn_tensor::DataError::TypeMismatch(format!(
                "Cannot convert {:?} to f64",
                other
            ))),
        }
    }

    fn scalar_f32(&self) -> Result<f32, burn_tensor::DataError> {
        self.scalar_f64().map(|v| v as f32)
    }

    fn scalar_i64(&self) -> Result<i64, burn_tensor::DataError> {
        use burn_tensor::DType;
        match self.dtype {
            DType::I64 => Ok(self.as_slice::<i64>()?[0]),
            DType::I32 => Ok(self.as_slice::<i32>()?[0] as i64),
            DType::I8 => Ok(self.as_slice::<i8>()?[0] as i64),
            DType::U8 => Ok(self.as_slice::<u8>()?[0] as i64),
            DType::U16 => Ok(self.as_slice::<u16>()?[0] as i64),
            DType::F32 => Ok(self.as_slice::<f32>()?[0] as i64),
            DType::F64 => Ok(self.as_slice::<f64>()?[0] as i64),
            other => Err(burn_tensor::DataError::TypeMismatch(format!(
                "Cannot convert {:?} to i64",
                other
            ))),
        }
    }

    fn to_i64_vec(&self) -> Result<Vec<i64>, burn_tensor::DataError> {
        use burn_tensor::DType;
        match self.dtype {
            DType::I64 | DType::I32 => self.clone().convert::<i64>().to_vec(),
            other => Err(burn_tensor::DataError::TypeMismatch(format!(
                "Cannot convert {:?} to Vec<i64>",
                other
            ))),
        }
    }

    fn to_f32_vec(&self) -> Result<Vec<f32>, burn_tensor::DataError> {
        use burn_tensor::DType;
        match self.dtype {
            DType::F32
            | DType::F64
            | DType::F16
            | DType::I64
            | DType::I32
            | DType::I8
            | DType::U8
            | DType::U16 => self.clone().convert::<f32>().to_vec(),
            other => Err(burn_tensor::DataError::TypeMismatch(format!(
                "Cannot convert {:?} to Vec<f32>",
                other
            ))),
        }
    }
}
