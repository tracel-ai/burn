use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;

use bytemuck::{AnyBitPattern, CheckedBitPattern, Zeroable, cast_mut};

use crate::element::{Element, ElementConversion};
use crate::tensor::DType;
use crate::{
    BoolStore, QuantMode, QuantScheme, QuantValue, QuantizedBytes, Reader, Writer, bf16, f16,
};

use super::{DataError, TensorData};

impl TensorData {
    /// Copies and converts the data to a [`Vec<E>`].
    ///
    /// By contract, this is equivalent to:
    /// `data.clone().try_into_vec_as::<E>()`
    ///
    /// Particular conversions may provide more efficient implementations.
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails, the conversion isn't supported, or the stored
    /// representation or element count is invalid.
    pub fn try_to_vec_as<E: Element>(&self) -> Result<Vec<E>, DataError> {
        self.clone().try_into_vec_as::<E>()
    }

    /// Converts the data to a [`Vec<E>`].
    ///
    /// By contract, this is equivalent to:
    /// `data.try_cast_as::<E>()?.try_into_vec::<E>()`
    ///
    /// Particular conversions may provide more efficient implementations.
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails, the conversion isn't supported, or the stored
    /// representation or element count is invalid.
    pub fn try_into_vec_as<E: Element>(self) -> Result<Vec<E>, DataError> {
        self.try_cast_as::<E>()?.into_vec_unchecked::<E>()
    }

    /// Copies the stored values to a vector without dtype conversion.
    #[deprecated(since = "0.22.0", note = "use try_to_vec::<E>()")]
    pub fn to_vec<E: Element>(&self) -> Result<Vec<E>, DataError> {
        self.try_to_vec::<E>()
    }

    /// Converts the stored values into a vector without dtype conversion.
    #[deprecated(since = "0.22.0", note = "use try_into_vec::<E>()")]
    pub fn into_vec<E: Element>(self) -> Result<Vec<E>, DataError> {
        self.try_into_vec::<E>()
    }

    /// Copies the stored values to a vector without dtype conversion.
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails, the stored dtype doesn't match `E`, or the byte
    /// representation is invalid for `E`.
    pub fn try_to_vec<E: Element>(&self) -> Result<Vec<E>, DataError> {
        Ok(self.as_slice()?.to_vec())
    }

    /// Converts the stored values into a vector without dtype conversion.
    ///
    /// This may reuse the underlying allocation when its layout and ownership permit it.
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails, the stored dtype doesn't match `E`, or the byte
    /// representation is invalid for `E`.
    pub fn try_into_vec<E: Element>(self) -> Result<Vec<E>, DataError> {
        // This means we cannot call `into_vec` for QFloat
        if !self.matches_target_dtype::<E>() {
            return Err(DataError::dtype_mismatch_as::<E>(self.dtype));
        }

        self.into_vec_unchecked()
    }

    /// Returns the tensor data as a vector of scalar values. Does not check dtype.
    fn into_vec_unchecked<E: Element>(self) -> Result<Vec<E>, DataError> {
        let mut me = self;
        me.bytes.read(Reader::new())?;
        me.bytes = match me.bytes.try_into_vec::<E>() {
            Ok(elems) => return Ok(elems),
            Err(bytes) => bytes,
        };

        // The bytes might have been deserialized and allocated with a different align.
        // In that case, we have to memcopy the data into a new vector, more suitably allocated
        Ok(
            bytemuck::checked::try_cast_slice(me.bytes.read(Reader::new())?)
                .map_err(DataError::InvalidRepresentation)?
                .to_vec(),
        )
    }

    /// Returns an iterator over the values of the tensor data.
    pub fn iter<E: Element>(&self) -> Box<dyn Iterator<Item = E> + '_> {
        if E::dtype() == self.dtype {
            Box::new(bytemuck::checked::cast_slice(&self.bytes).iter().copied())
        } else {
            match self.dtype {
                DType::I8 => Box::new(
                    bytemuck::checked::cast_slice(&self.bytes)
                        .iter()
                        .map(|e: &i8| e.elem::<E>()),
                ),
                DType::I16 => Box::new(
                    bytemuck::checked::cast_slice(&self.bytes)
                        .iter()
                        .map(|e: &i16| e.elem::<E>()),
                ),
                DType::I32 => Box::new(
                    bytemuck::checked::cast_slice(&self.bytes)
                        .iter()
                        .map(|e: &i32| e.elem::<E>()),
                ),
                DType::I64 => Box::new(
                    bytemuck::checked::cast_slice(&self.bytes)
                        .iter()
                        .map(|e: &i64| e.elem::<E>()),
                ),
                DType::U8 => Box::new(self.bytes.iter().map(|e| e.elem::<E>())),
                DType::U16 => Box::new(
                    bytemuck::checked::cast_slice(&self.bytes)
                        .iter()
                        .map(|e: &u16| e.elem::<E>()),
                ),
                DType::U32 => Box::new(
                    bytemuck::checked::cast_slice(&self.bytes)
                        .iter()
                        .map(|e: &u32| e.elem::<E>()),
                ),
                DType::U64 => Box::new(
                    bytemuck::checked::cast_slice(&self.bytes)
                        .iter()
                        .map(|e: &u64| e.elem::<E>()),
                ),
                DType::BF16 => Box::new(
                    bytemuck::checked::cast_slice(&self.bytes)
                        .iter()
                        .map(|e: &bf16| e.elem::<E>()),
                ),
                DType::F16 => Box::new(
                    bytemuck::checked::cast_slice(&self.bytes)
                        .iter()
                        .map(|e: &f16| e.elem::<E>()),
                ),
                DType::F32 | DType::Flex32 => Box::new(
                    bytemuck::checked::cast_slice(&self.bytes)
                        .iter()
                        .map(|e: &f32| e.elem::<E>()),
                ),
                DType::F64 => Box::new(
                    bytemuck::checked::cast_slice(&self.bytes)
                        .iter()
                        .map(|e: &f64| e.elem::<E>()),
                ),
                // bool is a byte value equal to either 0 or 1
                DType::Bool(BoolStore::Native) | DType::Bool(BoolStore::U8) => {
                    Box::new(self.bytes.iter().map(|e| e.elem::<E>()))
                }
                DType::Bool(BoolStore::U32) => Box::new(
                    bytemuck::checked::cast_slice(&self.bytes)
                        .iter()
                        .map(|e: &u32| e.elem::<E>()),
                ),
                DType::QFloat(scheme) => match scheme {
                    QuantScheme {
                        mode: QuantMode::Symmetric,
                        value:
                            QuantValue::Q8F
                            | QuantValue::Q8S
                            // Represent sub-byte values as i8
                            | QuantValue::Q4F
                            | QuantValue::Q4S
                            | QuantValue::Q2F
                            | QuantValue::Q2S,
                        ..
                    } => {
                        // Quantized int8 values
                        let q_bytes = QuantizedBytes {
                            bytes: self.bytes.clone(),
                            scheme,
                            num_elements: self.num_elements(),
                        };
                        let (values, _) = q_bytes.into_vec_i8();

                        Box::new(
                            values
                                .iter()
                                .map(|e: &i8| e.elem::<E>())
                                .collect::<Vec<_>>()
                                .into_iter(),
                        )
                    }
                    QuantScheme {
                        mode: QuantMode::Symmetric,
                        value:
                            QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1,
                        ..
                    } => {
                        unimplemented!("Not yet implemented for iteration");
                    }
                },
            }
        }
    }

    /// Converts the data to the dtype represented by `E`.
    ///
    /// # Panics
    ///
    /// Panics if storage access fails, the conversion isn't supported, or the stored
    /// representation or element count is invalid.
    #[track_caller]
    pub fn convert<E: Element>(self) -> Self {
        // TODO: deprecate?
        self.try_cast_as::<E>()
            .unwrap_or_else(|err| panic!("Failed to convert TensorData: {err}"))
    }

    /// Converts the data to `dtype`.
    ///
    /// # Panics
    ///
    /// Panics if storage access fails, the conversion isn't supported, or the stored
    /// representation or element count is invalid.
    #[track_caller]
    pub fn convert_dtype(self, dtype: DType) -> Self {
        // TODO: deprecate?
        self.try_cast(dtype)
            .unwrap_or_else(|err| panic!("Failed to convert TensorData to {dtype:?}: {err}"))
    }

    /// Converts the data to the dtype represented by `E`.
    ///
    /// By contract, this is equivalent to:
    /// `data.try_cast(E::dtype())`
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails, the conversion isn't supported, or the stored
    /// representation or element count is invalid.
    pub fn try_cast_as<E: Element>(self) -> Result<TensorData, DataError> {
        self.try_cast(E::dtype())
    }

    /// Converts the data to `dtype`.
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails, the conversion isn't supported, or the stored
    /// representation or element count is invalid.
    pub fn try_cast(self, dtype: DType) -> Result<TensorData, DataError> {
        if dtype == self.dtype {
            Ok(self)
        } else if dtype.size() == self.dtype.size()
            && !matches!(
                self.dtype,
                DType::Bool(BoolStore::Native) | DType::QFloat(_)
            )
            && !matches!(dtype, DType::Bool(BoolStore::Native) | DType::QFloat(_))
        {
            self.try_cast_inplace(dtype)
        } else {
            self.try_cast_clone(dtype)
        }
    }

    // Self-to-Self casts should be stripped before this point.
    fn try_cast_inplace(self, dtype: DType) -> Result<TensorData, DataError> {
        // Convert self.dtype to generic parameter:
        match self.dtype {
            DType::F64 => self.try_cast_inplace_from::<f64>(dtype),
            DType::F32 | DType::Flex32 => self.try_cast_inplace_from::<f32>(dtype),
            DType::F16 => self.try_cast_inplace_from::<f16>(dtype),
            DType::BF16 => self.try_cast_inplace_from::<bf16>(dtype),
            DType::I64 => self.try_cast_inplace_from::<i64>(dtype),
            DType::I32 => self.try_cast_inplace_from::<i32>(dtype),
            DType::I16 => self.try_cast_inplace_from::<i16>(dtype),
            DType::I8 => self.try_cast_inplace_from::<i8>(dtype),
            DType::U64 => self.try_cast_inplace_from::<u64>(dtype),
            DType::U32 => self.try_cast_inplace_from::<u32>(dtype),
            DType::U16 => self.try_cast_inplace_from::<u16>(dtype),
            DType::U8 => self.try_cast_inplace_from::<u8>(dtype),
            DType::Bool(BoolStore::U8) => self.try_cast_inplace_from::<u8>(dtype),
            DType::Bool(BoolStore::U32) => self.try_cast_inplace_from::<u32>(dtype),
            DType::Bool(BoolStore::Native) | DType::QFloat(_) => Err(DataError::DTypeMismatch {
                expected: dtype,
                actual: self.dtype,
            }),
        }
    }

    fn try_cast_inplace_from<Current>(self, dtype: DType) -> Result<TensorData, DataError>
    where
        Current: Element + AnyBitPattern,
    {
        // Convert target dtype to generic parameter.
        match dtype {
            DType::F64 => self.try_convert_inplace::<Current, f64>(),
            DType::F32 | DType::Flex32 => self.try_convert_inplace::<Current, f32>(),
            DType::F16 => self.try_convert_inplace::<Current, f16>(),
            DType::BF16 => self.try_convert_inplace::<Current, bf16>(),
            DType::I64 => self.try_convert_inplace::<Current, i64>(),
            DType::I32 => self.try_convert_inplace::<Current, i32>(),
            DType::I16 => self.try_convert_inplace::<Current, i16>(),
            DType::I8 => self.try_convert_inplace::<Current, i8>(),
            DType::U64 => self.try_convert_inplace::<Current, u64>(),
            DType::U32 => self.try_convert_inplace::<Current, u32>(),
            DType::U16 => self.try_convert_inplace::<Current, u16>(),
            DType::U8 => self.try_convert_inplace::<Current, u8>(),
            DType::Bool(BoolStore::U8) => self
                .try_convert_inplace_bool::<Current, u8>()
                .map(TensorData::into_bool_u8),
            DType::Bool(BoolStore::U32) => self
                .try_convert_inplace_bool::<Current, u32>()
                .map(TensorData::into_bool_u32),
            DType::Bool(BoolStore::Native) | DType::QFloat(_) => Err(DataError::DTypeMismatch {
                expected: dtype,
                actual: Current::dtype(),
            }),
        }
    }

    fn try_convert_inplace<Current, Target>(self) -> Result<TensorData, DataError>
    where
        Current: Element + AnyBitPattern,
        Target: Element + AnyBitPattern,
    {
        self.try_convert_inplace_with::<Current, Target>(|x| x.elem())
    }

    fn try_convert_inplace_bool<Current, Target>(self) -> Result<TensorData, DataError>
    where
        Current: Element + AnyBitPattern,
        Target: Element + AnyBitPattern,
    {
        self.try_convert_inplace_with::<Current, Target>(|x| x.to_bool().elem())
    }

    fn try_convert_inplace_with<Current, Target>(
        mut self,
        transform: impl Fn(&Current) -> Target,
    ) -> Result<TensorData, DataError>
    where
        Current: Element + AnyBitPattern,
        Target: Element + AnyBitPattern,
    {
        let expected = self.num_elements();
        let values =
            bytemuck::checked::try_cast_slice_mut::<_, Current>(self.bytes.write(Writer::new())?)
                .map_err(DataError::InvalidRepresentation)?;
        let actual = values.len();
        if actual != expected {
            return Err(DataError::ElementCountMismatch { expected, actual });
        }

        for x in values {
            let t = transform(x);
            let x = cast_mut::<_, Target>(x);
            *x = t;
        }

        self.dtype = Target::dtype();

        Ok(self)
    }

    fn try_cast_clone(self, dtype: DType) -> Result<TensorData, DataError> {
        // Convert self.dtype to generic parameter:
        match self.dtype {
            DType::F64 => self.try_cast_clone_from::<f64>(dtype),
            DType::F32 | DType::Flex32 => self.try_cast_clone_from::<f32>(dtype),
            DType::F16 => self.try_cast_clone_from::<f16>(dtype),
            DType::BF16 => self.try_cast_clone_from::<bf16>(dtype),
            DType::I64 => self.try_cast_clone_from::<i64>(dtype),
            DType::I32 => self.try_cast_clone_from::<i32>(dtype),
            DType::I16 => self.try_cast_clone_from::<i16>(dtype),
            DType::I8 => self.try_cast_clone_from::<i8>(dtype),
            DType::U64 => self.try_cast_clone_from::<u64>(dtype),
            DType::U32 => self.try_cast_clone_from::<u32>(dtype),
            DType::U16 => self.try_cast_clone_from::<u16>(dtype),
            DType::U8 => self.try_cast_clone_from::<u8>(dtype),
            DType::Bool(BoolStore::Native) => self.try_cast_clone_from::<bool>(dtype),
            DType::Bool(BoolStore::U8) => self.try_cast_clone_from::<u8>(dtype),
            DType::Bool(BoolStore::U32) => self.try_cast_clone_from::<u32>(dtype),
            DType::QFloat(_) => Err(DataError::UnsupportedConversion {
                to: dtype,
                from: self.dtype,
            }),
        }
    }

    fn try_cast_clone_from<Current>(self, dtype: DType) -> Result<TensorData, DataError>
    where
        Current: Element + CheckedBitPattern,
    {
        // Convert target dtype to generic parameter.
        match dtype {
            DType::F64 => self.try_convert_clone::<Current, f64>(),
            DType::F32 | DType::Flex32 => self.try_convert_clone::<Current, f32>(),
            DType::F16 => self.try_convert_clone::<Current, f16>(),
            DType::BF16 => self.try_convert_clone::<Current, bf16>(),
            DType::I64 => self.try_convert_clone::<Current, i64>(),
            DType::I32 => self.try_convert_clone::<Current, i32>(),
            DType::I16 => self.try_convert_clone::<Current, i16>(),
            DType::I8 => self.try_convert_clone::<Current, i8>(),
            DType::U64 => self.try_convert_clone::<Current, u64>(),
            DType::U32 => self.try_convert_clone::<Current, u32>(),
            DType::U16 => self.try_convert_clone::<Current, u16>(),
            DType::U8 => self.try_convert_clone::<Current, u8>(),
            DType::Bool(BoolStore::Native) => self.try_convert_clone::<Current, bool>(),
            DType::Bool(BoolStore::U8) => self
                .try_convert_clone_bool::<Current, u8>()
                .map(TensorData::into_bool_u8),
            DType::Bool(BoolStore::U32) => self
                .try_convert_clone_bool::<Current, u32>()
                .map(TensorData::into_bool_u32),
            DType::QFloat(_) => Err(DataError::UnsupportedConversion {
                to: dtype,
                from: self.dtype,
            }),
        }
    }

    fn try_convert_clone<Current, Target>(self) -> Result<TensorData, DataError>
    where
        Current: Element + CheckedBitPattern,
        Target: Element + Zeroable,
    {
        self.try_convert_clone_with::<Current, Target>(|x| x.elem())
    }

    fn try_convert_clone_bool<Current, Target>(self) -> Result<TensorData, DataError>
    where
        Current: Element + CheckedBitPattern,
        Target: Element + Zeroable,
    {
        self.try_convert_clone_with::<Current, Target>(|x| x.to_bool().elem())
    }

    fn try_convert_clone_with<Current, Target>(
        self,
        transform: impl Fn(&Current) -> Target,
    ) -> Result<TensorData, DataError>
    where
        Current: Element + CheckedBitPattern,
        Target: Element + Zeroable,
    {
        let expected = self.num_elements();
        let values =
            bytemuck::checked::try_cast_slice::<_, Current>(self.bytes.read(Reader::new())?)
                .map_err(DataError::InvalidRepresentation)?;
        let actual = values.len();
        if actual != expected {
            return Err(DataError::ElementCountMismatch { expected, actual });
        }
        let mut out: Vec<Target> = vec![Zeroable::zeroed(); expected];

        for (value, out) in values.iter().zip(&mut out) {
            *out = transform(value);
        }

        Ok(TensorData::new(out, self.shape))
    }
}
