use core::f32;

use crate::indexing::AsIndex;
use alloc::boxed::Box;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use bytemuck::{AnyBitPattern, CheckedBitPattern, Zeroable, cast_mut, checked::CheckedCastError};
use core::ops::{Index, IndexMut};
use rand::Rng;
use thiserror::Error;

use crate::Scalar;
use crate::distribution::Distribution;
use crate::element::{Element, ElementConversion};
use crate::tensor::DType;
use crate::tensor::ravel_index;
use crate::{
    AccessError, BoolStore, Bytes, ExecutionError, QuantLevel, QuantMode, QuantScheme, QuantValue,
    QuantizedBytes, Reader, Shape, Writer, bf16, f16,
};

use serde::{Deserialize, Serialize};

/// Errors that can occur while accessing or converting [`TensorData`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DataError {
    /// Host access to the underlying storage failed.
    #[error("Failed to access TensorData storage: {0}")]
    StorageAccess(#[from] AccessError),

    /// The stored bytes aren't a valid representation of the requested element type.
    #[error("TensorData storage is invalid for the requested element type: {0}")]
    InvalidRepresentation(CheckedCastError),

    /// The stored dtype doesn't match the requested dtype.
    #[error("Expected data type {expected:?}, but got {actual:?}")]
    DTypeMismatch {
        /// The expected storage DType.
        expected: DType,

        /// The actual storage DType.
        actual: DType,
    },

    /// Unsupported data conversion.
    #[error("Unsupported data conversion from {from:?} to {to:?}")]
    UnsupportedConversion {
        /// The source DType.
        from: DType,

        /// The destination DType.
        to: DType,
    },

    /// The byte storage doesn't match the number of elements described by the shape.
    #[error("TensorData shape describes {expected} element(s), but storage contains {actual}")]
    ElementCountMismatch {
        /// The number of elements described by the shape.
        expected: usize,

        /// The number of elements present in storage.
        actual: usize,
    },
}

/// Errors that can occur while reading host data from a tensor.
#[derive(Debug, Error)]
pub enum TensorReadError {
    /// Tensor execution failed while reading the data from the device.
    #[error(transparent)]
    Execution(#[from] ExecutionError),

    /// The resulting [`TensorData`] could not satisfy the requested data operation.
    #[error(transparent)]
    Data(#[from] DataError),

    /// The tensor shape does not satisfy the read operation's element-count requirement.
    #[error("Expected {expected} tensor element(s), but got {actual}")]
    InvalidShape {
        /// The required number of elements.
        expected: usize,

        /// The actual number of elements.
        actual: usize,
    },
}

impl DataError {
    /// Creates a [`DataError::DTypeMismatch`] for the requested element type `E`.
    fn dtype_mismatch_as<E: Element>(actual: DType) -> Self {
        Self::DTypeMismatch {
            expected: E::dtype(),
            actual,
        }
    }
}

/// Data structure for tensors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorData {
    /// The values of the tensor (as bytes).
    pub bytes: Bytes,

    /// The shape of the tensor.
    #[serde(with = "shape_inner")]
    pub shape: Shape,

    /// The data type of the tensor.
    pub dtype: DType,
}

// For backward compatibility with shape `Vec<usize>`
mod shape_inner {
    use crate::SmallVec;

    use super::*;

    pub fn serialize<S: serde::Serializer>(
        shape: &Shape,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        shape.as_slice().serialize(serializer)
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Shape, D::Error> {
        let dims = SmallVec::<[usize; _]>::deserialize(deserializer)?;
        Ok(Shape::new_raw(dims))
    }
}

impl TensorData {
    /// Creates a new tensor data structure.
    pub fn new<E: Element, S: Into<Shape>>(value: Vec<E>, shape: S) -> Self {
        // Ensure shape is valid
        let shape = shape.into();
        Self::check_data_len(&value, &shape);

        Self {
            bytes: Bytes::from_elems(value),
            shape,
            dtype: E::dtype(),
        }
    }

    /// Creates a new quantized tensor data structure.
    pub fn quantized<E: Element, S: Into<Shape>>(
        value: Vec<E>,
        shape: S,
        scheme: QuantScheme,
        qparams: &[f32],
    ) -> Self {
        let shape = shape.into();
        Self::check_data_len(&value, &shape);

        let q_bytes = QuantizedBytes::new(value, scheme, qparams);

        Self {
            bytes: q_bytes.bytes,
            shape,
            dtype: DType::QFloat(q_bytes.scheme),
        }
    }

    /// Creates a new tensor data structure from raw bytes.
    pub fn from_bytes<S: Into<Shape>>(bytes: Bytes, shape: S, dtype: DType) -> Self {
        Self {
            bytes,
            shape: shape.into(),
            dtype,
        }
    }

    /// Creates a new tensor data structure from raw bytes stored in a vector.
    ///
    /// Prefer [`TensorData::new`] or [`TensorData::quantized`] over this method unless you are
    /// certain that the byte representation is valid.
    pub fn from_bytes_vec<S: Into<Shape>>(bytes: Vec<u8>, shape: S, dtype: DType) -> Self {
        Self {
            bytes: Bytes::from_bytes_vec(bytes),
            shape: shape.into(),
            dtype,
        }
    }

    // Check that the input vector contains a correct number of elements
    fn check_data_len<E: Element>(data: &[E], shape: &Shape) {
        let expected_data_len = numel(shape);
        let num_data = data.len();
        assert_eq!(
            expected_data_len, num_data,
            "Shape {shape:?} is invalid for input of size {num_data:?}",
        );
    }

    /// Returns an [`Index`] view wrapper of the [`TensorData`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_std::*;
    ///
    /// let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let view: TensorDataView<f64> = data.try_view().unwrap();
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails or the dtype, byte representation, or element
    /// count is incompatible with the requested view.
    pub fn try_view<E: Element>(&self) -> Result<TensorDataView<'_, E>, DataError> {
        TensorDataView::<E>::try_view(self)
    }

    /// Returns a [`TensorDataView<E>`] of the [`TensorData`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_std::*;
    ///
    /// let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let view: TensorDataView<f64> = data.view();
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    /// ```
    ///
    /// # Returns
    /// The view.
    ///
    /// # Panics
    ///
    /// Panics if the view can't be created because storage access fails or the dtype, byte
    /// representation, or element count is incompatible with `E`.
    #[track_caller]
    pub fn view<E: Element>(&self) -> TensorDataView<'_, E> {
        self.try_view()
            .unwrap_or_else(|err| panic!("Failed to create TensorData view: {err}"))
    }

    /// Returns a [`TensorDataViewMut<E>`] of the [`TensorData`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_std::*;
    ///
    /// let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let mut view: TensorDataViewMut<f64> = data.try_mut_view().unwrap();
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    ///
    /// view[&[0, 0]] = 10.0;
    /// assert_eq!(view[&[0, 0]], 10.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails or the dtype, byte representation, or element
    /// count is incompatible with the requested view.
    pub fn try_mut_view<E: Element>(&mut self) -> Result<TensorDataViewMut<'_, E>, DataError> {
        TensorDataViewMut::<E>::try_mut_view(self)
    }

    /// Returns a [`TensorDataViewMut<E>`] of the [`TensorData`].
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_std::*;
    ///
    /// let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let mut view: TensorDataViewMut<f64> = data.mut_view();
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    ///
    /// view[&[0, 0]] = 10.0;
    /// assert_eq!(view[&[0, 0]], 10.0);
    /// ```
    ///
    /// # Returns
    /// The mut view.
    ///
    /// # Panics
    ///
    /// Panics if the view can't be created because storage access fails or the dtype, byte
    /// representation, or element count is incompatible with `E`.
    #[track_caller]
    pub fn mut_view<E: Element>(&mut self) -> TensorDataViewMut<'_, E> {
        self.try_mut_view()
            .unwrap_or_else(|err| panic!("Failed to create mutable TensorData view: {err}"))
    }

    /// Returns the immutable slice view of the tensor data.
    ///
    /// This materializes lazy storage into host-accessible memory when necessary.
    ///
    /// # Errors
    ///
    /// Returns an error if host access fails, the target element type doesn't match the stored
    /// type, or the stored byte representation is invalid for `E`.
    pub fn as_slice<E: Element>(&self) -> Result<&[E], DataError> {
        if self.matches_target_dtype::<E>() {
            bytemuck::checked::try_cast_slice(self.bytes.read(Reader::new())?)
                .map_err(DataError::InvalidRepresentation)
        } else {
            Err(DataError::dtype_mismatch_as::<E>(self.dtype))
        }
    }

    /// Returns the mutable slice view of the tensor data.
    ///
    /// This materializes lazy storage and performs copy-on-write when necessary.
    ///
    /// # Errors
    ///
    /// Returns an error if host access fails, the target element type doesn't match the stored
    /// type, or the stored byte representation is invalid for `E`.
    pub fn as_mut_slice<E: Element>(&mut self) -> Result<&mut [E], DataError> {
        if self.matches_target_dtype::<E>() {
            bytemuck::checked::try_cast_slice_mut(self.bytes.write(Writer::new())?)
                .map_err(DataError::InvalidRepresentation)
        } else {
            Err(DataError::dtype_mismatch_as::<E>(self.dtype))
        }
    }

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

    fn matches_target_dtype<E: Element>(&self) -> bool {
        let target_dtype = E::dtype();
        match self.dtype {
            DType::Bool(BoolStore::U8) => {
                matches!(target_dtype, DType::U8 | DType::Bool(BoolStore::U8))
            }
            DType::Bool(BoolStore::U32) => {
                matches!(target_dtype, DType::U32 | DType::Bool(BoolStore::U32))
            }
            dtype => dtype == target_dtype,
        }
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
                        level: QuantLevel::Tensor | QuantLevel::Block(_),
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
                        level: QuantLevel::Tensor | QuantLevel::Block(_),
                        mode: QuantMode::Symmetric,
                        value:
                            QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1,
                        ..
                    } => {
                        unimplemented!("Not yet implemented for iteration");
                    }
                    QuantScheme {
                        level: QuantLevel::BlockTensor { .. },
                        ..
                    } => {
                        unimplemented!("two-level quantization is not supported yet")
                    }
                },
            }
        }
    }

    /// Returns the rank (the number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of elements of the tensor data.
    pub fn num_elements(&self) -> usize {
        numel(&self.shape)
    }

    /// Populates the data with random values.
    pub fn random<E: Element, R: Rng, S: Into<Shape>>(
        shape: S,
        distribution: Distribution,
        rng: &mut R,
    ) -> Self {
        let shape = shape.into();
        let num_elements = numel(&shape);
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(E::random(distribution, rng));
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with zeros.
    pub fn zeros<E: Element, S: Into<Shape>>(shape: S) -> TensorData {
        let shape = shape.into();
        let num_elements = numel(&shape);
        let mut data = Vec::<E>::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(0.elem());
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with ones.
    pub fn ones<E: Element, S: Into<Shape>>(shape: S) -> TensorData {
        let shape = shape.into();
        let num_elements = numel(&shape);
        let mut data = Vec::<E>::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(1.elem());
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with the given value
    pub fn full<E: Element, S: Into<Shape>>(shape: S, fill_value: E) -> TensorData {
        let shape = shape.into();
        let num_elements = numel(&shape);
        let mut data = Vec::<E>::with_capacity(num_elements);
        for _ in 0..num_elements {
            data.push(fill_value)
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with the given value
    pub fn full_dtype<E: Into<Scalar>, S: Into<Shape>>(
        shape: S,
        fill_value: E,
        dtype: DType,
    ) -> TensorData {
        let fill_value = fill_value.into();
        match dtype {
            DType::F64 => Self::full::<f64, _>(shape, fill_value.elem()),
            DType::F32 | DType::Flex32 => Self::full::<f32, _>(shape, fill_value.elem()),
            DType::F16 => Self::full::<f16, _>(shape, fill_value.elem()),
            DType::BF16 => Self::full::<bf16, _>(shape, fill_value.elem()),
            DType::I64 => Self::full::<i64, _>(shape, fill_value.elem()),
            DType::I32 => Self::full::<i32, _>(shape, fill_value.elem()),
            DType::I16 => Self::full::<i16, _>(shape, fill_value.elem()),
            DType::I8 => Self::full::<i8, _>(shape, fill_value.elem()),
            DType::U64 => Self::full::<u64, _>(shape, fill_value.elem()),
            DType::U32 => Self::full::<u32, _>(shape, fill_value.elem()),
            DType::U16 => Self::full::<u16, _>(shape, fill_value.elem()),
            DType::U8 => Self::full::<u8, _>(shape, fill_value.elem()),
            DType::Bool(BoolStore::Native) => Self::full::<bool, _>(shape, fill_value.elem()),
            DType::Bool(BoolStore::U8) => {
                Self::full::<u8, _>(shape, fill_value.elem()).into_bool_u8()
            }
            DType::Bool(BoolStore::U32) => {
                Self::full::<u32, _>(shape, fill_value.elem()).into_bool_u32()
            }
            DType::QFloat(_) => unreachable!(),
        }
    }

    // Unchecked, used to overwrite the dtype
    fn into_bool_u8(mut self) -> Self {
        self.dtype = DType::Bool(BoolStore::U8);
        self
    }

    // Unchecked, used to overwrite the dtype
    fn into_bool_u32(mut self) -> Self {
        self.dtype = DType::Bool(BoolStore::U32);
        self
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

    /// Returns the data as a slice of bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns the bytes representation of the data.
    pub fn into_bytes(self) -> Bytes {
        self.bytes
    }
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

impl<E: Element, const A: usize> From<[E; A]> for TensorData {
    fn from(elems: [E; A]) -> Self {
        TensorData::new(elems.to_vec(), [A])
    }
}

impl<const A: usize> From<[usize; A]> for TensorData {
    fn from(elems: [usize; A]) -> Self {
        TensorData::new(elems.iter().map(|&e| e as i64).collect(), [A])
    }
}

impl From<&[usize]> for TensorData {
    fn from(elems: &[usize]) -> Self {
        let mut data = Vec::with_capacity(elems.len());
        for elem in elems.iter() {
            data.push(*elem as i64);
        }

        TensorData::new(data, [elems.len()])
    }
}

impl<E: Element> From<&[E]> for TensorData {
    fn from(elems: &[E]) -> Self {
        let mut data = Vec::with_capacity(elems.len());
        for elem in elems.iter() {
            data.push(*elem);
        }

        TensorData::new(data, [elems.len()])
    }
}

impl<E: Element, const A: usize, const B: usize> From<[[E; B]; A]> for TensorData {
    fn from(elems: [[E; B]; A]) -> Self {
        let mut data = Vec::with_capacity(A * B);
        for elem in elems.into_iter().take(A) {
            for elem in elem.into_iter().take(B) {
                data.push(elem);
            }
        }

        TensorData::new(data, [A, B])
    }
}

impl<E: Element, const A: usize, const B: usize, const C: usize> From<[[[E; C]; B]; A]>
    for TensorData
{
    fn from(elems: [[[E; C]; B]; A]) -> Self {
        let mut data = Vec::with_capacity(A * B * C);

        for elem in elems.into_iter().take(A) {
            for elem in elem.into_iter().take(B) {
                for elem in elem.into_iter().take(C) {
                    data.push(elem);
                }
            }
        }

        TensorData::new(data, [A, B, C])
    }
}

impl<E: Element, const A: usize, const B: usize, const C: usize, const D: usize>
    From<[[[[E; D]; C]; B]; A]> for TensorData
{
    fn from(elems: [[[[E; D]; C]; B]; A]) -> Self {
        let mut data = Vec::with_capacity(A * B * C * D);

        for elem in elems.into_iter().take(A) {
            for elem in elem.into_iter().take(B) {
                for elem in elem.into_iter().take(C) {
                    for elem in elem.into_iter().take(D) {
                        data.push(elem);
                    }
                }
            }
        }

        TensorData::new(data, [A, B, C, D])
    }
}

impl<Elem: Element, const A: usize, const B: usize, const C: usize, const D: usize, const E: usize>
    From<[[[[[Elem; E]; D]; C]; B]; A]> for TensorData
{
    fn from(elems: [[[[[Elem; E]; D]; C]; B]; A]) -> Self {
        let mut data = Vec::with_capacity(A * B * C * D * E);

        for elem in elems.into_iter().take(A) {
            for elem in elem.into_iter().take(B) {
                for elem in elem.into_iter().take(C) {
                    for elem in elem.into_iter().take(D) {
                        for elem in elem.into_iter().take(E) {
                            data.push(elem);
                        }
                    }
                }
            }
        }

        TensorData::new(data, [A, B, C, D, E])
    }
}
impl core::fmt::Display for TensorData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let fmt = match self.dtype {
            DType::F64 => format!("{:?}", self.as_slice::<f64>().unwrap()),
            DType::F32 | DType::Flex32 => format!("{:?}", self.as_slice::<f32>().unwrap()),
            DType::F16 => format!("{:?}", self.as_slice::<f16>().unwrap()),
            DType::BF16 => format!("{:?}", self.as_slice::<bf16>().unwrap()),
            DType::I64 => format!("{:?}", self.as_slice::<i64>().unwrap()),
            DType::I32 => format!("{:?}", self.as_slice::<i32>().unwrap()),
            DType::I16 => format!("{:?}", self.as_slice::<i16>().unwrap()),
            DType::I8 => format!("{:?}", self.as_slice::<i8>().unwrap()),
            DType::U64 => format!("{:?}", self.as_slice::<u64>().unwrap()),
            DType::U32 => format!("{:?}", self.as_slice::<u32>().unwrap()),
            DType::U16 => format!("{:?}", self.as_slice::<u16>().unwrap()),
            DType::U8 => format!("{:?}", self.as_slice::<u8>().unwrap()),
            DType::Bool(BoolStore::Native) => format!("{:?}", self.as_slice::<bool>().unwrap()),
            DType::Bool(BoolStore::U8) => format!("{:?}", self.as_slice::<u8>().unwrap()),
            DType::Bool(BoolStore::U32) => format!("{:?}", self.as_slice::<u32>().unwrap()),
            DType::QFloat(scheme) => match scheme {
                QuantScheme {
                    level: QuantLevel::Tensor | QuantLevel::Block(_),
                    mode: QuantMode::Symmetric,
                    value:
                        QuantValue::Q8F
                        | QuantValue::Q8S
                        // Display sub-byte values as i8
                        | QuantValue::Q4F
                        | QuantValue::Q4S
                        | QuantValue::Q2F
                        | QuantValue::Q2S,
                    ..
                } => {
                    format!("{:?} {scheme:?}", self.iter::<i8>().collect::<Vec<_>>())
                },
                QuantScheme {
                        level: QuantLevel::Tensor | QuantLevel::Block(_),
                        mode: QuantMode::Symmetric,
                        value:
                            QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1,
                        ..
                    } => {
                        unimplemented!("Can't format yet");
                    }
                QuantScheme {
                    level: QuantLevel::BlockTensor { .. },
                    ..
                } => {
                    unimplemented!("two-level quantization is not supported yet")
                }
            },
        };
        f.write_str(fmt.as_str())
    }
}

/// Typed [`Index`] view over a [`TensorData`].
///
/// Creating a view materializes lazy storage into host-accessible memory when necessary. It does
/// not perform dtype conversion.
///
/// # Example
/// ```rust,no_run
/// use burn_std::*;
///
/// let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
/// let view: TensorDataView<f64> = data.view();
///
/// assert_eq!(view.shape(), &data.shape);
/// assert_eq!(&view.dtype(), &data.dtype);
///
/// assert_eq!(view[&[0, 0]], 1.0);
/// assert_eq!(view[&[0, 1]], 2.0);
/// assert_eq!(view[&[1, 0]], 3.0);
/// assert_eq!(view[&[1, 1]], 4.0);
/// ```
#[derive(Debug)]
pub struct TensorDataView<'a, E: Element> {
    values: &'a [E],
    shape: &'a Shape,
    dtype: DType,
}

impl<'a, E: Element> TensorDataView<'a, E> {
    /// Creates a typed indexed view over `data`.
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_std::*;
    ///
    /// let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let view: TensorDataView<f64> = data.try_view().unwrap();
    ///
    /// assert_eq!(view.shape(), &data.shape);
    /// assert_eq!(&view.dtype(), &data.dtype);
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails or the dtype, byte representation, or element
    /// count is incompatible with `E`.
    pub fn try_view(data: &'a TensorData) -> Result<TensorDataView<'a, E>, DataError> {
        let shape = &data.shape;
        let dtype = data.dtype;
        let expected = shape.num_elements();
        let values = data.as_slice::<E>()?;
        let actual = values.len();

        if actual != expected {
            return Err(DataError::ElementCountMismatch { expected, actual });
        }

        Ok(TensorDataView {
            values,
            shape,
            dtype,
        })
    }

    /// Returns the shape of the view.
    pub fn shape(&self) -> &Shape {
        self.shape
    }

    /// Returns the dtype of the view.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Ravels the index via [`ravel_index`] and the view's shape.
    pub fn ravel_index<I: AsIndex>(&self, index: &[I]) -> usize {
        ravel_index(index, self.shape)
    }
}

impl<'a, I: AsIndex, E: Element> Index<&[I]> for TensorDataView<'a, E> {
    type Output = E;

    fn index(&self, index: &[I]) -> &Self::Output {
        let o = self.ravel_index(index);
        &self.values[o]
    }
}

/// Typed mutable [`IndexMut`] view over a [`TensorData`].
///
/// Creating a mutable view materializes lazy storage and performs copy-on-write when necessary.
/// It does not perform dtype conversion.
///
/// # Example
/// ```rust,no_run
/// use burn_std::*;
///
/// let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
/// let shape = data.shape.clone();
/// let dtype = data.dtype;
/// let mut view: TensorDataViewMut<f64> = data.mut_view();
///
/// assert_eq!(view.shape(), &shape);
/// assert_eq!(&view.dtype(), &dtype);
///
/// assert_eq!(view[&[0, 0]], 1.0);
/// assert_eq!(view[&[0, 1]], 2.0);
/// assert_eq!(view[&[1, 0]], 3.0);
/// assert_eq!(view[&[1, 1]], 4.0);
///
/// view[&[0, 0]] = 10.0;
/// assert_eq!(view[&[0, 0]], 10.0);
/// ```
#[derive(Debug)]
pub struct TensorDataViewMut<'a, E: Element> {
    values: &'a mut [E],
    // `as_mut_slice` borrows the entire `TensorData`, so the view can't also retain a reference to
    // its shape. Keep an owned copy until storage and metadata can be borrowed as disjoint fields.
    shape: Shape,
    dtype: DType,
}

impl<'a, E: Element> TensorDataViewMut<'a, E> {
    /// Creates a typed mutable indexed view over `data`.
    ///
    /// # Example
    /// ```rust,no_run
    /// use burn_std::*;
    ///
    /// let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
    /// let shape = data.shape.clone();
    /// let dtype = data.dtype;
    ///
    /// let mut view: TensorDataViewMut<f64> =
    ///     TensorDataViewMut::try_mut_view(&mut data).unwrap();
    ///
    /// assert_eq!(view.shape(), &shape);
    /// assert_eq!(&view.dtype(), &dtype);
    ///
    /// assert_eq!(view[&[0, 0]], 1.0);
    /// assert_eq!(view[&[0, 1]], 2.0);
    /// assert_eq!(view[&[1, 0]], 3.0);
    /// assert_eq!(view[&[1, 1]], 4.0);
    ///
    /// view[&[0, 0]] = 10.0;
    /// assert_eq!(view[&[0, 0]], 10.0);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if storage access fails or the dtype, byte representation, or element
    /// count is incompatible with `E`.
    pub fn try_mut_view(data: &'a mut TensorData) -> Result<TensorDataViewMut<'a, E>, DataError> {
        let shape = data.shape.clone();
        let dtype = data.dtype;
        let expected = shape.num_elements();
        let values = data.as_mut_slice::<E>()?;
        let actual = values.len();

        if actual != expected {
            return Err(DataError::ElementCountMismatch { expected, actual });
        }

        Ok(TensorDataViewMut {
            values,
            shape,
            dtype,
        })
    }

    /// Returns the shape of the view.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the dtype of the view.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Ravels the dims via [`ravel_index`] and the view's shape.
    pub fn ravel_index<I: AsIndex>(&self, index: &[I]) -> usize {
        ravel_index(index, &self.shape)
    }
}

impl<'a, I, E> Index<&[I]> for TensorDataViewMut<'a, E>
where
    I: AsIndex,
    E: Element,
{
    type Output = E;

    fn index(&self, index: &[I]) -> &Self::Output {
        let o = self.ravel_index::<I>(index);
        &self.values[o]
    }
}

impl<'a, I, E> IndexMut<&[I]> for TensorDataViewMut<'a, E>
where
    I: AsIndex,
    E: Element,
{
    fn index_mut(&mut self, index: &[I]) -> &mut Self::Output {
        let o = self.ravel_index::<I>(index);
        &mut self.values[o]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AccessPolicy, AllocationController, AllocationProperty, shape};
    use alloc::vec;
    use core::mem::{MaybeUninit, align_of, size_of};
    use rand::{
        SeedableRng,
        rngs::{StdRng, SysRng},
    };

    #[test]
    fn should_have_rank() {
        let shape = [3, 5, 6];
        let data = TensorData::random::<f32, _, _>(
            shape,
            Distribution::Default,
            &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
        );

        assert_eq!(data.rank(), 3);
    }

    #[test]
    fn into_vec_should_yield_same_value_as_iter() {
        let shape = [3, 5, 6];
        let data = TensorData::random::<f32, _, _>(
            shape,
            Distribution::Default,
            &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
        );

        let expected = data.iter::<f32>().collect::<Vec<f32>>();
        let actual = data.try_into_vec::<f32>().unwrap();

        assert_eq!(expected, actual);
    }

    #[test]
    #[should_panic]
    fn into_vec_should_assert_wrong_dtype() {
        let shape = [3, 5, 6];
        let data = TensorData::random::<f32, _, _>(
            shape,
            Distribution::Default,
            &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
        );

        data.try_into_vec::<i32>().unwrap();
    }

    #[test]
    fn should_have_right_num_elements() {
        let shape = [3, 5, 6];
        let num_elements: usize = shape.iter().product();
        let data = TensorData::random::<f32, _, _>(
            shape,
            Distribution::Default,
            &mut StdRng::try_from_rng(&mut SysRng).unwrap(),
        );

        assert_eq!(num_elements, data.bytes.len() / 4); // f32 stored as u8s
        assert_eq!(num_elements, data.as_slice::<f32>().unwrap().len());
    }

    #[test]
    fn should_have_right_shape() {
        let data = TensorData::from([[3.0, 5.0, 6.0]]);
        assert_eq!(data.shape, shape![1, 3]);

        let data = TensorData::from([[4.0, 5.0, 8.0], [3.0, 5.0, 6.0]]);
        assert_eq!(data.shape, shape![2, 3]);

        let data = TensorData::from([3.0, 5.0, 6.0]);
        assert_eq!(data.shape, shape![3]);
    }

    #[test]
    fn should_convert_bytes_correctly() {
        let mut vector: Vec<f32> = Vec::with_capacity(5);
        vector.push(2.0);
        vector.push(3.0);
        let data1 = TensorData::new(vector, vec![2]);

        let factor = size_of::<f32>() / size_of::<u8>();
        assert_eq!(data1.bytes.len(), 2 * factor);
        assert_eq!(data1.bytes.capacity(), 5 * factor);
    }

    #[test]
    fn should_convert_bytes_correctly_inplace() {
        fn test_precision<E: Element>() {
            let data = TensorData::new((0..32).collect(), [32]);
            let self1 = data.clone().convert::<E>();
            for (i, val) in self1.try_into_vec::<E>().unwrap().into_iter().enumerate() {
                assert_eq!(i as u32, val.elem::<u32>())
            }
        }
        test_precision::<f32>();
        test_precision::<f16>();
        test_precision::<i64>();
        test_precision::<i32>();
    }

    #[test]
    fn should_convert_negative_values_to_bool_store() {
        for store in [BoolStore::U8, BoolStore::U32, BoolStore::Native] {
            let data = TensorData::from([-1i32, 0, 1, -12]).convert_dtype(DType::Bool(store));
            assert_eq!(data.dtype, DType::Bool(store));
            assert_eq!(
                data.iter::<bool>().collect::<Vec<_>>(),
                [true, false, true, true]
            );

            let data = TensorData::from([-1.5f32, 0.0, 0.5]).convert_dtype(DType::Bool(store));
            assert_eq!(data.iter::<bool>().collect::<Vec<_>>(), [true, false, true]);
        }
    }

    macro_rules! test_dtypes {
    ($test_name:ident, $($dtype:ty),*) => {
        $(
            paste::paste! {
                #[test]
                fn [<$test_name _ $dtype:snake>]() {
                    let full_dtype = TensorData::full_dtype([2, 16], 4, <$dtype>::dtype());
                    let full = TensorData::full::<$dtype, _>([2, 16], 4.elem());
                    assert_eq!(full_dtype, full);
                }
            }
        )*
    };
}

    test_dtypes!(
        should_create_with_dtype,
        bool,
        i8,
        i16,
        i32,
        i64,
        u8,
        u16,
        u32,
        u64,
        f16,
        bf16,
        f32,
        f64
    );

    #[test]
    fn should_serialize_deserialize_tensor_data() {
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        assert_eq!(
            data.as_bytes(),
            [
                0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192,
                64
            ]
        );
        let serialized = serde_json::to_string(&data).unwrap();
        let deserialized: TensorData = serde_json::from_str(&serialized).unwrap();
        assert_eq!(data, deserialized);
    }

    #[test]
    fn should_deserialize_tensor_data_with_shape_inner() {
        // TensorData `shape` was previously a Vec<usize>.
        let serialized = r#"{
        "bytes": [0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64],
        "shape": [2, 3],
        "dtype": "F32"
    }"#;

        let data: TensorData = serde_json::from_str(serialized).unwrap();
        assert_eq!(data.shape, shape![2, 3]);
        assert_eq!(
            data.as_slice::<f32>().unwrap(),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn should_serialize_shape_as_flat_array() {
        // Ensure the new Shape serializes identically to how Vec<usize> used to,
        // i.e. as a flat JSON array, not as an object like `{"dims": [2, 3]}`.
        let data = TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let serialized = serde_json::to_string(&data).unwrap();
        let json: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        assert_eq!(json["shape"], serde_json::json!([2, 3]));
    }

    #[test]
    fn test_tensor_data_try_view_dtype_mismatch() {
        let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let dtype = data.dtype;

        assert_eq!(
            data.try_view::<i32>().unwrap_err(),
            DataError::DTypeMismatch {
                expected: <i32 as Element>::dtype(),
                actual: dtype,
            }
        );
    }

    #[test]
    #[should_panic(expected = "Expected data type")]
    fn test_tensor_data_expect_view_dtype_mismatch() {
        let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let _view = data.view::<i32>();
    }

    #[test]
    fn test_tensor_data_try_mut_view_dtype_mismatch() {
        let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);

        let result = data.try_mut_view::<i32>();
        assert_eq!(
            result.unwrap_err(),
            DataError::DTypeMismatch {
                actual: data.dtype,
                expected: <i32 as Element>::dtype(),
            }
        );
    }

    #[test]
    fn try_view_validates_storage() {
        let invalid_representation = TensorData::from_bytes_vec(vec![0; 3], [1], DType::F32);
        assert!(matches!(
            invalid_representation.try_view::<f32>(),
            Err(DataError::InvalidRepresentation(_))
        ));

        let invalid_count = TensorData::from_bytes_vec(vec![0; 8], [1], DType::F32);
        assert!(matches!(
            invalid_count.try_view::<f32>(),
            Err(DataError::ElementCountMismatch {
                expected: 1,
                actual: 2
            })
        ));

        let invalid_bool = TensorData::from_bytes_vec(vec![2], [1], DType::Bool(BoolStore::Native));
        assert!(matches!(
            invalid_bool.try_view::<bool>(),
            Err(DataError::InvalidRepresentation(_))
        ));
    }

    #[test]
    fn try_view_propagates_materialization_failure() {
        #[derive(Debug)]
        struct FailingController;

        impl AllocationController for FailingController {
            fn alloc_align(&self) -> usize {
                align_of::<f32>()
            }

            fn property(&self) -> AllocationProperty {
                AllocationProperty::Other
            }

            fn capacity(&self) -> usize {
                size_of::<f32>()
            }

            fn memory(&self, _policy: AccessPolicy) -> Result<&[MaybeUninit<u8>], AccessError> {
                Err(AccessError::Read("test read failure".into()))
            }

            unsafe fn memory_mut(
                &mut self,
                _policy: AccessPolicy,
            ) -> Result<&mut [MaybeUninit<u8>], AccessError> {
                Err(AccessError::Read("test write failure".into()))
            }
        }

        // SAFETY: The controller never exposes its inaccessible storage as initialized memory.
        let bytes =
            unsafe { Bytes::from_controller(Box::new(FailingController), size_of::<f32>()) };
        let data = TensorData::from_bytes(bytes, [1], DType::F32);

        assert!(matches!(
            data.try_view::<f32>(),
            Err(DataError::StorageAccess(AccessError::Read(reason))) if reason == "test read failure"
        ));
    }

    #[test]
    fn try_mut_view_validates_storage() {
        let mut invalid_representation = TensorData::from_bytes_vec(vec![0; 3], [1], DType::F32);
        assert!(matches!(
            invalid_representation.try_mut_view::<f32>(),
            Err(DataError::InvalidRepresentation(_))
        ));
    }

    #[test]
    fn try_cast_propagates_invalid_storage() {
        let invalid_inplace = TensorData::from_bytes_vec(vec![0; 3], [1], DType::F32);
        assert!(matches!(
            invalid_inplace.try_cast(DType::I32),
            Err(DataError::InvalidRepresentation(_))
        ));

        let invalid_clone = TensorData::from_bytes_vec(vec![0; 3], [1], DType::F32);
        assert!(matches!(
            invalid_clone.try_cast(DType::F64),
            Err(DataError::InvalidRepresentation(_))
        ));

        let invalid_count = TensorData::from_bytes_vec(vec![0; 8], [1], DType::F32);
        assert!(matches!(
            invalid_count.try_cast(DType::F64),
            Err(DataError::ElementCountMismatch {
                expected: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn try_cast_reports_unsupported_quantized_conversion() {
        let scheme = QuantScheme::default();
        let target = DType::QFloat(scheme);

        assert_eq!(
            TensorData::from([1.0f32]).try_cast(target),
            Err(DataError::UnsupportedConversion {
                from: DType::F32,
                to: target,
            })
        );

        let quantized = TensorData::quantized(vec![0i8], [1], scheme, &[1.0]);
        assert_eq!(
            quantized.try_cast(DType::F32),
            Err(DataError::UnsupportedConversion {
                from: target,
                to: DType::F32,
            })
        );
    }

    #[test]
    #[should_panic(expected = "Expected data type")]
    fn test_tensor_data_expect_mut_view_dtype_mismatch() {
        let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let _view = data.mut_view::<i32>();
    }

    #[test]
    fn test_tensor_data_index_view() {
        let data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let view = data.view::<f64>();

        assert_eq!(view.shape(), &data.shape);

        assert_eq!(view[&[0, 0]], 1.0);
        assert_eq!(view[&[0, 1]], 2.0);
        assert_eq!(view[&[1, 0]], 3.0);
        assert_eq!(view[&[1, 1]], 4.0);
    }

    #[test]
    fn test_tensor_data_index_mut_view() {
        let mut data = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        let shape = data.shape.clone();

        let mut view = data.mut_view::<f64>();

        assert_eq!(view.shape(), &shape);

        assert_eq!(view[&[0, 0]], 1.0);
        assert_eq!(view[&[0, 1]], 2.0);
        assert_eq!(view[&[1, 0]], 3.0);
        assert_eq!(view[&[1, 1]], 4.0);

        view[&[0, 0]] = 10.0;
        assert_eq!(view[&[0, 0]], 10.0);
    }

    #[test]
    fn test_to_vec_as() {
        let data = TensorData::from([0.0f32, 1.0, 2.5]);

        // Same-dtype copy.
        assert_eq!(data.try_to_vec_as::<f32>().unwrap(), vec![0.0f32, 1.0, 2.5]);

        // Widening cast (different element size).
        assert_eq!(data.try_to_vec_as::<f64>().unwrap(), vec![0.0f64, 1.0, 2.5]);

        // Float to int cast (same element size) truncates.
        assert_eq!(data.try_to_vec_as::<i32>().unwrap(), vec![0i32, 1, 2]);

        // The source data is borrowed, not consumed.
        data.assert_eq(&TensorData::from([0.0f32, 1.0, 2.5]), true);
    }

    #[test]
    fn test_into_vec_as() {
        let data = TensorData::from([0i32, 1, 2, 3]);

        // Same-dtype conversion.
        assert_eq!(
            data.clone().try_into_vec_as::<i32>().unwrap(),
            vec![0i32, 1, 2, 3]
        );

        // Int to float cast.
        assert_eq!(
            data.clone().try_into_vec_as::<f32>().unwrap(),
            vec![0.0f32, 1.0, 2.0, 3.0]
        );

        // Narrowing int cast.
        assert_eq!(data.try_into_vec_as::<u8>().unwrap(), vec![0u8, 1, 2, 3]);
    }
}
