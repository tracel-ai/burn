use core::f32;

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use bytemuck::checked::CheckedCastError;
use rand::Rng;
use thiserror::Error;

use crate::Scalar;
use crate::distribution::Distribution;
use crate::element::{Element, ElementConversion};
use crate::tensor::DType;
use crate::{
    AccessError, BoolStore, Bytes, ExecutionError, QuantMode, QuantScheme, QuantValue,
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
    pub(super) fn dtype_mismatch_as<E: Element>(actual: DType) -> Self {
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
    /// Returns the rank (the number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns the total number of elements of the tensor data.
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

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
        global: Option<f32>,
    ) -> Self {
        let shape = shape.into();
        Self::check_data_len(&value, &shape);

        let q_bytes = QuantizedBytes::new(value, shape.clone(), scheme, qparams, global);

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
        let expected_data_len = shape.num_elements();
        let num_data = data.len();
        assert_eq!(
            expected_data_len, num_data,
            "Shape {shape:?} is invalid for input of size {num_data:?}",
        );
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

    pub(super) fn matches_target_dtype<E: Element>(&self) -> bool {
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

    /// Populates the data with random values.
    pub fn random<E: Element, R: Rng, S: Into<Shape>>(
        shape: S,
        distribution: Distribution,
        rng: &mut R,
    ) -> Self {
        let shape = shape.into();
        let data = (0..shape.num_elements())
            .map(|_| E::random(distribution, rng))
            .collect();

        Self::new(data, shape)
    }

    /// Populates the data with random values.
    pub fn try_random_dtype<R: Rng, S: Into<Shape>>(
        shape: S,
        distribution: Distribution,
        rng: &mut R,
        dtype: DType,
    ) -> Result<Self, DataError> {
        Ok(match dtype {
            DType::F64 => Self::random::<f64, _, _>(shape, distribution, rng),
            DType::F32 | DType::Flex32 => Self::random::<f32, _, _>(shape, distribution, rng),
            DType::F16 => Self::random::<f16, _, _>(shape, distribution, rng),
            DType::BF16 => Self::random::<bf16, _, _>(shape, distribution, rng),
            DType::I64 => Self::random::<i64, _, _>(shape, distribution, rng),
            DType::I32 => Self::random::<i32, _, _>(shape, distribution, rng),
            DType::I16 => Self::random::<i16, _, _>(shape, distribution, rng),
            DType::I8 => Self::random::<i8, _, _>(shape, distribution, rng),
            DType::U64 => Self::random::<u64, _, _>(shape, distribution, rng),
            DType::U32 => Self::random::<u32, _, _>(shape, distribution, rng),
            DType::U16 => Self::random::<u16, _, _>(shape, distribution, rng),
            DType::U8 => Self::random::<u8, _, _>(shape, distribution, rng),
            DType::Bool(BoolStore::Native) => Self::random::<bool, _, _>(shape, distribution, rng),
            DType::Bool(BoolStore::U8) => Self::random::<u8, _, _>(shape, distribution, rng)
                .unchecked_cast(DType::Bool(BoolStore::U8)),
            DType::Bool(BoolStore::U32) => Self::random::<u32, _, _>(shape, distribution, rng)
                .unchecked_cast(DType::Bool(BoolStore::U32)),
            DType::QFloat(_) => {
                return Err(DataError::UnsupportedConversion {
                    from: DType::F64,
                    to: dtype,
                });
            }
        })
    }

    /// Populates the data with zeros.
    pub fn zeros<E: Element, S: Into<Shape>>(shape: S) -> Self {
        Self::full(shape, 0.elem::<E>())
    }

    /// Populate the data with zeros of the target dtype.
    ///
    /// # Returns
    ///
    /// Returns a [`TensorData`] populated with zeros of the target dtype;
    /// or an error if the data cannot be populated with zeros of the target dtype.
    pub fn try_zeros_dtype<S: Into<Shape>>(shape: S, dtype: DType) -> Result<Self, DataError> {
        Self::try_full_dtype(shape, 0, dtype)
    }

    /// Populate the data with zeros of the target dtype.
    ///
    /// # Panics
    ///
    /// Panics if the data cannot be populated with zeros of the target dtype.
    pub fn zeros_dtype<S: Into<Shape>>(shape: S, dtype: DType) -> Self {
        Self::try_zeros_dtype(shape, dtype)
            .unwrap_or_else(|err| panic!("Failed to create tensor data: {}", err))
    }

    /// Populates the data with ones.
    pub fn ones<E: Element, S: Into<Shape>>(shape: S) -> Self {
        Self::full(shape, 1.elem::<E>())
    }

    /// Populate the data with ones of the target dtype.
    ///
    /// # Returns
    ///
    /// Returns a [`TensorData`] populated with ones of the target dtype;
    /// or an error if the data cannot be populated with ones of the target dtype.
    pub fn try_ones_dtype<S: Into<Shape>>(shape: S, dtype: DType) -> Result<Self, DataError> {
        Self::try_full_dtype(shape, 1, dtype)
    }

    /// Populate the data with ones of the target dtype.
    ///
    /// # Panics
    ///
    /// Panics if the data cannot be populated with ones of the target dtype.
    pub fn ones_dtype<S: Into<Shape>>(shape: S, dtype: DType) -> Self {
        Self::try_ones_dtype(shape, dtype)
            .unwrap_or_else(|err| panic!("Failed to create tensor data: {}", err))
    }

    /// Populates the data with the given value
    pub fn full<E: Element, S: Into<Shape>>(shape: S, fill_value: E) -> Self {
        let shape = shape.into();
        let data: Vec<E> = vec![fill_value; shape.num_elements()];
        Self::new(data, shape)
    }

    /// Populates the data with the given value
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the populated `TensorData` if successful, or a `DataError` if the conversion fails.
    pub fn try_full_dtype<E: Into<Scalar>, S: Into<Shape>>(
        shape: S,
        fill_value: E,
        dtype: DType,
    ) -> Result<Self, DataError> {
        let fill_value = fill_value.into();
        Ok(match dtype {
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
            DType::Bool(BoolStore::U8) => Self::full::<u8, _>(shape, fill_value.elem())
                .unchecked_cast(DType::Bool(BoolStore::U8)),
            DType::Bool(BoolStore::U32) => Self::full::<u32, _>(shape, fill_value.elem())
                .unchecked_cast(DType::Bool(BoolStore::U32)),
            DType::QFloat(_) => {
                return Err(DataError::UnsupportedConversion {
                    from: fill_value.dtype(),
                    to: dtype,
                });
            }
        })
    }

    /// Populates the data with the given value
    ///
    /// # Panics
    ///
    /// Panics if the conversion fails.
    pub fn full_dtype<E: Into<Scalar>, S: Into<Shape>>(
        shape: S,
        fill_value: E,
        dtype: DType,
    ) -> Self {
        Self::try_full_dtype(shape, fill_value, dtype)
            .unwrap_or_else(|err| panic!("Failed to create tensor data: {}", err))
    }

    // Unchecked, used to overwrite the dtype
    pub(super) fn unchecked_cast(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
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
                        mode: QuantMode::Symmetric,
                        value:
                            QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1,
                        ..
                    } => {
                        unimplemented!("Can't format yet");
                    }
                QuantScheme {
                    mode: QuantMode::Lookup,
                    ..
                } => {
                    format!("<lookup-quantized> {scheme:?}")
                }
            },
        };
        f.write_str(fmt.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use ::rand::{
        SeedableRng,
        rngs::{StdRng, SysRng},
    };
    use alloc::vec;
    use core::mem::{MaybeUninit, align_of, size_of};

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
    fn try_view_propagates_storage_access_failure() {
        use alloc::boxed::Box;

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

        let quantized = TensorData::quantized(vec![0i8], [1], scheme, &[1.0], None);
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

    #[test]
    fn test_try_random_dtype() {
        fn check<E: Element + PartialEq>(dtype: DType) {
            let data = TensorData::try_random_dtype(
                &[2, 2],
                Distribution::Bernoulli(0.5),
                &mut StdRng::seed_from_u64(0),
                dtype,
            )
            .unwrap();
            assert_eq!(data.shape.dims(), [2, 2]);
            let _vec = data.try_into_vec::<E>().unwrap();
        }

        check::<f32>(DType::F32);
        check::<f64>(DType::F64);
        check::<f16>(DType::F16);
        check::<bf16>(DType::BF16);

        check::<i64>(DType::I64);
        check::<i32>(DType::I32);
        check::<i16>(DType::I16);
        check::<i8>(DType::I8);

        check::<u64>(DType::U64);
        check::<u32>(DType::U32);
        check::<u16>(DType::U16);
        check::<u8>(DType::U8);

        check::<bool>(DType::Bool(BoolStore::Native));
        check::<u8>(DType::Bool(BoolStore::U8));
        check::<u32>(DType::Bool(BoolStore::U32));

        {
            assert_eq!(
                TensorData::try_random_dtype(
                    &[2, 2],
                    Distribution::Bernoulli(0.5),
                    &mut StdRng::seed_from_u64(0),
                    DType::QFloat(Default::default())
                )
                .unwrap_err(),
                DataError::UnsupportedConversion {
                    from: DType::F64,
                    to: DType::QFloat(Default::default()),
                }
            );
        }
    }

    #[test]
    fn test_zeros() {
        fn check<E: Element + PartialEq>() {
            let data = TensorData::zeros::<E, _>(&[2, 2]);
            assert_eq!(data.shape.dims(), [2, 2]);
            assert_eq!(&data.try_into_vec::<E>().unwrap(), &vec![0.elem::<E>(); 4]);
        }

        check::<f32>();
        check::<f64>();

        check::<i64>();
        check::<i32>();
        check::<i16>();
        check::<i8>();

        check::<u64>();
        check::<u32>();
        check::<u16>();
        check::<u8>();

        check::<bool>();
    }

    #[test]
    fn test_try_zeros_dtype() {
        fn check<E: Element + PartialEq>(dtype: DType) {
            let data = TensorData::try_zeros_dtype(&[2, 2], dtype).unwrap();
            assert_eq!(data.shape.dims(), [2, 2]);
            assert_eq!(&data.try_into_vec::<E>().unwrap(), &vec![0.elem::<E>(); 4]);
        }

        check::<f32>(DType::F32);
        check::<f64>(DType::F64);
        check::<f16>(DType::F16);
        check::<bf16>(DType::BF16);

        check::<i64>(DType::I64);
        check::<i32>(DType::I32);
        check::<i16>(DType::I16);
        check::<i8>(DType::I8);

        check::<u64>(DType::U64);
        check::<u32>(DType::U32);
        check::<u16>(DType::U16);
        check::<u8>(DType::U8);

        check::<bool>(DType::Bool(BoolStore::Native));
        check::<u8>(DType::Bool(BoolStore::U8));
        check::<u32>(DType::Bool(BoolStore::U32));

        assert_eq!(
            TensorData::try_zeros_dtype(&[2, 2], DType::QFloat(Default::default())).unwrap_err(),
            DataError::UnsupportedConversion {
                from: DType::I64,
                to: DType::QFloat(Default::default()),
            }
        );
    }

    #[test]
    fn test_ones() {
        fn check<E: Element + PartialEq>() {
            let data = TensorData::ones::<E, _>(&[2, 2]);
            assert_eq!(data.shape.dims(), [2, 2]);
            assert_eq!(&data.try_into_vec::<E>().unwrap(), &vec![1.elem::<E>(); 4]);
        }

        check::<f32>();
        check::<f64>();

        check::<i64>();
        check::<i32>();
        check::<i16>();
        check::<i8>();

        check::<u64>();
        check::<u32>();
        check::<u16>();
        check::<u8>();

        check::<bool>();
    }

    #[test]
    fn test_try_ones_dtype() {
        fn check<E: Element + PartialEq>(dtype: DType) {
            let data = TensorData::try_ones_dtype(&[2, 2], dtype).unwrap();
            assert_eq!(data.shape.dims(), [2, 2]);
            assert_eq!(&data.try_into_vec::<E>().unwrap(), &vec![1.elem::<E>(); 4]);
        }

        check::<f32>(DType::F32);
        check::<f64>(DType::F64);
        check::<f16>(DType::F16);
        check::<bf16>(DType::BF16);

        check::<i64>(DType::I64);
        check::<i32>(DType::I32);
        check::<i16>(DType::I16);
        check::<i8>(DType::I8);

        check::<u64>(DType::U64);
        check::<u32>(DType::U32);
        check::<u16>(DType::U16);
        check::<u8>(DType::U8);

        check::<bool>(DType::Bool(BoolStore::Native));
        check::<u8>(DType::Bool(BoolStore::U8));
        check::<u32>(DType::Bool(BoolStore::U32));

        assert_eq!(
            TensorData::try_ones_dtype(&[2, 2], DType::QFloat(Default::default())).unwrap_err(),
            DataError::UnsupportedConversion {
                from: DType::I64,
                to: DType::QFloat(Default::default()),
            }
        );
    }
}
