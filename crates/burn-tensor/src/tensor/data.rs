use core::f32;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use bytemuck::{checked::CheckedCastError, AnyBitPattern};
use half::{bf16, f16};

use crate::{
    quantization::{
        Quantization, QuantizationScheme, QuantizationStrategy, QuantizationType, QuantizedBytes,
    },
    tensor::bytes::Bytes,
    DType, Distribution, Element, ElementConversion,
};

use num_traits::pow::Pow;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use rand::RngCore;

use super::quantization::QuantizationMode;

/// The things that can go wrong when manipulating tensor data.
#[derive(Debug)]
pub enum DataError {
    /// Failed to cast the values to a specified element type.
    CastError(CheckedCastError),
    /// Invalid target element type.
    TypeMismatch(String),
}

/// Data structure for tensors.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TensorData {
    /// The values of the tensor (as bytes).
    bytes: Bytes,

    /// The shape of the tensor.
    pub shape: Vec<usize>,

    /// The data type of the tensor.
    pub dtype: DType,
}

impl TensorData {
    /// Creates a new tensor data structure.
    pub fn new<E: Element, S: Into<Vec<usize>>>(value: Vec<E>, shape: S) -> Self {
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
    pub fn quantized<E: Element, S: Into<Vec<usize>>>(
        value: Vec<E>,
        shape: S,
        strategy: QuantizationStrategy,
    ) -> Self {
        let shape = shape.into();
        Self::check_data_len(&value, &shape);

        let q_bytes = QuantizedBytes::new(value, strategy);

        Self {
            bytes: q_bytes.bytes,
            shape,
            dtype: DType::QFloat(q_bytes.scheme),
        }
    }

    /// Creates a new tensor data structure from raw bytes.
    ///
    /// Prefer [`TensorData::new`] or [`TensorData::quantized`] over this method unless you are
    /// certain that the bytes representation is valid.
    pub fn from_bytes<S: Into<Vec<usize>>>(bytes: Vec<u8>, shape: S, dtype: DType) -> Self {
        Self {
            bytes: Bytes::from_bytes_vec(bytes),
            shape: shape.into(),
            dtype,
        }
    }

    // Check that the input vector contains a correct number of elements
    fn check_data_len<E: Element>(data: &[E], shape: &Vec<usize>) {
        let expected_data_len = Self::numel(shape);
        let num_data = data.len();
        assert_eq!(
            expected_data_len, num_data,
            "Shape {:?} is invalid for input of size {:?}",
            shape, num_data,
        );
    }

    fn try_as_slice<E: Element>(&self) -> Result<&[E], DataError> {
        bytemuck::checked::try_cast_slice(&self.bytes).map_err(DataError::CastError)
    }

    /// Returns the immutable slice view of the tensor data.
    pub fn as_slice<E: Element>(&self) -> Result<&[E], DataError> {
        if E::dtype() == self.dtype {
            self.try_as_slice()
        } else {
            Err(DataError::TypeMismatch(format!(
                "Invalid target element type (expected {:?}, got {:?})",
                self.dtype,
                E::dtype()
            )))
        }
    }

    /// Returns the mutable slice view of the tensor data.
    ///
    /// # Panics
    /// If the target element type is different from the stored element type.
    pub fn as_mut_slice<E: Element>(&mut self) -> Result<&mut [E], DataError> {
        if E::dtype() == self.dtype {
            bytemuck::checked::try_cast_slice_mut(&mut self.bytes).map_err(DataError::CastError)
        } else {
            Err(DataError::TypeMismatch(format!(
                "Invalid target element type (expected {:?}, got {:?})",
                self.dtype,
                E::dtype()
            )))
        }
    }

    /// Returns the tensor data as a vector of scalar values.
    pub fn to_vec<E: Element>(&self) -> Result<Vec<E>, DataError> {
        Ok(self.as_slice()?.to_vec())
    }

    /// Returns the tensor data as a vector of scalar values.
    pub fn into_vec<E: Element>(self) -> Result<Vec<E>, DataError> {
        // This means we cannot call `into_vec` for QFloat
        if E::dtype() != self.dtype {
            return Err(DataError::TypeMismatch(format!(
                "Invalid target element type (expected {:?}, got {:?})",
                self.dtype,
                E::dtype()
            )));
        }

        let mut me = self;
        me.bytes = match me.bytes.try_into_vec::<E>() {
            Ok(elems) => return Ok(elems),
            Err(bytes) => bytes,
        };
        // The bytes might have been deserialized and allocated with a different align.
        // In that case, we have to memcopy the data into a new vector, more suitably allocated
        Ok(bytemuck::checked::try_cast_slice(me.as_bytes())
            .map_err(DataError::CastError)?
            .to_vec())
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
                DType::F32 => Box::new(
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
                DType::Bool => Box::new(self.bytes.iter().map(|e| e.elem::<E>())),
                DType::QFloat(scheme) => match scheme {
                    QuantizationScheme::PerTensor(_mode, QuantizationType::QInt8) => {
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
                    QuantizationScheme::PerBlock(_mode, _dtype, _block_layout) => todo!(),
                },
            }
        }
    }

    /// Returns the total number of elements of the tensor data.
    pub fn num_elements(&self) -> usize {
        Self::numel(&self.shape)
    }

    fn numel(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    /// Populates the data with random values.
    pub fn random<E: Element, R: RngCore, S: Into<Vec<usize>>>(
        shape: S,
        distribution: Distribution,
        rng: &mut R,
    ) -> Self {
        let shape = shape.into();
        let num_elements = Self::numel(&shape);
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(E::random(distribution, rng));
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with zeros.
    pub fn zeros<E: Element, S: Into<Vec<usize>>>(shape: S) -> TensorData {
        let shape = shape.into();
        let num_elements = Self::numel(&shape);
        let mut data = Vec::<E>::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(0.elem());
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with ones.
    pub fn ones<E: Element, S: Into<Vec<usize>>>(shape: S) -> TensorData {
        let shape = shape.into();
        let num_elements = Self::numel(&shape);
        let mut data = Vec::<E>::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(1.elem());
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with the given value
    pub fn full<E: Element, S: Into<Vec<usize>>>(shape: S, fill_value: E) -> TensorData {
        let shape = shape.into();
        let num_elements = Self::numel(&shape);
        let mut data = Vec::<E>::with_capacity(num_elements);
        for _ in 0..num_elements {
            data.push(fill_value)
        }

        TensorData::new(data, shape)
    }

    /// Converts the data to a different element type.
    pub fn convert<E: Element>(self) -> Self {
        self.convert_dtype(E::dtype())
    }

    /// Converts the data to a different element type.
    pub fn convert_dtype(self, dtype: DType) -> Self {
        if dtype == self.dtype {
            self
        } else if dtype.size() == self.dtype.size()
            && !matches!(self.dtype, DType::Bool | DType::QFloat(_))
            && !matches!(dtype, DType::Bool | DType::QFloat(_))
        {
            match self.dtype {
                DType::F64 => self.convert_inplace_dtype::<f64>(dtype),
                DType::F32 => self.convert_inplace_dtype::<f32>(dtype),
                DType::F16 => self.convert_inplace_dtype::<f16>(dtype),
                DType::BF16 => self.convert_inplace_dtype::<bf16>(dtype),
                DType::I64 => self.convert_inplace_dtype::<i64>(dtype),
                DType::I32 => self.convert_inplace_dtype::<i32>(dtype),
                DType::I16 => self.convert_inplace_dtype::<i16>(dtype),
                DType::I8 => self.convert_inplace_dtype::<i8>(dtype),
                DType::U64 => self.convert_inplace_dtype::<u64>(dtype),
                DType::U32 => self.convert_inplace_dtype::<u32>(dtype),
                DType::U16 => self.convert_inplace_dtype::<u16>(dtype),
                DType::U8 => self.convert_inplace_dtype::<u8>(dtype),
                DType::Bool | DType::QFloat(_) => unreachable!(),
            }
        } else {
            match dtype {
                DType::F64 => TensorData::new(self.iter::<f64>().collect(), self.shape),
                DType::F32 => TensorData::new(self.iter::<f32>().collect(), self.shape),
                DType::F16 => TensorData::new(self.iter::<f16>().collect(), self.shape),
                DType::BF16 => TensorData::new(self.iter::<bf16>().collect(), self.shape),
                DType::I64 => TensorData::new(self.iter::<i64>().collect(), self.shape),
                DType::I32 => TensorData::new(self.iter::<i32>().collect(), self.shape),
                DType::I16 => TensorData::new(self.iter::<i16>().collect(), self.shape),
                DType::I8 => TensorData::new(self.iter::<i8>().collect(), self.shape),
                DType::U64 => TensorData::new(self.iter::<u64>().collect(), self.shape),
                DType::U32 => TensorData::new(self.iter::<u32>().collect(), self.shape),
                DType::U16 => TensorData::new(self.iter::<u16>().collect(), self.shape),
                DType::U8 => TensorData::new(self.iter::<u8>().collect(), self.shape),
                DType::Bool => TensorData::new(self.iter::<bool>().collect(), self.shape),
                DType::QFloat(_) => unreachable!(),
            }
        }
    }

    fn convert_inplace_dtype<Current: Element + AnyBitPattern>(self, dtype: DType) -> Self {
        match dtype {
            DType::F64 => self.convert_inplace::<Current, f64>(),
            DType::F32 => self.convert_inplace::<Current, f32>(),
            DType::F16 => self.convert_inplace::<Current, f16>(),
            DType::BF16 => self.convert_inplace::<Current, bf16>(),
            DType::I64 => self.convert_inplace::<Current, i64>(),
            DType::I32 => self.convert_inplace::<Current, i32>(),
            DType::I16 => self.convert_inplace::<Current, i16>(),
            DType::I8 => self.convert_inplace::<Current, i8>(),
            DType::U64 => self.convert_inplace::<Current, u64>(),
            DType::U32 => self.convert_inplace::<Current, u32>(),
            DType::U16 => self.convert_inplace::<Current, u16>(),
            DType::U8 => self.convert_inplace::<Current, u8>(),
            DType::Bool | DType::QFloat(_) => unreachable!(),
        }
    }

    fn convert_inplace<Current: Element + AnyBitPattern, Target: Element>(mut self) -> Self {
        let step = core::mem::size_of::<Current>();

        for offset in 0..(self.bytes.len() / step) {
            let start = offset * step;
            let end = start + step;

            let slice_old = &mut self.bytes[start..end];
            let val: Current = *bytemuck::from_bytes(slice_old);
            let val = &val.elem::<Target>();
            let slice_new = bytemuck::bytes_of(val);

            slice_old.clone_from_slice(slice_new);
        }
        self.dtype = Target::dtype();

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

    /// Applies the data quantization strategy.
    ///
    /// # Panics
    ///
    /// Panics if the data type is not supported for quantization.
    pub fn with_quantization(self, quantization: QuantizationStrategy) -> Self {
        assert_eq!(
            self.dtype,
            DType::F32,
            "Only f32 data type can be quantized"
        );
        match &quantization {
            QuantizationStrategy::PerTensorAffineInt8(strategy) => TensorData::quantized(
                strategy.quantize(self.as_slice().unwrap()),
                self.shape,
                quantization,
            ),
            QuantizationStrategy::PerTensorSymmetricInt8(strategy) => TensorData::quantized(
                strategy.quantize(self.as_slice().unwrap()),
                self.shape,
                quantization,
            ),
        }
    }

    /// Dequantizes the data according to its quantization scheme.
    pub fn dequantize(self) -> Result<Self, DataError> {
        if let DType::QFloat(scheme) = self.dtype {
            let num_elements = self.num_elements();
            let q_bytes = QuantizedBytes {
                bytes: self.bytes,
                scheme,
                num_elements,
            };

            let values = q_bytes.dequantize().0;
            Ok(Self::new(values, self.shape))
        } else {
            Err(DataError::TypeMismatch(format!(
                "Expected quantized data, got {:?}",
                self.dtype
            )))
        }
    }

    /// Asserts the data is approximately equal to another data.
    ///
    /// # Arguments
    ///
    /// * `other` - The other data.
    /// * `precision` - The precision of the comparison.
    ///
    /// # Panics
    ///
    /// Panics if the data is not approximately equal.
    #[track_caller]
    pub fn assert_approx_eq(&self, other: &Self, precision: usize) {
        let tolerance = 0.1.pow(precision as f64);

        self.assert_approx_eq_diff(other, tolerance)
    }

    /// Asserts the data is equal to another data.
    ///
    /// # Arguments
    ///
    /// * `other` - The other data.
    /// * `strict` - If true, the data types must the be same.
    ///              Otherwise, the comparison is done in the current data type.
    ///
    /// # Panics
    ///
    /// Panics if the data is not equal.
    #[track_caller]
    pub fn assert_eq(&self, other: &Self, strict: bool) {
        if strict {
            assert_eq!(
                self.dtype, other.dtype,
                "Data types differ ({:?} != {:?})",
                self.dtype, other.dtype
            );
        }

        match self.dtype {
            DType::F64 => self.assert_eq_elem::<f64>(other),
            DType::F32 => self.assert_eq_elem::<f32>(other),
            DType::F16 => self.assert_eq_elem::<f16>(other),
            DType::BF16 => self.assert_eq_elem::<bf16>(other),
            DType::I64 => self.assert_eq_elem::<i64>(other),
            DType::I32 => self.assert_eq_elem::<i32>(other),
            DType::I16 => self.assert_eq_elem::<i16>(other),
            DType::I8 => self.assert_eq_elem::<i8>(other),
            DType::U64 => self.assert_eq_elem::<u64>(other),
            DType::U32 => self.assert_eq_elem::<u32>(other),
            DType::U16 => self.assert_eq_elem::<u16>(other),
            DType::U8 => self.assert_eq_elem::<u8>(other),
            DType::Bool => self.assert_eq_elem::<bool>(other),
            DType::QFloat(q) => {
                // Strict or not, it doesn't make sense to compare quantized data to not quantized data for equality
                let q_other = if let DType::QFloat(q_other) = other.dtype {
                    q_other
                } else {
                    panic!("Quantized data differs from other not quantized data")
                };
                match (q, q_other) {
                    (
                        QuantizationScheme::PerTensor(
                            QuantizationMode::Affine,
                            QuantizationType::QInt8,
                        ),
                        QuantizationScheme::PerTensor(
                            QuantizationMode::Affine,
                            QuantizationType::QInt8,
                        ),
                    )
                    | (
                        QuantizationScheme::PerTensor(
                            QuantizationMode::Symmetric,
                            QuantizationType::QInt8,
                        ),
                        QuantizationScheme::PerTensor(
                            QuantizationMode::Symmetric,
                            QuantizationType::QInt8,
                        ),
                    ) => self.assert_eq_elem::<i8>(other),
                    _ => panic!("Quantization schemes differ ({:?} != {:?})", q, q_other),
                }
            }
        }
    }

    #[track_caller]
    fn assert_eq_elem<E: Element>(&self, other: &Self) {
        let mut message = String::new();
        if self.shape != other.shape {
            message += format!(
                "\n  => Shape is different: {:?} != {:?}",
                self.shape, other.shape
            )
            .as_str();
        }

        let mut num_diff = 0;
        let max_num_diff = 5;
        for (i, (a, b)) in self.iter::<E>().zip(other.iter::<E>()).enumerate() {
            if a.cmp(&b).is_ne() {
                // Only print the first 5 different values.
                if num_diff < max_num_diff {
                    message += format!("\n  => Position {i}: {a} != {b}").as_str();
                }
                num_diff += 1;
            }
        }

        if num_diff >= max_num_diff {
            message += format!("\n{} more errors...", num_diff - max_num_diff).as_str();
        }

        if !message.is_empty() {
            panic!("Tensors are not eq:{}", message);
        }
    }

    /// Asserts the data is approximately equal to another data.
    ///
    /// # Arguments
    ///
    /// * `other` - The other data.
    /// * `tolerance` - The tolerance of the comparison.
    ///
    /// # Panics
    ///
    /// Panics if the data is not approximately equal.
    #[track_caller]
    pub fn assert_approx_eq_diff(&self, other: &Self, tolerance: f64) {
        let mut message = String::new();
        if self.shape != other.shape {
            message += format!(
                "\n  => Shape is different: {:?} != {:?}",
                self.shape, other.shape
            )
            .as_str();
        }

        let iter = self.iter::<f64>().zip(other.iter::<f64>());

        let mut num_diff = 0;
        let max_num_diff = 5;

        for (i, (a, b)) in iter.enumerate() {
            //if they are both nan, then they are equally nan
            let both_nan = a.is_nan() && b.is_nan();
            //this works for both infinities
            let both_inf = a.is_infinite() && b.is_infinite() && ((a > 0.) == (b > 0.));

            if both_nan || both_inf {
                continue;
            }

            let err = (a - b).abs();

            if self.dtype.is_float() {
                if let Some((err, tolerance)) = compare_floats(a, b, self.dtype, tolerance) {
                    // Only print the first 5 different values.
                    if num_diff < max_num_diff {
                        message += format!(
                            "\n  => Position {i}: {a} != {b} | difference {err} > tolerance \
                         {tolerance}"
                        )
                        .as_str();
                    }
                    num_diff += 1;
                }
            } else if err > tolerance || err.is_nan() {
                // Only print the first 5 different values.
                if num_diff < max_num_diff {
                    message += format!(
                        "\n  => Position {i}: {a} != {b} | difference {err} > tolerance \
                         {tolerance}"
                    )
                    .as_str();
                }
                num_diff += 1;
            }
        }

        if num_diff >= max_num_diff {
            message += format!("\n{} more errors...", num_diff - 5).as_str();
        }

        if !message.is_empty() {
            panic!("Tensors are not approx eq:{}", message);
        }
    }

    /// Asserts each value is within a given range.
    ///
    /// # Arguments
    ///
    /// * `range` - The range.
    ///
    /// # Panics
    ///
    /// If any value is not within the half-open range bounded inclusively below
    /// and exclusively above (`start..end`).
    pub fn assert_within_range<E: Element>(&self, range: core::ops::Range<E>) {
        let start = range.start.elem::<f32>();
        let end = range.end.elem::<f32>();

        for elem in self.iter::<f32>() {
            if elem < start || elem >= end {
                panic!("Element ({elem:?}) is not within range {range:?}");
            }
        }
    }

    /// Asserts each value is within a given inclusive range.
    ///
    /// # Arguments
    ///
    /// * `range` - The range.
    ///
    /// # Panics
    ///
    /// If any value is not within the half-open range bounded inclusively (`start..=end`).
    pub fn assert_within_range_inclusive<E: Element>(&self, range: core::ops::RangeInclusive<E>) {
        let start = range.start().elem::<f32>();
        let end = range.end().elem::<f32>();

        for elem in self.iter::<f32>() {
            if elem < start || elem > end {
                panic!("Element ({elem:?}) is not within range {range:?}");
            }
        }
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

impl<
        Elem: Element,
        const A: usize,
        const B: usize,
        const C: usize,
        const D: usize,
        const E: usize,
    > From<[[[[[Elem; E]; D]; C]; B]; A]> for TensorData
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
            DType::F32 => format!("{:?}", self.as_slice::<f32>().unwrap()),
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
            DType::Bool => format!("{:?}", self.as_slice::<bool>().unwrap()),
            DType::QFloat(scheme) => match scheme {
                QuantizationScheme::PerTensor(
                    QuantizationMode::Affine,
                    QuantizationType::QInt8,
                )
                | QuantizationScheme::PerTensor(
                    QuantizationMode::Symmetric,
                    QuantizationType::QInt8,
                ) => {
                    format!("{:?} {scheme:?}", self.try_as_slice::<i8>().unwrap())
                }
                QuantizationScheme::PerBlock(_mode, _dtype, _block_layout) => todo!(),
            },
        };
        f.write_str(fmt.as_str())
    }
}

fn compare_floats(value: f64, other: f64, ty: DType, tolerance: f64) -> Option<(f64, f64)> {
    let epsilon_deviations = tolerance / f32::EPSILON as f64;
    let epsilon = match ty {
        DType::F64 => f32::EPSILON as f64, // Don't increase precision beyond `f32`, see below
        DType::F32 => f32::EPSILON as f64,
        DType::F16 => half::f16::EPSILON.to_f64(),
        DType::BF16 => half::bf16::EPSILON.to_f64(),
        _ => unreachable!(),
    };
    let tolerance_norm = epsilon_deviations * epsilon;
    // Clamp to 1.0 so we don't require more precision than `tolerance`. This is because literals
    // have a fixed number of digits, so increasing precision breaks things
    let value_abs = value.abs().max(1.0);
    let tolerance_adjusted = tolerance_norm * value_abs;

    let err = (value - other).abs();

    if err > tolerance_adjusted || err.is_nan() {
        Some((err, tolerance_adjusted))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::{quantization::AffineQuantization, Shape};

    use super::*;
    use alloc::vec;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn into_vec_should_yield_same_value_as_iter() {
        let shape = Shape::new([3, 5, 6]);
        let data = TensorData::random::<f32, _, _>(
            shape,
            Distribution::Default,
            &mut StdRng::from_os_rng(),
        );

        let expected = data.iter::<f32>().collect::<Vec<f32>>();
        let actual = data.into_vec::<f32>().unwrap();

        assert_eq!(expected, actual);
    }

    #[test]
    #[should_panic]
    fn into_vec_should_assert_wrong_dtype() {
        let shape = Shape::new([3, 5, 6]);
        let data = TensorData::random::<f32, _, _>(
            shape,
            Distribution::Default,
            &mut StdRng::from_os_rng(),
        );

        data.into_vec::<i32>().unwrap();
    }

    #[test]
    fn should_have_right_num_elements() {
        let shape = Shape::new([3, 5, 6]);
        let num_elements = shape.num_elements();
        let data = TensorData::random::<f32, _, _>(
            shape,
            Distribution::Default,
            &mut StdRng::from_os_rng(),
        );

        assert_eq!(num_elements, data.bytes.len() / 4); // f32 stored as u8s
        assert_eq!(num_elements, data.as_slice::<f32>().unwrap().len());
    }

    #[test]
    fn should_have_right_shape() {
        let data = TensorData::from([[3.0, 5.0, 6.0]]);
        assert_eq!(data.shape, vec![1, 3]);

        let data = TensorData::from([[4.0, 5.0, 8.0], [3.0, 5.0, 6.0]]);
        assert_eq!(data.shape, vec![2, 3]);

        let data = TensorData::from([3.0, 5.0, 6.0]);
        assert_eq!(data.shape, vec![3]);
    }

    #[test]
    fn should_assert_appox_eq_limit() {
        let data1 = TensorData::from([[3.0, 5.0, 6.0]]);
        let data2 = TensorData::from([[3.03, 5.0, 6.0]]);

        data1.assert_approx_eq(&data2, 2);
    }

    #[test]
    #[should_panic]
    fn should_assert_approx_eq_above_limit() {
        let data1 = TensorData::from([[3.0, 5.0, 6.0]]);
        let data2 = TensorData::from([[3.031, 5.0, 6.0]]);

        data1.assert_approx_eq(&data2, 2);
    }

    #[test]
    #[should_panic]
    fn should_assert_appox_eq_check_shape() {
        let data1 = TensorData::from([[3.0, 5.0, 6.0, 7.0]]);
        let data2 = TensorData::from([[3.0, 5.0, 6.0]]);

        data1.assert_approx_eq(&data2, 2);
    }

    #[test]
    fn should_convert_bytes_correctly() {
        let mut vector: Vec<f32> = Vec::with_capacity(5);
        vector.push(2.0);
        vector.push(3.0);
        let data1 = TensorData::new(vector, vec![2]);

        let factor = core::mem::size_of::<f32>() / core::mem::size_of::<u8>();
        assert_eq!(data1.bytes.len(), 2 * factor);
        assert_eq!(data1.bytes.capacity(), 5 * factor);
    }

    #[test]
    fn should_convert_bytes_correctly_inplace() {
        fn test_precision<E: Element>() {
            let data = TensorData::new((0..32).collect(), [32]);
            for (i, val) in data
                .clone()
                .convert::<E>()
                .into_vec::<E>()
                .unwrap()
                .into_iter()
                .enumerate()
            {
                assert_eq!(i as u32, val.elem::<u32>())
            }
        }
        test_precision::<f32>();
        test_precision::<f16>();
        test_precision::<i64>();
        test_precision::<i32>();
    }

    #[test]
    #[should_panic = "Expected quantized data"]
    fn should_not_dequantize() {
        let data = TensorData::from([[3.0, 5.0, 6.0, 7.0]]);
        data.dequantize().unwrap();
    }

    #[test]
    fn should_support_dequantize() {
        // Quantized [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
        let data = TensorData::quantized(
            vec![-128i8, -77, -26, 25, 76, 127],
            [2, 3],
            QuantizationStrategy::PerTensorAffineInt8(AffineQuantization::init(0.019607844, -128)),
        );

        let output = data.dequantize().unwrap();

        output.assert_approx_eq(&TensorData::from([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]), 4);
    }
}
