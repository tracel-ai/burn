use core::f32;

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use bytemuck::{AnyBitPattern, CheckedBitPattern, Zeroable, cast_mut, checked::CheckedCastError};
use rand::RngCore;

use crate::distribution::Distribution;
use crate::element::{Element, ElementConversion};
use crate::quantization::{QuantLevel, QuantMode, QuantScheme, QuantValue, QuantizedBytes};
use crate::{Bytes, DType, bf16, f16};

/// Data structure for tensors.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TensorData {
    /// The values of the tensor (as bytes).
    pub bytes: Bytes,

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
    pub fn from_bytes<S: Into<Vec<usize>>>(bytes: Bytes, shape: S, dtype: DType) -> Self {
        Self {
            bytes,
            shape: shape.into(),
            dtype,
        }
    }

    /// Creates a new tensor data structure from raw bytes stored in a vector.
    ///
    /// Prefer [`TensorData::new`] or [`TensorData::quantized`] over this method unless you are
    /// certain that the bytes representation is valid.
    pub fn from_bytes_vec<S: Into<Vec<usize>>>(bytes: Vec<u8>, shape: S, dtype: DType) -> Self {
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
            "Shape {shape:?} is invalid for input of size {num_data:?}",
        );
    }

    /// Returns the immutable slice view of the tensor data.
    pub fn as_slice<E: Element>(&self) -> Result<&[E], DataError> {
        if E::dtype() == self.dtype {
            match E::dtype() {
                // The only way to create a bool `TensorData` with invalid values is by unsafely modifying
                // the dtype. This should be considered unsafe to begin with, so we unsafely cast bool
                // to u8 to skip bit validation. Validation iterates through the entire vector, so it's slow.
                DType::Bool => {
                    let slice = bytemuck::checked::try_cast_slice::<_, u8>(&self.bytes)
                        .map_err(DataError::CastError)?;
                    Ok(unsafe { core::mem::transmute::<&[u8], &[E]>(slice) })
                }
                _ => bytemuck::checked::try_cast_slice(&self.bytes).map_err(DataError::CastError),
            }
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
            match E::dtype() {
                // The only way to create a bool `TensorData` with invalid values is by unsafely modifying
                // the dtype. This should be considered unsafe to begin with, so we unsafely cast bool
                // to u8 to skip bit validation. Validation iterates through the entire vector, so it's slow.
                DType::Bool => {
                    let slice = bytemuck::checked::try_cast_slice_mut::<_, u8>(&mut self.bytes)
                        .map_err(DataError::CastError)?;
                    Ok(unsafe { core::mem::transmute::<&mut [u8], &mut [E]>(slice) })
                }
                _ => bytemuck::checked::try_cast_slice_mut(&mut self.bytes)
                    .map_err(DataError::CastError),
            }
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

        match E::dtype() {
            // The only way to create a bool `TensorData` with invalid values is by unsafely modifying
            // the dtype. This should be considered unsafe to begin with, so we unsafely cast bool
            // to u8 to skip bit validation. Validation iterates through the entire vector, so it's slow.
            DType::Bool => {
                let vec = self.into_vec_unchecked::<u8>()?;
                Ok(unsafe { core::mem::transmute::<Vec<u8>, Vec<E>>(vec) })
            }
            _ => self.into_vec_unchecked(),
        }
    }

    /// Returns the tensor data as a vector of scalar values. Does not check dtype.
    fn into_vec_unchecked<E: Element>(self) -> Result<Vec<E>, DataError> {
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
                DType::Bool => Box::new(self.bytes.iter().map(|e| e.elem::<E>())),
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

    #[allow(dead_code)]
    /// Populates the data with the given value
    pub fn full_dtype<E: Element, S: Into<Vec<usize>>>(
        shape: S,
        fill_value: E,
        dtype: DType,
    ) -> TensorData {
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
            DType::Bool => Self::full::<bool, _>(shape, fill_value.elem()),
            DType::QFloat(_) => unreachable!(),
        }
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
                DType::F32 | DType::Flex32 => self.convert_inplace_dtype::<f32>(dtype),
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
            match self.dtype {
                DType::F64 => self.convert_clone_dtype::<f64>(dtype),
                DType::F32 | DType::Flex32 => self.convert_clone_dtype::<f32>(dtype),
                DType::F16 => self.convert_clone_dtype::<f16>(dtype),
                DType::BF16 => self.convert_clone_dtype::<bf16>(dtype),
                DType::I64 => self.convert_clone_dtype::<i64>(dtype),
                DType::I32 => self.convert_clone_dtype::<i32>(dtype),
                DType::I16 => self.convert_clone_dtype::<i16>(dtype),
                DType::I8 => self.convert_clone_dtype::<i8>(dtype),
                DType::U64 => self.convert_clone_dtype::<u64>(dtype),
                DType::U32 => self.convert_clone_dtype::<u32>(dtype),
                DType::U16 => self.convert_clone_dtype::<u16>(dtype),
                DType::U8 => self.convert_clone_dtype::<u8>(dtype),
                DType::Bool => self.convert_clone_dtype::<bool>(dtype),
                DType::QFloat(_) => unreachable!(),
            }
        }
    }

    fn convert_inplace_dtype<Current: Element + AnyBitPattern>(self, dtype: DType) -> Self {
        match dtype {
            DType::F64 => self.convert_inplace::<Current, f64>(),
            DType::F32 | DType::Flex32 => self.convert_inplace::<Current, f32>(),
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

    fn convert_inplace<Current: Element + AnyBitPattern, Target: Element + AnyBitPattern>(
        mut self,
    ) -> Self {
        for x in bytemuck::cast_slice_mut::<_, Current>(&mut self.bytes) {
            let t: Target = x.elem();
            let x = cast_mut::<_, Target>(x);
            *x = t;
        }

        self.dtype = Target::dtype();

        self
    }

    fn convert_clone_dtype<Current: Element + CheckedBitPattern>(self, dtype: DType) -> Self {
        match dtype {
            DType::F64 => self.convert_clone::<Current, f64>(),
            DType::F32 | DType::Flex32 => self.convert_clone::<Current, f32>(),
            DType::F16 => self.convert_clone::<Current, f16>(),
            DType::BF16 => self.convert_clone::<Current, bf16>(),
            DType::I64 => self.convert_clone::<Current, i64>(),
            DType::I32 => self.convert_clone::<Current, i32>(),
            DType::I16 => self.convert_clone::<Current, i16>(),
            DType::I8 => self.convert_clone::<Current, i8>(),
            DType::U64 => self.convert_clone::<Current, u64>(),
            DType::U32 => self.convert_clone::<Current, u32>(),
            DType::U16 => self.convert_clone::<Current, u16>(),
            DType::U8 => self.convert_clone::<Current, u8>(),
            DType::Bool => self.convert_clone::<Current, bool>(),
            DType::QFloat(_) => unreachable!(),
        }
    }

    fn convert_clone<Current: Element + CheckedBitPattern, Target: Element + Zeroable>(
        self,
    ) -> Self {
        let this = bytemuck::checked::cast_slice::<_, Current>(&self.bytes);
        let mut out: Vec<Target> = ::alloc::vec![Zeroable::zeroed(); self.num_elements()];

        for (x, out) in this.iter().zip(&mut out) {
            *out = x.elem();
        }

        Self::new(out, self.shape)
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
            DType::Bool => format!("{:?}", self.as_slice::<bool>().unwrap()),
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
            },
        };
        f.write_str(fmt.as_str())
    }
}

/// The things that can go wrong when manipulating tensor data.
#[derive(Debug)]
pub enum DataError {
    /// Failed to cast the values to a specified element type.
    CastError(CheckedCastError),
    /// Invalid target element type.
    TypeMismatch(String),
}

impl core::error::Error for DataError {}

impl core::fmt::Display for DataError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("{self:?}").as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn should_have_rank() {
        let shape = [3, 5, 6];
        let data = TensorData::random::<f32, _, _>(
            shape,
            Distribution::Default,
            &mut StdRng::from_os_rng(),
        );

        assert_eq!(data.rank(), 3);
    }

    #[test]
    fn into_vec_should_yield_same_value_as_iter() {
        let shape = [3, 5, 6];
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
        let shape = [3, 5, 6];
        let data = TensorData::random::<f32, _, _>(
            shape,
            Distribution::Default,
            &mut StdRng::from_os_rng(),
        );

        data.into_vec::<i32>().unwrap();
    }

    #[test]
    fn should_have_right_num_elements() {
        let shape = [3, 5, 6];
        let num_elements: usize = shape.iter().product();
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
}
