use core::any::{Any, TypeId};

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use half::{bf16, f16};

use crate::{tensor::Shape, DType, Distribution, Element, ElementConversion};

use num_traits::pow::Pow;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use rand::RngCore;

/// The things that can go wrong when manipulating tensor data.
#[derive(Debug)]
pub enum DataError {
    /// Failed to cast the values to a specified element type.
    CastError(bytemuck::checked::CheckedCastError),
    /// Invalid target element type.
    TypeMismatch(String),
}

/// Data structure for tensors.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TensorData<const D: usize> {
    /// The values of the tensor (as bytes).
    value: Vec<u8>,

    /// The shape of the tensor.
    #[serde(flatten)]
    pub shape: Shape<D>,

    /// The data type of the tensor.
    pub dtype: DType,
}

impl<const D: usize> TensorData<D> {
    /// Creates a new tensor data structure.
    pub fn new<E: Element, S: Into<Shape<D>>>(value: Vec<E>, shape: S) -> Self {
        Self {
            value: bytemuck::checked::cast_slice(&value).to_vec(),
            shape: shape.into(),
            dtype: E::dtype(),
        }
    }

    /// Returns the immutable slice view of the tensor data.
    pub fn as_slice<E: Element>(&self) -> Result<&[E], DataError> {
        if E::dtype() == self.dtype {
            bytemuck::checked::try_cast_slice(&self.value).map_err(DataError::CastError)
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
            bytemuck::checked::try_cast_slice_mut(&mut self.value).map_err(DataError::CastError)
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

    /// Returns an iterator over the values of the tensor data.
    pub fn iter<E: Element>(&self) -> Box<dyn Iterator<Item = E> + '_> {
        if E::dtype() == self.dtype {
            Box::new(bytemuck::checked::cast_slice(&self.value).iter().copied())
        } else {
            match self.dtype {
                DType::I8 => Box::new(
                    bytemuck::checked::cast_slice(&self.value)
                        .iter()
                        .map(|e: &i8| e.elem::<E>()),
                ),
                DType::I16 => Box::new(
                    bytemuck::checked::cast_slice(&self.value)
                        .iter()
                        .map(|e: &i16| e.elem::<E>()),
                ),
                DType::I32 => Box::new(
                    bytemuck::checked::cast_slice(&self.value)
                        .iter()
                        .map(|e: &i32| e.elem::<E>()),
                ),
                DType::I64 => Box::new(
                    bytemuck::checked::cast_slice(&self.value)
                        .iter()
                        .map(|e: &i64| e.elem::<E>()),
                ),
                DType::U8 => Box::new(self.value.iter().map(|e| e.elem::<E>())),
                DType::U32 => Box::new(
                    bytemuck::checked::cast_slice(&self.value)
                        .iter()
                        .map(|e: &u32| e.elem::<E>()),
                ),
                DType::U64 => Box::new(
                    bytemuck::checked::cast_slice(&self.value)
                        .iter()
                        .map(|e: &u64| e.elem::<E>()),
                ),
                DType::BF16 => Box::new(
                    bytemuck::checked::cast_slice(&self.value)
                        .iter()
                        .map(|e: &bf16| e.elem::<E>()),
                ),
                DType::F16 => Box::new(
                    bytemuck::checked::cast_slice(&self.value)
                        .iter()
                        .map(|e: &f16| e.elem::<E>()),
                ),
                DType::F32 => Box::new(
                    bytemuck::checked::cast_slice(&self.value)
                        .iter()
                        .map(|e: &f32| e.elem::<E>()),
                ),
                DType::F64 => Box::new(
                    bytemuck::checked::cast_slice(&self.value)
                        .iter()
                        .map(|e: &f64| e.elem::<E>()),
                ),
                // bool is a byte value equal to either 0 or 1
                DType::Bool => Box::new(self.value.iter().map(|e| e.elem::<E>())),
            }
        }
    }

    /// Populates the data with random values.
    pub fn random<E: Element, R: RngCore>(
        shape: Shape<D>,
        distribution: Distribution,
        rng: &mut R,
    ) -> Self {
        let num_elements = shape.num_elements();
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(E::random(distribution, rng));
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with zeros.
    pub fn zeros<E: Element, S: Into<Shape<D>>>(shape: S) -> TensorData<D> {
        let shape = shape.into();
        let num_elements = shape.num_elements();
        let mut data = Vec::<E>::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(0.elem());
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with ones.
    pub fn ones<E: Element, S: Into<Shape<D>>>(shape: S) -> TensorData<D> {
        let shape = shape.into();
        let num_elements = shape.num_elements();
        let mut data = Vec::<E>::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(1.elem());
        }

        TensorData::new(data, shape)
    }

    /// Populates the data with the given value
    pub fn full<E: Element, S: Into<Shape<D>>>(shape: S, fill_value: E) -> TensorData<D> {
        let shape = shape.into();
        let num_elements = shape.num_elements();
        let mut data = Vec::<E>::with_capacity(num_elements);
        for _ in 0..num_elements {
            data.push(fill_value)
        }

        TensorData::new(data, shape)
    }

    /// Converts the data to a different element type.
    pub fn convert<E: Element>(self) -> Self {
        if E::dtype() == self.dtype {
            self
        } else {
            TensorData::new(self.iter::<E>().collect(), self.shape)
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
                self.shape.dims, other.shape.dims
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

            let err = ((a - b).pow(2.0f64)).sqrt();

            if err > tolerance || err.is_nan() {
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
}

impl<E: Element, const A: usize> From<[E; A]> for TensorData<1> {
    fn from(elems: [E; A]) -> Self {
        TensorData::new(elems.to_vec(), [A])
    }
}

impl<E: Element> From<&[E]> for TensorData<1> {
    fn from(elems: &[E]) -> Self {
        let mut data = Vec::with_capacity(elems.len());
        for elem in elems.iter() {
            data.push(*elem);
        }

        TensorData::new(data, [elems.len()])
    }
}

impl<E: Element, const A: usize, const B: usize> From<[[E; B]; A]> for TensorData<2> {
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
    for TensorData<3>
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
    From<[[[[E; D]; C]; B]; A]> for TensorData<4>
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

impl<const D: usize> core::fmt::Display for TensorData<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("{:?}", &self.value).as_str())
    }
}

/// Data structure for serializing and deserializing tensor data.
#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq, Eq, Clone, new)]
// #[deprecated(
//     since = "0.14.0",
//     note = "the internal data format has changed, please use `TensorData` instead"
// )]
pub struct DataSerialize<E> {
    /// The values of the tensor.
    pub value: Vec<E>,
    /// The shape of the tensor.
    pub shape: Vec<usize>,
}

/// Data structure for tensors.
#[derive(new, Debug, Clone, PartialEq, Eq)]
// #[deprecated(
//     since = "0.14.0",
//     note = "the internal data format has changed, please use `TensorData` instead"
// )]
pub struct Data<E, const D: usize> {
    /// The values of the tensor.
    pub value: Vec<E>,

    /// The shape of the tensor.
    pub shape: Shape<D>,
}

#[allow(deprecated)]
impl<const D: usize, E: Element> Data<E, D> {
    /// Converts the data to a different element type.
    pub fn convert<EOther: Element>(self) -> Data<EOther, D> {
        let value: Vec<EOther> = self.value.into_iter().map(|a| a.elem()).collect();

        Data {
            value,
            shape: self.shape,
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
    pub fn assert_within_range<EOther: Element>(&self, range: core::ops::Range<EOther>) {
        let start = range.start.elem::<f32>();
        let end = range.end.elem::<f32>();

        for elem in self.value.iter() {
            let elem = elem.elem::<f32>();
            if elem < start || elem >= end {
                panic!("Element ({elem:?}) is not within range {range:?}");
            }
        }
    }
}

#[allow(deprecated)]
impl<E: Element> DataSerialize<E> {
    /// Converts the data to a different element type.
    pub fn convert<EOther: Element>(self) -> DataSerialize<EOther> {
        if TypeId::of::<E>() == TypeId::of::<EOther>() {
            let cast: Box<dyn Any> = Box::new(self);
            let cast: Box<DataSerialize<EOther>> = cast.downcast().unwrap();
            return *cast;
        }

        let value: Vec<EOther> = self.value.into_iter().map(|a| a.elem()).collect();

        DataSerialize {
            value,
            shape: self.shape,
        }
    }

    /// Creates a new [DataSerialize] struct from [TensorData].
    /// Used for backward compatibility.
    pub fn from_tensor_data<const D: usize>(data: TensorData<D>) -> Self {
        Self {
            value: data.to_vec().unwrap(),
            shape: data.shape.dims.to_vec(),
        }
    }

    /// Converts the data to the new [TensorData] format.
    pub fn into_tensor_data<const D: usize>(self) -> TensorData<D> {
        assert_eq!(self.shape.len(), D);

        TensorData::new(self.value, self.shape)
    }
}

#[allow(deprecated)]
impl<E: Element, const D: usize> Data<E, D> {
    /// Populates the data with random values.
    pub fn random<R: RngCore>(shape: Shape<D>, distribution: Distribution, rng: &mut R) -> Self {
        let num_elements = shape.num_elements();
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(E::random(distribution, rng));
        }

        Data::new(data, shape)
    }
}

#[allow(deprecated)]
impl<E: core::fmt::Debug, const D: usize> Data<E, D>
where
    E: Element,
{
    /// Populates the data with zeros.
    pub fn zeros<S: Into<Shape<D>>>(shape: S) -> Data<E, D> {
        let shape = shape.into();
        let num_elements = shape.num_elements();
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(0.elem());
        }

        Data::new(data, shape)
    }
}

#[allow(deprecated)]
impl<E: core::fmt::Debug, const D: usize> Data<E, D>
where
    E: Element,
{
    /// Populates the data with ones.
    pub fn ones(shape: Shape<D>) -> Data<E, D> {
        let num_elements = shape.num_elements();
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(1.elem());
        }

        Data::new(data, shape)
    }
}

#[allow(deprecated)]
impl<E: core::fmt::Debug, const D: usize> Data<E, D>
where
    E: Element,
{
    /// Populates the data with the given value
    pub fn full(shape: Shape<D>, fill_value: E) -> Data<E, D> {
        let num_elements = shape.num_elements();
        let mut data = Vec::with_capacity(num_elements);
        for _ in 0..num_elements {
            data.push(fill_value)
        }

        Data::new(data, shape)
    }
}

#[allow(deprecated)]
impl<E: core::fmt::Debug + Copy, const D: usize> Data<E, D> {
    /// Serializes the data.
    ///
    /// # Returns
    ///
    /// The serialized data.
    pub fn serialize(&self) -> DataSerialize<E> {
        DataSerialize {
            value: self.value.clone(),
            shape: self.shape.dims.to_vec(),
        }
    }
}

#[allow(deprecated)]
impl<E: Into<f64> + Clone + core::fmt::Debug + PartialEq, const D: usize> Data<E, D> {
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
                self.shape.dims, other.shape.dims
            )
            .as_str();
        }

        let iter = self.value.clone().into_iter().zip(other.value.clone());

        let mut num_diff = 0;
        let max_num_diff = 5;

        for (i, (a, b)) in iter.enumerate() {
            let a: f64 = a.into();
            let b: f64 = b.into();

            //if they are both nan, then they are equally nan
            let both_nan = a.is_nan() && b.is_nan();
            //this works for both infinities
            let both_inf = a.is_infinite() && b.is_infinite() && ((a > 0.) == (b > 0.));

            if both_nan || both_inf {
                continue;
            }

            let err = ((a - b).pow(2.0f64)).sqrt();

            if err > tolerance || err.is_nan() {
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
}

#[allow(deprecated)]
impl<const D: usize> Data<usize, D> {
    /// Converts the usize data to a different element type.
    pub fn from_usize<O: num_traits::FromPrimitive>(self) -> Data<O, D> {
        let value: Vec<O> = self
            .value
            .into_iter()
            .map(|a| num_traits::FromPrimitive::from_usize(a).unwrap())
            .collect();

        Data {
            value,
            shape: self.shape,
        }
    }
}

#[allow(deprecated)]
impl<E: Clone, const D: usize> From<&DataSerialize<E>> for Data<E, D> {
    fn from(data: &DataSerialize<E>) -> Self {
        let mut dims = [0; D];
        dims[..D].copy_from_slice(&data.shape[..D]);
        Data::new(data.value.clone(), Shape::new(dims))
    }
}

#[allow(deprecated)]
impl<E, const D: usize> From<DataSerialize<E>> for Data<E, D> {
    fn from(data: DataSerialize<E>) -> Self {
        let mut dims = [0; D];
        dims[..D].copy_from_slice(&data.shape[..D]);
        Data::new(data.value, Shape::new(dims))
    }
}

#[allow(deprecated)]
impl<E: core::fmt::Debug + Copy, const A: usize> From<[E; A]> for Data<E, 1> {
    fn from(elems: [E; A]) -> Self {
        let mut data = Vec::with_capacity(2 * A);
        for elem in elems.into_iter() {
            data.push(elem);
        }

        Data::new(data, Shape::new([A]))
    }
}

#[allow(deprecated)]
impl<E: core::fmt::Debug + Copy> From<&[E]> for Data<E, 1> {
    fn from(elems: &[E]) -> Self {
        let mut data = Vec::with_capacity(elems.len());
        for elem in elems.iter() {
            data.push(*elem);
        }

        Data::new(data, Shape::new([elems.len()]))
    }
}

#[allow(deprecated)]
impl<E: core::fmt::Debug + Copy, const A: usize, const B: usize> From<[[E; B]; A]> for Data<E, 2> {
    fn from(elems: [[E; B]; A]) -> Self {
        let mut data = Vec::with_capacity(A * B);
        for elem in elems.into_iter().take(A) {
            for elem in elem.into_iter().take(B) {
                data.push(elem);
            }
        }

        Data::new(data, Shape::new([A, B]))
    }
}

#[allow(deprecated)]
impl<E: core::fmt::Debug + Copy, const A: usize, const B: usize, const C: usize>
    From<[[[E; C]; B]; A]> for Data<E, 3>
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

        Data::new(data, Shape::new([A, B, C]))
    }
}

#[allow(deprecated)]
impl<
        E: core::fmt::Debug + Copy,
        const A: usize,
        const B: usize,
        const C: usize,
        const D: usize,
    > From<[[[[E; D]; C]; B]; A]> for Data<E, 4>
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

        Data::new(data, Shape::new([A, B, C, D]))
    }
}

#[allow(deprecated)]
impl<E: core::fmt::Debug, const D: usize> core::fmt::Display for Data<E, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("{:?}", &self.value).as_str())
    }
}

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn should_have_right_num_elements() {
        let shape = Shape::new([3, 5, 6]);
        let num_elements = shape.num_elements();
        let data =
            TensorData::random::<f32, _>(shape, Distribution::Default, &mut StdRng::from_entropy());

        assert_eq!(num_elements, data.value.len());
    }

    #[test]
    fn should_have_right_shape() {
        let data = TensorData::from([[3.0, 5.0, 6.0]]);
        assert_eq!(data.shape, Shape::new([1, 3]));

        let data = TensorData::from([[4.0, 5.0, 8.0], [3.0, 5.0, 6.0]]);
        assert_eq!(data.shape, Shape::new([2, 3]));

        let data = TensorData::from([3.0, 5.0, 6.0]);
        assert_eq!(data.shape, Shape::new([3]));
    }

    #[test]
    fn should_assert_appox_eq_limit() {
        let data1 = TensorData::from([[3.0, 5.0, 6.0]]);
        let data2 = TensorData::from([[3.01, 5.0, 6.0]]);

        data1.assert_approx_eq(&data2, 2);
    }

    #[test]
    #[should_panic]
    fn should_assert_appox_eq_above_limit() {
        let data1 = TensorData::from([[3.0, 5.0, 6.0]]);
        let data2 = TensorData::from([[3.011, 5.0, 6.0]]);

        data1.assert_approx_eq(&data2, 2);
    }

    #[test]
    #[should_panic]
    fn should_assert_appox_eq_check_shape() {
        let data1 = TensorData::from([[3.0, 5.0, 6.0, 7.0]]);
        let data2 = TensorData::from([[3.0, 5.0, 6.0]]);

        data1.assert_approx_eq(&data2, 2);
    }

    // #[test]
    // fn tensor_data_should_have_right_num_elements() {
    //     let shape = Shape::new([3, 5, 6]);
    //     let num_elements = shape.num_elements();
    //     let data =
    //         TensorData::new(shape, Distribution::Default, &mut StdRng::from_entropy());

    //     assert_eq!(num_elements, data.value.len());
    // }
}
