use core::any::{Any, TypeId};

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::{tensor::Shape, Element, ElementConversion};

use num_traits::pow::Pow;

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

use rand::{distributions::Standard, Rng, RngCore};

/// Data structure for serializing and deserializing tensor data.
#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq, Eq, Clone, new)]
pub struct DataSerialize<E> {
    /// The values of the tensor.
    pub value: Vec<E>,
    /// The shape of the tensor.
    pub shape: Vec<usize>,
}

/// Data structure for tensors.
#[derive(new, Debug, Clone, PartialEq, Eq)]
pub struct Data<E, const D: usize> {
    /// The values of the tensor.
    pub value: Vec<E>,

    /// The shape of the tensor.
    pub shape: Shape<D>,
}

/// Distribution for random value of a tensor.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Distribution {
    /// Uniform distribution from 0 (inclusive) to 1 (exclusive).
    Default,

    /// Bernoulli distribution with the given probability.
    Bernoulli(f64),

    /// Uniform distribution. The range is inclusive.
    Uniform(f64, f64),

    /// Normal distribution with the given mean and standard deviation.
    Normal(f64, f64),
}

/// Distribution sampler for random value of a tensor.
#[derive(new)]
pub struct DistributionSampler<'a, E, R>
where
    Standard: rand::distributions::Distribution<E>,
    E: rand::distributions::uniform::SampleUniform,
    R: RngCore,
{
    kind: DistributionSamplerKind<E>,
    rng: &'a mut R,
}

/// Distribution sampler kind for random value of a tensor.
pub enum DistributionSamplerKind<E>
where
    Standard: rand::distributions::Distribution<E>,
    E: rand::distributions::uniform::SampleUniform,
{
    /// Standard distribution.
    Standard(rand::distributions::Standard),

    /// Uniform distribution.
    Uniform(rand::distributions::Uniform<E>),

    /// Bernoulli distribution.
    Bernoulli(rand::distributions::Bernoulli),

    /// Normal distribution.
    Normal(rand_distr::Normal<f64>),
}

impl<'a, E, R> DistributionSampler<'a, E, R>
where
    Standard: rand::distributions::Distribution<E>,
    E: rand::distributions::uniform::SampleUniform,
    E: Element,
    R: RngCore,
{
    /// Sames a random value from the distribution.
    pub fn sample(&mut self) -> E {
        match &self.kind {
            DistributionSamplerKind::Standard(distribution) => self.rng.sample(distribution),
            DistributionSamplerKind::Uniform(distribution) => self.rng.sample(distribution),
            DistributionSamplerKind::Bernoulli(distribution) => {
                if self.rng.sample(distribution) {
                    1.elem()
                } else {
                    0.elem()
                }
            }
            DistributionSamplerKind::Normal(distribution) => self.rng.sample(distribution).elem(),
        }
    }
}

impl Distribution {
    /// Creates a new distribution sampler.
    ///
    /// # Arguments
    ///
    /// * `rng` - The random number generator.
    ///
    /// # Returns
    ///
    /// The distribution sampler.
    pub fn sampler<R, E>(self, rng: &'_ mut R) -> DistributionSampler<'_, E, R>
    where
        R: RngCore,
        E: Element + rand::distributions::uniform::SampleUniform,
        Standard: rand::distributions::Distribution<E>,
    {
        let kind = match self {
            Distribution::Default => {
                DistributionSamplerKind::Standard(rand::distributions::Standard {})
            }
            Distribution::Uniform(low, high) => DistributionSamplerKind::Uniform(
                rand::distributions::Uniform::new(low.elem::<E>(), high.elem::<E>()),
            ),
            Distribution::Bernoulli(prob) => DistributionSamplerKind::Bernoulli(
                rand::distributions::Bernoulli::new(prob).unwrap(),
            ),
            Distribution::Normal(mean, std) => {
                DistributionSamplerKind::Normal(rand_distr::Normal::new(mean, std).unwrap())
            }
        };

        DistributionSampler::new(kind, rng)
    }
}

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
}

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

impl<E: Clone, const D: usize> From<&DataSerialize<E>> for Data<E, D> {
    fn from(data: &DataSerialize<E>) -> Self {
        let mut dims = [0; D];
        dims[..D].copy_from_slice(&data.shape[..D]);
        Data::new(data.value.clone(), Shape::new(dims))
    }
}

impl<E, const D: usize> From<DataSerialize<E>> for Data<E, D> {
    fn from(data: DataSerialize<E>) -> Self {
        let mut dims = [0; D];
        dims[..D].copy_from_slice(&data.shape[..D]);
        Data::new(data.value, Shape::new(dims))
    }
}

impl<E: core::fmt::Debug + Copy, const A: usize> From<[E; A]> for Data<E, 1> {
    fn from(elems: [E; A]) -> Self {
        let mut data = Vec::with_capacity(2 * A);
        for elem in elems.into_iter() {
            data.push(elem);
        }

        Data::new(data, Shape::new([A]))
    }
}

impl<E: core::fmt::Debug + Copy> From<&[E]> for Data<E, 1> {
    fn from(elems: &[E]) -> Self {
        let mut data = Vec::with_capacity(elems.len());
        for elem in elems.iter() {
            data.push(*elem);
        }

        Data::new(data, Shape::new([elems.len()]))
    }
}

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

impl<E: core::fmt::Debug, const D: usize> core::fmt::Display for Data<E, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("{:?}", &self.value).as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn should_have_right_num_elements() {
        let shape = Shape::new([3, 5, 6]);
        let num_elements = shape.num_elements();
        let data =
            Data::<f32, 3>::random(shape, Distribution::Default, &mut StdRng::from_entropy());

        assert_eq!(num_elements, data.value.len());
    }

    #[test]
    fn should_have_right_shape() {
        let data = Data::from([[3.0, 5.0, 6.0]]);
        assert_eq!(data.shape, Shape::new([1, 3]));

        let data = Data::from([[4.0, 5.0, 8.0], [3.0, 5.0, 6.0]]);
        assert_eq!(data.shape, Shape::new([2, 3]));

        let data = Data::from([3.0, 5.0, 6.0]);
        assert_eq!(data.shape, Shape::new([3]));
    }

    #[test]
    fn should_assert_appox_eq_limit() {
        let data1 = Data::<f32, 2>::from([[3.0, 5.0, 6.0]]);
        let data2 = Data::<f32, 2>::from([[3.01, 5.0, 6.0]]);

        data1.assert_approx_eq(&data2, 2);
    }

    #[test]
    #[should_panic]
    fn should_assert_appox_eq_above_limit() {
        let data1 = Data::<f32, 2>::from([[3.0, 5.0, 6.0]]);
        let data2 = Data::<f32, 2>::from([[3.011, 5.0, 6.0]]);

        data1.assert_approx_eq(&data2, 2);
    }

    #[test]
    #[should_panic]
    fn should_assert_appox_eq_check_shape() {
        let data1 = Data::<f32, 2>::from([[3.0, 5.0, 6.0, 7.0]]);
        let data2 = Data::<f32, 2>::from([[3.0, 5.0, 6.0]]);

        data1.assert_approx_eq(&data2, 2);
    }
}
