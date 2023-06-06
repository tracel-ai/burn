use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::{tensor::Shape, Element, ElementConversion};

use rand::{distributions::Standard, Rng, RngCore};

#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq, Eq, Clone, new)]
pub struct DataSerialize<E> {
    pub value: Vec<E>,
    pub shape: Vec<usize>,
}

#[derive(new, Debug, Clone, PartialEq, Eq)]
pub struct Data<E, const D: usize> {
    pub value: Vec<E>,
    pub shape: Shape<D>,
}

#[derive(Clone, Copy)]
pub enum Distribution<E> {
    Standard,
    Bernoulli(f64),
    Uniform(E, E),
    Normal(f64, f64),
}

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

pub enum DistributionSamplerKind<E>
where
    Standard: rand::distributions::Distribution<E>,
    E: rand::distributions::uniform::SampleUniform,
{
    Standard(rand::distributions::Standard),
    Uniform(rand::distributions::Uniform<E>),
    Bernoulli(rand::distributions::Bernoulli),
    Normal(rand_distr::Normal<f64>),
}

impl<'a, E, R> DistributionSampler<'a, E, R>
where
    Standard: rand::distributions::Distribution<E>,
    E: rand::distributions::uniform::SampleUniform,
    E: Element,
    R: RngCore,
{
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

impl<E> Distribution<E>
where
    Standard: rand::distributions::Distribution<E>,
    E: rand::distributions::uniform::SampleUniform,
{
    pub fn sampler<R: RngCore>(self, rng: &'_ mut R) -> DistributionSampler<'_, E, R> {
        let kind = match self {
            Distribution::Standard => {
                DistributionSamplerKind::Standard(rand::distributions::Standard {})
            }
            Distribution::Uniform(low, high) => {
                DistributionSamplerKind::Uniform(rand::distributions::Uniform::new(low, high))
            }
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

impl<E> Distribution<E>
where
    E: Element,
{
    pub fn convert<EOther: Element>(self) -> Distribution<EOther> {
        match self {
            Distribution::Standard => Distribution::Standard,
            Distribution::Uniform(a, b) => {
                Distribution::Uniform(EOther::from_elem(a), EOther::from_elem(b))
            }
            Distribution::Bernoulli(prob) => Distribution::Bernoulli(prob),
            Distribution::Normal(mean, std) => Distribution::Normal(mean, std),
        }
    }
}

impl<const D: usize, E: Element> Data<E, D> {
    pub fn convert<EOther: Element>(self) -> Data<EOther, D> {
        let value: Vec<EOther> = self.value.into_iter().map(|a| a.elem()).collect();

        Data {
            value,
            shape: self.shape,
        }
    }
}

impl<E: Element> DataSerialize<E> {
    pub fn convert<EOther: Element>(self) -> DataSerialize<EOther> {
        let value: Vec<EOther> = self.value.into_iter().map(|a| a.elem()).collect();

        DataSerialize {
            value,
            shape: self.shape,
        }
    }
}

impl<const D: usize> Data<bool, D> {
    pub fn convert<E: Element>(self) -> Data<E, D> {
        let value: Vec<E> = self.value.into_iter().map(|a| (a as i64).elem()).collect();

        Data {
            value,
            shape: self.shape,
        }
    }
}
impl<E: Element, const D: usize> Data<E, D> {
    pub fn random<R: RngCore>(shape: Shape<D>, distribution: Distribution<E>, rng: &mut R) -> Self {
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
    pub fn zeros<S: Into<Shape<D>>>(shape: S) -> Data<E, D> {
        let shape = shape.into();
        let num_elements = shape.num_elements();
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(0.elem());
        }

        Data::new(data, shape)
    }
    pub fn zeros_(shape: Shape<D>, _kind: E) -> Data<E, D> {
        Self::zeros(shape)
    }
}

impl<E: core::fmt::Debug, const D: usize> Data<E, D>
where
    E: Element,
{
    pub fn ones(shape: Shape<D>) -> Data<E, D> {
        let num_elements = shape.num_elements();
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(1.elem());
        }

        Data::new(data, shape)
    }
    pub fn ones_(shape: Shape<D>, _kind: E) -> Data<E, D> {
        Self::ones(shape)
    }
}

impl<E: core::fmt::Debug + Copy, const D: usize> Data<E, D> {
    pub fn serialize(&self) -> DataSerialize<E> {
        DataSerialize {
            value: self.value.clone(),
            shape: self.shape.dims.to_vec(),
        }
    }
}

impl<E: Into<f64> + Clone + core::fmt::Debug + PartialEq, const D: usize> Data<E, D> {
    pub fn assert_approx_eq(&self, other: &Self, precision: usize) {
        let mut message = String::new();
        if self.shape != other.shape {
            message += format!(
                "\n  => Shape is different: {:?} != {:?}",
                self.shape.dims, other.shape.dims
            )
            .as_str();
        }

        let iter = self
            .value
            .clone()
            .into_iter()
            .zip(other.value.clone().into_iter());

        let mut num_diff = 0;
        let max_num_diff = 5;

        for (i, (a, b)) in iter.enumerate() {
            let a: f64 = a.into();
            let b: f64 = b.into();

            let err = libm::sqrt(libm::pow(a - b, 2.0));
            let tolerance = libm::pow(0.1, precision as f64);

            if err > tolerance {
                // Only print the first 5 differents values.
                if num_diff < max_num_diff {
                    message += format!(
                        "\n  => Position {i}: {a} != {b} | difference {err} > tolerance {tolerance}"
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

    pub fn assert_in_range(&self, min: E, max: E) {
        let min: f64 = min.into();
        let max: f64 = max.into();

        for item in self.value.iter() {
            let item: f64 = item.clone().into();

            if item < min || item > max {
                panic!("Element ({item}) is not within the range of ({min},{max})");
            }
        }
    }
}

impl<const D: usize> Data<usize, D> {
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
            Data::<f32, 3>::random(shape, Distribution::Standard, &mut StdRng::from_entropy());

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
