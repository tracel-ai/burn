use super::ops::{Ones, Zeros};
use crate::{tensor::Shape, Element, ElementConversion};
use rand::{distributions::Standard, prelude::StdRng, Rng};

#[derive(serde::Serialize, serde::Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct DataSerialize<P> {
    pub value: Vec<P>,
    pub shape: Vec<usize>,
}

#[derive(new, Debug, Clone, PartialEq, Eq)]
pub struct Data<P, const D: usize> {
    pub value: Vec<P>,
    pub shape: Shape<D>,
}

#[derive(Clone, Copy)]
pub enum Distribution<P> {
    Standard,
    Bernoulli(f64),
    Uniform(P, P),
    Normal(f64, f64),
}

#[derive(new)]
pub struct DistributionSampler<'a, P>
where
    Standard: rand::distributions::Distribution<P>,
    P: rand::distributions::uniform::SampleUniform,
{
    kind: DistributionSamplerKind<P>,
    rng: &'a mut StdRng,
}

pub enum DistributionSamplerKind<P>
where
    Standard: rand::distributions::Distribution<P>,
    P: rand::distributions::uniform::SampleUniform,
{
    Standard(rand::distributions::Standard),
    Uniform(rand::distributions::Uniform<P>),
    Bernoulli(rand::distributions::Bernoulli),
    Normal(statrs::distribution::Normal),
}

impl<'a, P> DistributionSampler<'a, P>
where
    Standard: rand::distributions::Distribution<P>,
    P: rand::distributions::uniform::SampleUniform,
    P: Element,
{
    pub fn sample(&mut self) -> P {
        match &self.kind {
            DistributionSamplerKind::Standard(distribution) => self.rng.sample(distribution),
            DistributionSamplerKind::Uniform(distribution) => self.rng.sample(distribution),
            DistributionSamplerKind::Bernoulli(distribution) => {
                if self.rng.sample(distribution) {
                    P::ones(&P::default())
                } else {
                    P::zeros(&P::default())
                }
            }
            DistributionSamplerKind::Normal(distribution) => {
                self.rng.sample(distribution).to_elem()
            }
        }
    }
}

impl<P> Distribution<P>
where
    Standard: rand::distributions::Distribution<P>,
    P: rand::distributions::uniform::SampleUniform,
{
    pub fn sampler(self, rng: &'_ mut StdRng) -> DistributionSampler<'_, P> {
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
            Distribution::Normal(mean, std) => DistributionSamplerKind::Normal(
                statrs::distribution::Normal::new(mean, std).unwrap(),
            ),
        };

        DistributionSampler::new(kind, rng)
    }
}

impl<P> Distribution<P>
where
    P: Element,
{
    pub fn convert<E: Element>(self) -> Distribution<E> {
        match self {
            Distribution::Standard => Distribution::Standard,
            Distribution::Uniform(a, b) => Distribution::Uniform(E::from_elem(a), E::from_elem(b)),
            Distribution::Bernoulli(prob) => Distribution::Bernoulli(prob),
            Distribution::Normal(mean, std) => Distribution::Normal(mean, std),
        }
    }
}

impl<const D: usize, P: Element> Data<P, D> {
    pub fn convert<E: Element>(self) -> Data<E, D> {
        let value: Vec<E> = self.value.into_iter().map(|a| a.to_elem()).collect();

        Data {
            value,
            shape: self.shape,
        }
    }
}

impl<P: Element> DataSerialize<P> {
    pub fn convert<E: Element>(self) -> DataSerialize<E> {
        let value: Vec<E> = self.value.into_iter().map(|a| a.to_elem()).collect();

        DataSerialize {
            value,
            shape: self.shape,
        }
    }
}

impl<const D: usize> Data<bool, D> {
    pub fn convert<E: Element>(self) -> Data<E, D> {
        let value: Vec<E> = self
            .value
            .into_iter()
            .map(|a| (a as i64).to_elem())
            .collect();

        Data {
            value,
            shape: self.shape,
        }
    }
}
impl<P: Element, const D: usize> Data<P, D> {
    pub fn random(shape: Shape<D>, distribution: Distribution<P>, rng: &mut StdRng) -> Self {
        let num_elements = shape.num_elements();
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(P::random(distribution, rng));
        }

        Data::new(data, shape)
    }
}
impl<P: std::fmt::Debug, const D: usize> Data<P, D>
where
    P: Zeros + Default,
{
    pub fn zeros<S: Into<Shape<D>>>(shape: S) -> Data<P, D> {
        let shape = shape.into();
        let elem = P::default();
        let num_elements = shape.num_elements();
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(elem.zeros());
        }

        Data::new(data, shape)
    }
    pub fn zeros_(shape: Shape<D>, _kind: P) -> Data<P, D> {
        Self::zeros(shape)
    }
}

impl<P: std::fmt::Debug, const D: usize> Data<P, D>
where
    P: Ones + Default,
{
    pub fn ones(shape: Shape<D>) -> Data<P, D> {
        let elem = P::default();
        let num_elements = shape.num_elements();
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(elem.ones());
        }

        Data::new(data, shape)
    }
    pub fn ones_(shape: Shape<D>, _kind: P) -> Data<P, D> {
        Self::ones(shape)
    }
}

impl<P: std::fmt::Debug + Copy, const D: usize> Data<P, D> {
    pub fn serialize(&self) -> DataSerialize<P> {
        DataSerialize {
            value: self.value.clone(),
            shape: self.shape.dims.to_vec(),
        }
    }
}

impl<P: Into<f64> + Clone + std::fmt::Debug + PartialEq, const D: usize> Data<P, D> {
    pub fn assert_approx_eq(&self, other: &Self, precision: usize) {
        assert_eq!(self.shape, other.shape);

        let mut eq = true;

        let iter = self
            .value
            .clone()
            .into_iter()
            .zip(other.value.clone().into_iter());

        for (a, b) in iter {
            let a: f64 = a.into();
            let b: f64 = b.into();
            let a = f64::round(10.0_f64.powi(precision as i32) * a);
            let b = f64::round(10.0_f64.powi(precision as i32) * b);

            if a != b {
                println!("a {:?}, b {:?}", a, b);
                eq = false;
            }
        }

        if !eq {
            assert_eq!(self.value, other.value);
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

impl<P: Clone, const D: usize> From<&DataSerialize<P>> for Data<P, D> {
    fn from(data: &DataSerialize<P>) -> Self {
        let mut dims = [0; D];
        dims[..D].copy_from_slice(&data.shape[..D]);
        Data::new(data.value.clone(), Shape::new(dims))
    }
}

impl<P, const D: usize> From<DataSerialize<P>> for Data<P, D> {
    fn from(data: DataSerialize<P>) -> Self {
        let mut dims = [0; D];
        dims[..D].copy_from_slice(&data.shape[..D]);
        Data::new(data.value, Shape::new(dims))
    }
}

impl<P: std::fmt::Debug + Copy, const A: usize> From<[P; A]> for Data<P, 1> {
    fn from(elems: [P; A]) -> Self {
        let mut data = Vec::with_capacity(2 * A);
        for elem in elems.into_iter().take(A) {
            data.push(elem);
        }

        Data::new(data, Shape::new([A]))
    }
}

impl<P: std::fmt::Debug + Copy, const A: usize, const B: usize> From<[[P; B]; A]> for Data<P, 2> {
    fn from(elems: [[P; B]; A]) -> Self {
        let mut data = Vec::with_capacity(A * B);
        for elem in elems.into_iter().take(A) {
            for elem in elem.into_iter().take(B) {
                data.push(elem);
            }
        }

        Data::new(data, Shape::new([A, B]))
    }
}

impl<P: std::fmt::Debug + Copy, const A: usize, const B: usize, const C: usize>
    From<[[[P; C]; B]; A]> for Data<P, 3>
{
    fn from(elems: [[[P; C]; B]; A]) -> Self {
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

impl<P: std::fmt::Debug, const D: usize> std::fmt::Display for Data<P, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("{:?}", &self.value).as_str())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn should_have_right_num_elements() {
        let shape = Shape::new([3, 5, 6]);
        let data =
            Data::<f32, 3>::random(shape, Distribution::Standard, &mut StdRng::from_entropy());
        assert_eq!(shape.num_elements(), data.value.len());
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
}
