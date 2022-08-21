use super::ops::{Ones, Zeros};
use crate::tensor::Shape;
use rand::{distributions::Standard, prelude::StdRng, Rng, SeedableRng};

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct DataSerialize<P> {
    pub value: Vec<P>,
    pub shape: Vec<usize>,
}

#[derive(new, Debug, Clone, PartialEq)]
pub struct Data<P, const D: usize> {
    pub value: Vec<P>,
    pub shape: Shape<D>,
}

pub enum Distribution<P> {
    Standard,
    Uniform(P, P),
}

pub struct DistributionSampler<P>
where
    Standard: rand::distributions::Distribution<P>,
    P: rand::distributions::uniform::SampleUniform,
{
    kind: DistributionSamplerKind<P>,
    rng: StdRng,
}

pub enum DistributionSamplerKind<P>
where
    Standard: rand::distributions::Distribution<P>,
    P: rand::distributions::uniform::SampleUniform,
{
    Standard(rand::distributions::Standard),
    Uniform(rand::distributions::Uniform<P>),
}

impl<P> DistributionSampler<P>
where
    Standard: rand::distributions::Distribution<P>,
    P: rand::distributions::uniform::SampleUniform,
{
    pub fn from_entropy(kind: DistributionSamplerKind<P>) -> Self {
        let rng = StdRng::from_entropy();

        Self { rng, kind }
    }

    pub fn sample(&mut self) -> P {
        match &self.kind {
            DistributionSamplerKind::Standard(distribution) => self.rng.sample(distribution),
            DistributionSamplerKind::Uniform(distribution) => self.rng.sample(distribution),
        }
    }
}

impl<P> Distribution<P>
where
    Standard: rand::distributions::Distribution<P>,
    P: rand::distributions::uniform::SampleUniform,
{
    pub fn sampler(self) -> DistributionSampler<P> {
        let kind = match self {
            Distribution::Standard => {
                DistributionSamplerKind::Standard(rand::distributions::Standard {})
            }
            Distribution::Uniform(low, high) => {
                DistributionSamplerKind::Uniform(rand::distributions::Uniform::new(low, high))
            }
        };

        DistributionSampler::from_entropy(kind)
    }
}

impl<P: std::fmt::Debug, const D: usize> Data<P, D>
where
    Standard: rand::distributions::Distribution<P>,
    P: rand::distributions::uniform::SampleUniform,
{
    pub fn random(shape: Shape<D>, distribution: Distribution<P>) -> Self {
        let num_elements = shape.num_elements();

        let mut sampler = distribution.sampler();
        let mut data = Vec::with_capacity(num_elements);

        for _ in 0..num_elements {
            data.push(sampler.sample());
        }

        Data::new(data, shape)
    }
    /// Usefull to force a kind
    pub fn random_(shape: Shape<D>, distribution: Distribution<P>, _kind: P) -> Self {
        Self::random(shape, distribution)
    }
}
impl<P: std::fmt::Debug, const D: usize> Data<P, D>
where
    P: Zeros<P> + Default,
{
    pub fn zeros(shape: Shape<D>) -> Data<P, D> {
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
    P: Ones<P> + Default,
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
        if self.shape != other.shape {
            return;
        }

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

impl<const D: usize> Data<f32, D> {
    pub fn from_f32<O: num_traits::FromPrimitive>(self) -> Data<O, D> {
        let value: Vec<O> = self
            .value
            .into_iter()
            .map(|a| num_traits::FromPrimitive::from_f32(a).unwrap())
            .collect();

        Data {
            value,
            shape: self.shape,
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
        for i in 0..D {
            dims[i] = data.shape[i];
        }

        Data::new(data.value.clone(), Shape::new(dims))
    }
}

impl<P, const D: usize> From<DataSerialize<P>> for Data<P, D> {
    fn from(data: DataSerialize<P>) -> Self {
        let mut dims = [0; D];
        for i in 0..D {
            dims[i] = data.shape[i];
        }

        Data::new(data.value, Shape::new(dims))
    }
}

impl<P: std::fmt::Debug + Copy, const A: usize> From<[P; A]> for Data<P, 1> {
    fn from(elems: [P; A]) -> Self {
        let mut data = Vec::with_capacity(2 * A);
        for i in 0..A {
            data.push(elems[i]);
        }

        Data::new(data, Shape::new([A]))
    }
}

impl<P: std::fmt::Debug + Copy, const A: usize, const B: usize> From<[[P; B]; A]> for Data<P, 2> {
    fn from(elems: [[P; B]; A]) -> Self {
        let mut data = Vec::with_capacity(A * B);
        for i in 0..A {
            for j in 0..B {
                data.push(elems[i][j]);
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
        for i in 0..A {
            for j in 0..B {
                for k in 0..C {
                    data.push(elems[i][j][k]);
                }
            }
        }

        Data::new(data, Shape::new([A, B, C]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_have_right_num_elements() {
        let shape = Shape::new([3, 5, 6]);
        let data = Data::<f32, 3>::random(shape.clone(), Distribution::Standard);
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
