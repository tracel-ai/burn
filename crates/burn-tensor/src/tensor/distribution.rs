use rand::{distr::StandardUniform, Rng, RngCore};

use crate::{Element, ElementConversion};

/// Distribution for random value of a tensor.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Distribution {
    /// Uniform distribution from 0 (inclusive) to 1 (exclusive).
    Default,

    /// Bernoulli distribution with the given probability.
    Bernoulli(f64),

    /// Uniform distribution. The range is inclusive.
    Uniform(f64, f64),

    /// Normal distribution with the given mean and standard deviation.
    Normal(f64, f64),

    /// Multinomial distribution with the given probabilities.
    Multinomial(Vec<f64>),
}

/// Distribution sampler for random value of a tensor.
#[derive(new)]
pub struct DistributionSampler<'a, E, R>
where
    StandardUniform: rand::distr::Distribution<E>,
    E: rand::distr::uniform::SampleUniform + std::cmp::PartialOrd,
    R: RngCore,
{
    kind: DistributionSamplerKind<E>,
    rng: &'a mut R,
}

/// Distribution sampler kind for random value of a tensor.
pub enum DistributionSamplerKind<E>
where
    StandardUniform: rand::distr::Distribution<E>,
    E: rand::distr::uniform::SampleUniform + std::cmp::PartialOrd,
{
    /// Standard distribution.
    Standard(rand::distr::StandardUniform),

    /// Uniform distribution.
    Uniform(rand::distr::Uniform<E>),

    /// Bernoulli distribution.
    Bernoulli(rand::distr::Bernoulli),

    /// Normal distribution.
    Normal(rand_distr::Normal<f64>),

    /// Multinomial (categorical) distribution.
    Multinomial(rand::distr::weighted::WeightedIndex<f64>),
}

impl<E, R> DistributionSampler<'_, E, R>
where
    StandardUniform: rand::distr::Distribution<E>,
    E: rand::distr::uniform::SampleUniform,
    E: Element,
    E: std::cmp::PartialOrd,
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
            DistributionSamplerKind::Multinomial(distribution) => (self.rng.sample(distribution) as f64).elem(),
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
        E: Element + rand::distr::uniform::SampleUniform + std::cmp::PartialOrd,
        StandardUniform: rand::distr::Distribution<E>,
    {
        let kind = match self {
            Distribution::Default => {
                DistributionSamplerKind::Standard(rand::distr::StandardUniform {})
            }
            Distribution::Uniform(low, high) => DistributionSamplerKind::Uniform(
                rand::distr::Uniform::new(low.elem::<E>(), high.elem::<E>()).unwrap(),
            ),
            Distribution::Bernoulli(prob) => {
                DistributionSamplerKind::Bernoulli(rand::distr::Bernoulli::new(prob).unwrap())
            }
            Distribution::Normal(mean, std) => {
                DistributionSamplerKind::Normal(rand_distr::Normal::new(mean, std).unwrap())
            }
            Distribution::Multinomial(vec) => 
                DistributionSamplerKind::Multinomial(rand::distr::weighted::WeightedIndex::new(vec).unwrap()),

        };

        DistributionSampler::new(kind, rng)
    }
}
