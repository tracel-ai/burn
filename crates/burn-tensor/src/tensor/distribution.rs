use rand::{distributions::Standard, Rng, RngCore};

use crate::{Element, ElementConversion};

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
