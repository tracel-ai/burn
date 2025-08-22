use rand::{Rng, RngCore, distr::StandardUniform};

use crate::{Element, ElementConversion};

/// Distribution for random value of a tensor.
#[derive(Debug, Default, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Distribution {
    /// Uniform distribution from 0 (inclusive) to 1 (exclusive).
    #[default]
    Default,

    /// Bernoulli distribution with the given probability.
    Bernoulli(f64),

    /// Uniform distribution `[low, high)`.
    Uniform(f64, f64),

    /// Normal distribution with the given mean and standard deviation.
    Normal(f64, f64),
}

/// Distribution sampler for random value of a tensor.
#[derive(new)]
pub struct DistributionSampler<'a, E, R>
where
    StandardUniform: rand::distr::Distribution<E>,
    E: rand::distr::uniform::SampleUniform,
    R: RngCore,
{
    kind: DistributionSamplerKind<E>,
    rng: &'a mut R,
}

/// Distribution sampler kind for random value of a tensor.
pub enum DistributionSamplerKind<E>
where
    StandardUniform: rand::distr::Distribution<E>,
    E: rand::distr::uniform::SampleUniform,
{
    /// Standard distribution.
    Standard(rand::distr::StandardUniform),

    /// Uniform distribution.
    Uniform(rand::distr::Uniform<E>),

    /// Bernoulli distribution.
    Bernoulli(rand::distr::Bernoulli),

    /// Normal distribution.
    Normal(rand_distr::Normal<f64>),
}

impl<E, R> DistributionSampler<'_, E, R>
where
    StandardUniform: rand::distr::Distribution<E>,
    E: rand::distr::uniform::SampleUniform,
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
        E: Element + rand::distr::uniform::SampleUniform,
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
        };

        DistributionSampler::new(kind, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_default() {
        let dist: Distribution = Default::default();

        assert_eq!(dist, Distribution::Default);
        assert_eq!(Distribution::default(), Distribution::Default);
    }
}
