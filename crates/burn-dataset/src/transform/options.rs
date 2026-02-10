use rand::prelude::StdRng;
use rand::rngs::SysRng;
use rand::{Rng, SeedableRng};

/// Defines a source for a `StdRng`.
///
/// # Examples
///
/// ```rust,no_run
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
/// use burn_dataset::transform::RngSource;
///
/// // Default via `StdRng::from_os_rng()` (`RngSource::Default`)
/// let system: RngSource = RngSource::default();
///
/// // From a fixed seed (`RngSource::Seed`)
/// let seeded: RngSource = 42.into();
///
/// // From an existing rng (`RngSource::Rng`)
/// let rng = StdRng::seed_from_u64(123);
/// let with_rng: RngSource = rng.into();
///
/// // Advances the original RNG and then clones its new state
/// let mut rng = StdRng::seed_from_u64(123);
/// let stateful: RngSource = (&mut rng).into();
/// ```
#[derive(Debug, Default, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum RngSource {
    /// Build a new rng from the system.
    #[default]
    Default,

    /// The rng is passed as a seed.
    Seed(u64),

    /// The rng is passed as an option.
    Rng(StdRng),
}

impl From<RngSource> for StdRng {
    fn from(source: RngSource) -> Self {
        match source {
            RngSource::Default => StdRng::try_from_rng(&mut SysRng).unwrap(),
            RngSource::Rng(mut rng) => rng.fork(),
            RngSource::Seed(seed) => StdRng::seed_from_u64(seed),
        }
    }
}

impl From<u64> for RngSource {
    fn from(seed: u64) -> Self {
        Self::Seed(seed)
    }
}

impl From<StdRng> for RngSource {
    fn from(rng: StdRng) -> Self {
        Self::Rng(rng)
    }
}

/// Users calling with a mutable rng expect state advancement,
/// So conversion from `&mut StdRng` advances the rng before cloning.
impl From<&mut StdRng> for RngSource {
    fn from(rng: &mut StdRng) -> Self {
        rng.next_u64();
        Self::Rng(rng.fork())
    }
}

/// Helper option to describe the size of a wrapper, relative to a wrapped object.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum SizeConfig {
    /// Use the size of the source dataset.
    #[default]
    Default,

    /// Use the size as a ratio of the source dataset size.
    ///
    /// Must be >= 0.
    Ratio(f64),

    /// Use a fixed size.
    Fixed(usize),
}

impl SizeConfig {
    /// Construct a source which will have the same size as the source dataset.
    pub fn source() -> Self {
        Self::Default
    }

    /// Resolve the effective size.
    ///
    /// ## Arguments
    ///
    /// - `source_size`: the size of the source dataset.
    ///
    /// ## Returns
    ///
    /// The resolved size of the wrapper dataset.
    pub fn resolve(self, source_size: usize) -> usize {
        match self {
            SizeConfig::Default => source_size,
            SizeConfig::Ratio(ratio) => {
                assert!(ratio >= 0.0, "Ratio must be positive: {ratio}");
                ((source_size as f64) * ratio) as usize
            }
            SizeConfig::Fixed(size) => size,
        }
    }
}

impl From<usize> for SizeConfig {
    fn from(size: usize) -> Self {
        Self::Fixed(size)
    }
}

impl From<f64> for SizeConfig {
    fn from(ratio: f64) -> Self {
        Self::Ratio(ratio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_rng_source_default() {
        let rng_source: RngSource = Default::default();
        assert_eq!(&rng_source, &RngSource::Default);
        assert_eq!(&rng_source, &RngSource::default());

        // Exercise the from_os_rng() call; but we don't know its seed;
        let _rng: StdRng = rng_source.into();
    }

    #[test]
    fn test_rng_source_seed() {
        let rng_source = RngSource::from(42);
        assert_eq!(&rng_source, &RngSource::Seed(42));

        let rng: StdRng = rng_source.into();
        let expected = StdRng::seed_from_u64(42);

        assert_eq!(rng, expected);
    }

    #[test]
    fn test_rng_source_rng() {
        let mut original = StdRng::seed_from_u64(42);

        // From StdRng.
        {
            let rng_source = RngSource::from(original.fork());
            let rng: StdRng = rng_source.into();
            assert_eq!(rng, original);
        }

        // From &mut StdRng.
        {
            let mut stateful = original.fork();

            let rng_source = RngSource::from(&mut stateful);
            assert_ne!(stateful, original);

            // Advance the rng.
            let rng: StdRng = rng_source.into();
            assert_eq!(rng, stateful);
        }
    }

    #[test]
    fn test_size_config() {
        assert_eq!(SizeConfig::default(), SizeConfig::Default);

        assert_eq!(SizeConfig::from(42), SizeConfig::Fixed(42));

        assert_eq!(SizeConfig::from(1.5), SizeConfig::Ratio(1.5));

        assert_eq!(SizeConfig::source(), SizeConfig::Default);
        assert_eq!(SizeConfig::source().resolve(50), 50);
    }
}
