use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Defines a source for a `StdRng`.
///
/// # Examples
///
/// ```rust,no_run
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
/// use burn_dataset::transform::RngSource;
///
/// // Default via OS randomness (`RngSource::Default`)
/// let system: RngSource = RngSource::default();
///
/// // From a fixed seed (`RngSource::Seed`)
/// let seeded: RngSource = 42.into();
///
/// // From an existing rng (`RngSource::Rng`)
/// let mut rng = StdRng::seed_from_u64(123);
/// let with_rng: RngSource = (&mut rng).into();
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RngSource {
    /// Build a new rng from the system.
    #[default]
    Default,

    /// The rng is passed as a seed.
    Seed(u64),

    /// The rng state is captured as seed bytes.
    Rng([u8; 32]),
}

impl From<RngSource> for StdRng {
    fn from(source: RngSource) -> Self {
        match source {
            RngSource::Default => rand::make_rng(),
            RngSource::Rng(seed) => StdRng::from_seed(seed),
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
    fn from(mut rng: StdRng) -> Self {
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        Self::Rng(seed)
    }
}

/// Advances the RNG and captures its state as seed bytes.
impl From<&mut StdRng> for RngSource {
    fn from(rng: &mut StdRng) -> Self {
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        Self::Rng(seed)
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

        // Exercise the make_rng() path
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
        // From StdRng (owned).
        {
            let rng = StdRng::seed_from_u64(42);
            let rng_source = RngSource::from(rng);
            matches!(rng_source, RngSource::Rng(_));

            // Converting back should produce a deterministic StdRng.
            let rng_a: StdRng = rng_source.clone().into();
            let rng_b: StdRng = rng_source.into();
            assert_eq!(rng_a, rng_b);
        }

        // From &mut StdRng (advances the original).
        {
            let mut rng = StdRng::seed_from_u64(42);
            let original = StdRng::seed_from_u64(42);

            let rng_source = RngSource::from(&mut rng);
            // Original rng was advanced by fill_bytes
            assert_ne!(rng, original);
            matches!(rng_source, RngSource::Rng(_));
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
