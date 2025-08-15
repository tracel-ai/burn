use rand::SeedableRng;
use rand::prelude::StdRng;

/// Helper option to create a rng from a variety of options.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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

impl RngSource {}

impl From<RngSource> for StdRng {
    fn from(source: RngSource) -> Self {
        match &source {
            RngSource::Default => StdRng::from_os_rng(),
            RngSource::Rng(rng) => rng.clone(),
            RngSource::Seed(seed) => StdRng::seed_from_u64(*seed),
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

/// Replacement sampling mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ReplacementMode {
    /// New samples are selected with replacement.
    #[default]
    WithReplacement,

    /// New samples are selected without replacement;
    /// unless sampling more than the source dataset.
    WithoutReplacement,
}

impl From<bool> for ReplacementMode {
    fn from(value: bool) -> Self {
        if value {
            Self::WithReplacement
        } else {
            Self::WithoutReplacement
        }
    }
}

impl From<ReplacementMode> for bool {
    fn from(mode: ReplacementMode) -> Self {
        match mode {
            ReplacementMode::WithReplacement => true,
            ReplacementMode::WithoutReplacement => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_rngsource() {
        assert_eq!(RngSource::default(), RngSource::Default);

        // from rng.
        let rng = StdRng::seed_from_u64(42);
        assert_eq!(RngSource::from(rng.clone()), RngSource::Rng(rng.clone()));
        assert_eq!(
            <RngSource as Into<StdRng>>::into(RngSource::from(rng.clone()).into()),
            rng.clone()
        );
        let self1 = &RngSource::from(rng.clone());
        assert_eq!(
            match self1 {
                RngSource::Default => StdRng::from_os_rng(),
                RngSource::Rng(rng) => rng.clone(),
                RngSource::Seed(seed) => StdRng::seed_from_u64(*seed),
            },
            rng.clone()
        );

        // from seed.
        assert_eq!(RngSource::from(42), RngSource::Seed(42));
        assert_eq!(
            <RngSource as Into<StdRng>>::into(RngSource::from(42).into()),
            StdRng::seed_from_u64(42)
        );
        let self1 = &RngSource::from(42);
        assert_eq!(
            match self1 {
                RngSource::Default => StdRng::from_os_rng(),
                RngSource::Rng(rng) => rng.clone(),
                RngSource::Seed(seed) => StdRng::seed_from_u64(*seed),
            },
            StdRng::seed_from_u64(42)
        );
    }

    #[test]
    fn test_sizesource() {
        assert_eq!(SizeConfig::default(), SizeConfig::Default);

        assert_eq!(SizeConfig::from(42), SizeConfig::Fixed(42));

        assert_eq!(SizeConfig::from(1.5), SizeConfig::Ratio(1.5));

        assert_eq!(SizeConfig::source(), SizeConfig::Default);
        assert_eq!(SizeConfig::source().resolve(50), 50);
    }

    #[test]
    fn test_replacementmode() {
        assert_eq!(ReplacementMode::default(), ReplacementMode::WithReplacement);

        assert_eq!(
            ReplacementMode::from(true),
            ReplacementMode::WithReplacement
        );
        assert_eq!(
            ReplacementMode::from(false),
            ReplacementMode::WithoutReplacement
        );

        let a: bool = ReplacementMode::WithReplacement.into();
        assert_eq!(a, true);

        let b: bool = ReplacementMode::WithoutReplacement.into();
        assert_eq!(b, false);
    }
}
