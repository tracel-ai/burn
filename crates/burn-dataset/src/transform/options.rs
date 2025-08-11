use rand::SeedableRng;
use rand::prelude::StdRng;

/// Helper option to create a rng from a variety of options.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RandomSource {
    /// Build a new rng from the system.
    #[default]
    System,

    /// The rng is passed as an option.
    FromRng(StdRng),

    /// The rng is passed as a seed.
    FromSeed(u64),
}

impl RandomSource {
    /// Convert the source into a rng.
    pub fn to_rng(self) -> StdRng {
        match self {
            RandomSource::System => StdRng::from_os_rng(),
            RandomSource::FromRng(rng) => rng,
            RandomSource::FromSeed(seed) => StdRng::seed_from_u64(seed),
        }
    }
}

impl From<RandomSource> for StdRng {
    fn from(source: RandomSource) -> Self {
        source.to_rng()
    }
}

impl<T> From<Option<T>> for RandomSource
where
    T: Into<RandomSource>,
{
    fn from(rng: Option<T>) -> Self {
        match rng {
            Some(rng) => rng.into(),
            None => Self::default(),
        }
    }
}

impl From<u64> for RandomSource {
    fn from(seed: u64) -> Self {
        Self::FromSeed(seed)
    }
}

impl From<StdRng> for RandomSource {
    fn from(rng: StdRng) -> Self {
        Self::FromRng(rng)
    }
}

/// Helper trait to add random source setters to a type.
pub trait WithRandomSourceSetters: Sized {
    /// The rng source.
    fn with_rng<R>(self, rng: R) -> Self
    where
        R: Into<RandomSource>;

    /// Use the system rng.
    fn with_system_rng(self) -> Self {
        self.with_rng(RandomSource::System)
    }

    /// Use an rng, built from a seed.
    fn with_seed(self, seed: u64) -> Self {
        self.with_rng(seed)
    }
}

/// Helper option to describe the size of a wrapper, relative to a wrapped object.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum WrapperSizeSource {
    /// Use the size of the source dataset.
    #[default]
    Source,

    /// Use the size as a ratio of the source dataset size.
    ///
    /// Must be >= 0.
    Ratio(f64),

    /// Use a fixed size.
    Fixed(usize),
}

impl WrapperSizeSource {
    /// Construct a source which will have the same size as the source dataset.
    pub fn source() -> Self {
        Self::Source
    }

    /// Construct a source which will have a size that is ``floor(<source> * <ratio>)``.
    pub fn ratio(ratio: f64) -> Self {
        assert!(ratio >= 0.0, "Ratio must be positive: {ratio}");
        Self::Ratio(ratio)
    }

    /// Construct a source which will have a fixed size.
    pub fn fixed(size: usize) -> Self {
        Self::Fixed(size)
    }

    /// Computes the effective size, given the source.
    pub fn evaluate_for_source(self, source_size: usize) -> usize {
        match self {
            WrapperSizeSource::Source => source_size,
            WrapperSizeSource::Ratio(ratio) => {
                assert!(ratio > 0.0, "Ratio must be positive: {ratio}");
                ((source_size as f64) * ratio) as usize
            }
            WrapperSizeSource::Fixed(size) => size,
        }
    }
}

impl<T> From<Option<T>> for WrapperSizeSource
where
    T: Into<WrapperSizeSource>,
{
    fn from(option: Option<T>) -> Self {
        match option {
            Some(option) => option.into(),
            None => Self::default(),
        }
    }
}

impl From<usize> for WrapperSizeSource {
    fn from(size: usize) -> Self {
        Self::Fixed(size)
    }
}

impl From<f64> for WrapperSizeSource {
    fn from(ratio: f64) -> Self {
        Self::Ratio(ratio)
    }
}

/// Helper trait to add size source setters to a type.
pub trait WithWrapperSizeSourceSetters: Sized {
    /// Set the size source.
    fn with_size<S>(self, size: S) -> Self
    where
        S: Into<WrapperSizeSource>;

    /// Set the size to a fixed size.
    fn with_fixed_size(self, size: usize) -> Self {
        self.with_size(size)
    }

    /// Set the size to be a multiple of the ration and the source size.
    fn with_size_ratio(self, size_ratio: f64) -> Self {
        self.with_size(size_ratio)
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

impl<T> From<Option<T>> for ReplacementMode
where
    T: Into<ReplacementMode>,
{
    fn from(option: Option<T>) -> Self {
        match option {
            Some(option) => option.into(),
            None => Self::default(),
        }
    }
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

/// Helper trait to add replacement mode setters to a type.
pub trait WithReplacementModeSetters: Sized {
    /// Set the replacement mode.
    fn with_replacement_mode<M>(self, mode: M) -> Self
    where
        M: Into<ReplacementMode>;

    /// Set the replacement mode to WithReplacement.
    fn with_replacement(self) -> Self {
        self.with_replacement_mode(ReplacementMode::WithReplacement)
    }

    /// Set the replacement mode to WithoutReplacement.
    fn without_replacement(self) -> Self {
        self.with_replacement_mode(ReplacementMode::WithoutReplacement)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    #[test]
    fn test_randomsource() {
        assert_eq!(RandomSource::default(), RandomSource::System);
        assert_eq!(
            RandomSource::from(None as Option<RandomSource>),
            RandomSource::default()
        );

        // from rng.
        let rng = StdRng::seed_from_u64(42);
        assert_eq!(
            RandomSource::from(rng.clone()),
            RandomSource::FromRng(rng.clone())
        );
        assert_eq!(
            <RandomSource as Into<StdRng>>::into(RandomSource::from(rng.clone()).into()),
            rng.clone()
        );
        assert_eq!(RandomSource::from(rng.clone()).to_rng(), rng.clone());

        // from seed.
        assert_eq!(RandomSource::from(42), RandomSource::FromSeed(42));
        assert_eq!(
            <RandomSource as Into<StdRng>>::into(RandomSource::from(42).into()),
            StdRng::seed_from_u64(42)
        );
        assert_eq!(RandomSource::from(42).to_rng(), StdRng::seed_from_u64(42));

        // from system
        assert_eq!(
            RandomSource::from(None as Option<StdRng>),
            RandomSource::System
        );
        assert_eq!(
            RandomSource::from(None as Option<u64>),
            RandomSource::System
        );
    }

    #[test]
    fn test_sizesource() {
        assert_eq!(WrapperSizeSource::default(), WrapperSizeSource::Source);
        assert_eq!(
            WrapperSizeSource::from(None as Option<WrapperSizeSource>),
            WrapperSizeSource::default()
        );

        assert_eq!(WrapperSizeSource::from(42), WrapperSizeSource::Fixed(42));
        assert_eq!(WrapperSizeSource::fixed(42), WrapperSizeSource::Fixed(42));
        assert_eq!(WrapperSizeSource::fixed(100).evaluate_for_source(50), 100);

        assert_eq!(WrapperSizeSource::from(1.5), WrapperSizeSource::Ratio(1.5));
        assert_eq!(WrapperSizeSource::ratio(1.5), WrapperSizeSource::Ratio(1.5));
        assert_eq!(WrapperSizeSource::ratio(1.5).evaluate_for_source(50), 75);

        assert_eq!(
            WrapperSizeSource::from(None as Option<usize>),
            WrapperSizeSource::Source
        );
        assert_eq!(
            WrapperSizeSource::from(None as Option<f64>),
            WrapperSizeSource::Source
        );
        assert_eq!(WrapperSizeSource::source(), WrapperSizeSource::Source);
        assert_eq!(WrapperSizeSource::source().evaluate_for_source(50), 50);
    }

    #[test]
    fn test_replacementmode() {
        assert_eq!(ReplacementMode::default(), ReplacementMode::WithReplacement);
        assert_eq!(
            ReplacementMode::from(None as Option<ReplacementMode>),
            ReplacementMode::default()
        );

        assert_eq!(
            ReplacementMode::from(true),
            ReplacementMode::WithReplacement
        );
        assert_eq!(
            ReplacementMode::from(false),
            ReplacementMode::WithoutReplacement
        );
    }
}
