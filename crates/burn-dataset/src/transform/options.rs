use rand::SeedableRng;
use rand::prelude::StdRng;

/// Helper option to create a rng from a variety of options.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum RandomSource {
    /// Build a new rng from the system.
    #[default]
    System,

    /// The rng is passed as an option.
    FromRng(Box<StdRng>),

    /// The rng is passed as a seed.
    FromSeed(u64),
}

impl RandomSource {
    /// Convert the source into a rng.
    pub fn build(&self) -> StdRng {
        match self {
            RandomSource::System => StdRng::from_os_rng(),
            RandomSource::FromRng(rng) => rng.as_ref().clone(),
            RandomSource::FromSeed(seed) => StdRng::seed_from_u64(*seed),
        }
    }
}

impl From<RandomSource> for StdRng {
    fn from(source: RandomSource) -> Self {
        source.build()
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
        Self::FromRng(Box::new(rng))
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

    /// Set the size to the size of the source.
    fn with_source_size(self) -> Self {
        self.with_size(WrapperSizeSource::Source)
    }

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
            RandomSource::FromRng(Box::new(rng.clone()))
        );
        assert_eq!(
            <RandomSource as Into<StdRng>>::into(RandomSource::from(rng.clone()).into()),
            rng.clone()
        );
        assert_eq!(RandomSource::from(rng.clone()).build(), rng.clone());

        // from seed.
        assert_eq!(RandomSource::from(42), RandomSource::FromSeed(42));
        assert_eq!(
            <RandomSource as Into<StdRng>>::into(RandomSource::from(42).into()),
            StdRng::seed_from_u64(42)
        );
        assert_eq!(RandomSource::from(42).build(), StdRng::seed_from_u64(42));

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

    #[derive(Default)]
    struct RandomSourceHolder {
        pub msg: String,

        pub rng: RandomSource,
    }
    impl WithRandomSourceSetters for RandomSourceHolder {
        fn with_rng<R>(self, rng: R) -> Self
        where
            R: Into<RandomSource>,
        {
            Self {
                rng: rng.into(),
                ..self
            }
        }
    }

    #[test]
    fn test_with_random_source_setters() {
        let holder = RandomSourceHolder::default();
        assert_eq!(holder.msg, "");
        assert_eq!(holder.rng, RandomSource::System);
        let _unused = holder.rng.build();

        let holder = RandomSourceHolder::default().with_system_rng();
        assert_eq!(holder.rng, RandomSource::System);
        let _unused = holder.rng.build();

        let holder = RandomSourceHolder::default().with_rng(42);
        assert_eq!(holder.rng, RandomSource::FromSeed(42));
        assert_eq!(holder.rng.build(), StdRng::seed_from_u64(42));

        let holder = RandomSourceHolder::default().with_seed(42);
        assert_eq!(holder.rng, RandomSource::FromSeed(42));
        assert_eq!(holder.rng.build(), StdRng::seed_from_u64(42));

        let rng = StdRng::from_os_rng();
        let holder = RandomSourceHolder::default().with_rng(rng.clone());
        assert_eq!(holder.rng, RandomSource::FromRng(Box::new(rng.clone())));
        assert_eq!(holder.rng.build(), rng.clone());
    }

    #[test]
    fn test_sizesource() {
        assert_eq!(WrapperSizeSource::default(), WrapperSizeSource::Source);
        assert_eq!(
            WrapperSizeSource::from(None as Option<WrapperSizeSource>),
            WrapperSizeSource::default()
        );

        assert_eq!(WrapperSizeSource::from(42), WrapperSizeSource::Fixed(42));
        assert_eq!(
            WrapperSizeSource::from(Some(42)),
            WrapperSizeSource::Fixed(42)
        );
        assert_eq!(WrapperSizeSource::fixed(42), WrapperSizeSource::Fixed(42));
        assert_eq!(WrapperSizeSource::fixed(100).evaluate_for_source(50), 100);

        assert_eq!(WrapperSizeSource::from(1.5), WrapperSizeSource::Ratio(1.5));
        assert_eq!(
            WrapperSizeSource::from(Some(1.5)),
            WrapperSizeSource::Ratio(1.5)
        );
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

    #[derive(Default)]
    struct SizeSourceHolder {
        pub msg: String,
        pub size: WrapperSizeSource,
    }

    impl WithWrapperSizeSourceSetters for SizeSourceHolder {
        fn with_size<S>(self, size: S) -> Self
        where
            S: Into<WrapperSizeSource>,
        {
            Self {
                size: size.into(),
                ..self
            }
        }
    }

    #[test]
    fn test_with_wrapper_size_source_setters() {
        let holder = SizeSourceHolder::default();
        assert_eq!(holder.msg, "");
        assert_eq!(holder.size, WrapperSizeSource::Source);

        let holder = SizeSourceHolder::default().with_source_size();
        assert_eq!(holder.msg, "");
        assert_eq!(holder.size, WrapperSizeSource::Source);

        let holder = SizeSourceHolder::default().with_size(42);
        assert_eq!(holder.size, WrapperSizeSource::Fixed(42));
        assert_eq!(holder.size.evaluate_for_source(50), 42);

        let holder = SizeSourceHolder::default().with_fixed_size(42);
        assert_eq!(holder.size, WrapperSizeSource::Fixed(42));
        assert_eq!(holder.size.evaluate_for_source(50), 42);

        let holder = SizeSourceHolder::default().with_size(1.5);
        assert_eq!(holder.size, WrapperSizeSource::Ratio(1.5));
        assert_eq!(holder.size.evaluate_for_source(50), 75);

        let holder = SizeSourceHolder::default().with_size_ratio(1.5);
        assert_eq!(holder.size, WrapperSizeSource::Ratio(1.5));
        assert_eq!(holder.size.evaluate_for_source(50), 75);
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

        let a: bool = ReplacementMode::WithReplacement.into();
        assert_eq!(a, true);

        let b: bool = ReplacementMode::WithoutReplacement.into();
        assert_eq!(b, false);
    }

    #[derive(Default)]
    struct ReplacementModeHolder {
        pub msg: String,
        pub mode: ReplacementMode,
    }

    impl WithReplacementModeSetters for ReplacementModeHolder {
        fn with_replacement_mode<M>(self, mode: M) -> Self
        where
            M: Into<ReplacementMode>,
        {
            Self {
                mode: mode.into(),
                ..self
            }
        }
    }

    #[test]
    fn test_with_replacement_mode_setters() {
        let holder = ReplacementModeHolder::default();
        assert_eq!(holder.msg, "");
        assert_eq!(holder.mode, ReplacementMode::WithReplacement);

        let holder = ReplacementModeHolder::default().with_replacement();
        assert_eq!(holder.mode, ReplacementMode::WithReplacement);

        let holder = ReplacementModeHolder::default().with_replacement_mode(true);
        assert_eq!(holder.mode, ReplacementMode::WithReplacement);

        let holder = ReplacementModeHolder::default().with_replacement_mode(Some(true));
        assert_eq!(holder.mode, ReplacementMode::WithReplacement);

        let holder = ReplacementModeHolder::default()
            .with_replacement_mode(ReplacementMode::WithReplacement);
        assert_eq!(holder.mode, ReplacementMode::WithReplacement);

        let holder = ReplacementModeHolder::default()
            .with_replacement_mode(Some(ReplacementMode::WithReplacement));
        assert_eq!(holder.mode, ReplacementMode::WithReplacement);

        let holder = ReplacementModeHolder::default().without_replacement();
        assert_eq!(holder.mode, ReplacementMode::WithoutReplacement);

        let holder = ReplacementModeHolder::default().with_replacement_mode(false);
        assert_eq!(holder.mode, ReplacementMode::WithoutReplacement);

        let holder = ReplacementModeHolder::default().with_replacement_mode(Some(false));
        assert_eq!(holder.mode, ReplacementMode::WithoutReplacement);

        let holder = ReplacementModeHolder::default()
            .with_replacement_mode(ReplacementMode::WithoutReplacement);
        assert_eq!(holder.mode, ReplacementMode::WithoutReplacement);

        let holder = ReplacementModeHolder::default()
            .with_replacement_mode(Some(ReplacementMode::WithoutReplacement));
        assert_eq!(holder.mode, ReplacementMode::WithoutReplacement);
    }
}
