//! Roofline bounds for autotune tables, shared by every cubecl-backed table so the fraction of
//! peak a level settles for is decided in one place.

use alloc::{sync::Arc, vec::Vec};
use core::time::Duration;
use cubecl::{
    config::{CubeClRuntimeConfig, RuntimeConfig, autotune::AutotuneLevel},
    tune::{AutotuneKey, Bounds, BoundsGenerator, Thresholds, TunableSet, TuneInputs},
};

/// Registers `bounds`, which states a problem's roofline, on `set`, so a round ends at the first
/// candidate that runs close enough to peak, unless autotune is configured for
/// [`AutotuneLevel::Full`].
///
/// At `Full` no generator is registered at all, so the tuner reports no bounds instead of an empty
/// set of them: logs and records then say bounds were off. The level is read again on every tune,
/// so a level changed through [`RuntimeConfig::set`] after the tuner was built still applies,
/// except that a tuner built at `Full` holds no generator to apply it to.
pub fn with_roofline_bounds<K, I, Out, F>(
    set: TunableSet<K, I, Out>,
    bounds: F,
) -> TunableSet<K, I, Out>
where
    K: AutotuneKey,
    I: TuneInputs,
    Out: 'static,
    F: for<'a> Fn(&K, &I::At<'a>, Thresholds) -> Bounds + Send + Sync + 'static,
{
    if configured_thresholds().is_none() {
        return set;
    }

    set.with_bounds(Arc::new(AtConfiguredLevel(bounds)))
}

/// The fractions of peak the configured level settles for, or `None` at `Full`.
fn configured_thresholds() -> Option<Thresholds> {
    let config = CubeClRuntimeConfig::get();

    match &config.autotune.level {
        AutotuneLevel::Full => None,
        level => Some(Thresholds::for_level(level)),
    }
}

struct AtConfiguredLevel<F>(F);

impl<K, I, F> BoundsGenerator<K, I> for AtConfiguredLevel<F>
where
    K: 'static,
    I: TuneInputs,
    F: for<'a> Fn(&K, &I::At<'a>, Thresholds) -> Bounds + Send + Sync + 'static,
{
    fn generate<'a>(&self, key: &K, inputs: &I::At<'a>) -> Bounds {
        match configured_thresholds() {
            Some(thresholds) => (self.0)(key, inputs, thresholds),
            // The level was switched to `Full` after the tuner was built.
            None => Bounds {
                bounds: Vec::new(),
                launch_overhead: Duration::ZERO,
            },
        }
    }
}
