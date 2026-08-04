//! Helper for registering roofline autotune bounds on a [`TunableSet`].

use alloc::sync::Arc;
use core::time::Duration;

use cubecl::{
    config::{CubeClRuntimeConfig, RuntimeConfig, autotune::AutotuneLevel},
    tune::{AutotuneKey, Bounds, Thresholds, TunableSet},
};

/// Returns roofline thresholds for an [`AutotuneLevel`], or `None` if bounds are disabled (`Full`).
///
/// A higher threshold is a *tighter* time limit, since the limit is the roofline time divided by
/// the threshold: a candidate has to land closer to peak before autotune stops looking. The levels
/// ramp up accordingly — `Minimal` settles for the first decent candidate, `Extensive` keeps
/// searching, and `Full` sets no limit at all and benchmarks everything.
///
/// The fractions themselves come from small ad-hoc observations, not a systematic sweep. They are
/// a starting point, expected to move, and nothing should depend on these exact values.
const fn thresholds_for_level(level: &AutotuneLevel) -> Option<Thresholds> {
    match level {
        AutotuneLevel::Minimal => Some(Thresholds::uniform(0.6)),
        AutotuneLevel::Balanced => Some(Thresholds::uniform(0.8)),
        AutotuneLevel::Extensive => Some(Thresholds::uniform(0.95)),
        AutotuneLevel::Full => None,
    }
}

/// Returns thresholds for the currently configured autotune level.
///
/// Read on every call rather than captured once, so that a level changed through
/// [`RuntimeConfig::set`] after a tuner was initialized still takes effect — the same way cubecl
/// reads `disable_short_circuit` fresh on each tune.
fn configured_thresholds() -> Option<Thresholds> {
    let config = CubeClRuntimeConfig::get();

    thresholds_for_level(&config.autotune.level)
}

/// A [`Bounds`] with nothing in it, which yields no time limit and short-circuits nothing.
fn no_bounds() -> Bounds {
    Bounds {
        bounds: alloc::vec::Vec::new(),
        launch_overhead: Duration::ZERO,
    }
}

/// Registers a roofline bounds generator built from `compute` on `set`, unless autotune is
/// configured for [`AutotuneLevel::Full`].
///
/// At `Full` no generator is registered at all, so the tuner reports no bounds instead of an empty
/// set of them: logs and records then say bounds were off, rather than showing a bounds object
/// with nothing in it.
pub(crate) fn with_bounds<K, I, Out>(
    set: TunableSet<K, I, Out>,
    compute: impl Fn(&K, &I, Thresholds) -> Bounds + Send + Sync + 'static,
) -> TunableSet<K, I, Out>
where
    K: AutotuneKey,
    I: Clone + Send + Sync + 'static,
    Out: 'static,
{
    if configured_thresholds().is_none() {
        return set;
    }

    set.with_bounds(Arc::new(move |key: &K, inputs: &I| {
        match configured_thresholds() {
            Some(thresholds) => compute(key, inputs, thresholds),
            // The level was switched to `Full` after the tuner was initialized.
            None => no_bounds(),
        }
    }))
}
