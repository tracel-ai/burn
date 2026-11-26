//! What every cubecl-backed autotune table shares, decided in one place: the roofline bounds a
//! level settles for, and how far down its candidates a round goes ([`PATIENCE`]).

use alloc::{sync::Arc, vec::Vec};
use core::time::Duration;
use cubecl::{
    config::{CubeClRuntimeConfig, RuntimeConfig, autotune::AutotuneLevel},
    tune::{
        AutotuneKey, Bounds, BoundsGenerator, OrderedSettings, Patience, Thresholds, TunableSet,
        TuneGroup, TuneInputs,
    },
};

/// When a round stops measuring a group: after [`min_measured`](Patience::min_measured)
/// candidates in priority order, once [`max_misses`](Patience::max_misses) in a row fail to beat
/// the leader. The minimum also holds the roofline's early stop back, so a table of that many
/// candidates or fewer is always raced whole.
///
/// Twenty misses, not fewer, while the estimates that order a group still miss some kernels by
/// several times: replayed over matmul and attention races on four devices, it settles on the
/// fastest candidate of each, measuring about a third of them.
pub const PATIENCE: Patience = Patience {
    min_measured: 10,
    max_misses: 20,
};

/// The priority the candidates of a [`raced_group`] sit at: one level, so a round over them is
/// their order of registration.
pub const RACED_PRIORITY: i8 = 0;

/// The group a table built on a [`TunableSet`] directly races its candidates in: one group at
/// one priority, its members measured in the order they are added, with the [`PATIENCE`] every
/// table shares. A member joins it at [`RACED_PRIORITY`].
pub fn raced_group<K: AutotuneKey>(name: &str) -> TuneGroup<K> {
    TuneGroup::ordered(
        name,
        |_key| RACED_PRIORITY,
        OrderedSettings::default().with_patience(PATIENCE),
    )
}

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
