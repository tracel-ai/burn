//! Numerical contracts for selected floating-point operations.

use super::{BurnConfig, RuntimeConfig};
use crate::sync::{AtomicU8, Ordering};

// Zero means that BurnConfig has not been initialized. The initialized values
// are published by on_loaded for both file loading and programmatic set.
static NAN_POLICY: AtomicU8 = AtomicU8::new(0);

/// NaN behavior for floating-point extrema, cumulative extrema, and clamp on
/// CubeCL backends (including fusion).
///
/// Both policies retain supported non-NaN and infinity behavior. Ordinary ties
/// select the lowest index. Nonempty indexed reductions return valid indices,
/// and paired value/index outputs select the same input element, even with NaNs.
#[derive(
    Default, Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum NanPolicy {
    /// Permit backend-dependent NaN results while retaining ordinary numerical
    /// contracts and valid indices. This does not guarantee NaN suppression.
    #[default]
    Native,
    /// Propagate NaNs, including NaN clamp bounds. Indexed extrema select the
    /// lowest NaN index; cumulative extrema propagate through the remaining scan.
    Propagate,
}

impl NanPolicy {
    /// Whether the covered operations must enforce NaN propagation.
    pub const fn propagates_nan(self) -> bool {
        matches!(self, Self::Propagate)
    }
}

/// Configuration for numerical contracts.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct NumericsConfig {
    /// NaN policy for the covered operations. Defaults to [`NanPolicy::Native`].
    pub nan_policy: NanPolicy,
}

/// Returns the executing process's NaN policy with lock-free steady-state lookup.
///
/// The first call initializes [`BurnConfig`] if needed. Configure it before any
/// Burn configuration read. Graphs use the replaying process's policy; remote
/// operations use the server's policy, independently of the client configuration.
#[inline]
pub fn nan_policy() -> NanPolicy {
    match NAN_POLICY.load(Ordering::Acquire) {
        1 => NanPolicy::Native,
        2 => NanPolicy::Propagate,
        _ => BurnConfig::get().numerics().nan_policy,
    }
}

pub(super) fn publish_nan_policy(policy: NanPolicy) {
    let value = match policy {
        NanPolicy::Native => 1,
        NanPolicy::Propagate => 2,
    };
    NAN_POLICY.store(value, Ordering::Release);
}
