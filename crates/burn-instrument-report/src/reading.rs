//! What a collected pass's times mean: how its kernels were timed, and whether the clean pass
//! splits into the device's time and the host's.

use std::time::Duration;

use burn_instrument::{CollectedPass, Component};

/// What a component's milliseconds are, given how the pass timed its kernels and whether an
/// uncollected run was measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TimeReading {
    /// Read back after the pass, so the kernels ran back to back: a component's time is its
    /// kernels' own device time.
    InPlace,
    /// Each read as it launched, so the kernels ran one at a time: a component's time is its
    /// share of the uncollected run, the one unperturbed number there is.
    ApportionedOnto {
        /// The uncollected run's wall time.
        wall: Duration,
        /// Every kernel's serialized time, which the shares are of.
        serialized: Duration,
    },
    /// Each read as it launched, and no uncollected run to apportion onto: a component's time
    /// is its kernels' serialized time, and only the proportions mean anything.
    Serialized,
}

impl TimeReading {
    pub(crate) fn new(pass: &CollectedPass) -> Self {
        match (pass.breakdown.timed_alone, pass.clean_wall) {
            (false, _) => Self::InPlace,
            (true, Some(wall)) => Self::ApportionedOnto {
                wall,
                serialized: pass.breakdown.total_time(),
            },
            (true, None) => Self::Serialized,
        }
    }

    /// This component's milliseconds: its kernels' own device time where they
    /// were measured in place, and its share of the clean pass where any
    /// launch was timed alone.
    ///
    /// In place, nothing waited on a kernel while the pass ran, so each
    /// measurement is the kernel as it runs in a full queue and needs no
    /// correcting. The components then sum to the device's time, and the clean pass's
    /// remainder is the host's — the `host` line.
    ///
    /// **Timed alone, the collected pass has no absolute clock.** Its per-launch device
    /// times are each measured with the kernel running *alone*, since timing a
    /// launch blocks on it, so two collections of the same work can read twice
    /// apart against a clean pass that barely moved.
    /// What the collected pass measures reliably is the **proportions**,
    /// because every kernel in it is measured the same way.
    ///
    /// So the proportions are apportioned onto the one unperturbed number
    /// there is: the median wall time of the clean pass. The result is
    /// absolute, sums to that wall, and is the model's time rather than the
    /// instrumentation's.
    ///
    /// What it assumes is that serialization inflates every kernel alike. It
    /// does not, quite — a fixed cost per launch over-weights a component that
    /// runs many small kernels against one that runs a few large ones — so a
    /// component is good to a few percent and not to the last digit.
    ///
    /// Timed alone with no clean pass measured, there is nothing to apportion
    /// onto: the components are the serialized kernels' own times, and only their
    /// proportions mean anything, which the header says.
    pub(crate) fn apportioned(self, component: &Component) -> Duration {
        self.apportion(component.total_time())
    }

    /// One measured duration in a component's milliseconds. See
    /// [`apportioned`](Self::apportioned).
    pub(crate) fn apportion(self, measured: Duration) -> Duration {
        let (wall, serialized) = match self {
            Self::ApportionedOnto { wall, serialized } => (wall, serialized),
            Self::InPlace | Self::Serialized => return measured,
        };
        if serialized.is_zero() {
            return Duration::ZERO;
        }
        Duration::from_secs_f64(
            wall.as_secs_f64() * measured.as_secs_f64() / serialized.as_secs_f64(),
        )
    }
}

/// Whether a pass splits into the device's time and the time the device waited on the host,
/// and why not where it does not: the clean pass less what its kernels took holds only when
/// that device time is a pipelined pass's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HostSplit {
    /// The clean pass splits.
    Measured {
        /// The time the clean pass's device spent waiting on the host.
        host: Duration,
        /// How long the collected pass's device waited on the host, which `host` is split
        /// across the components by.
        waited: Duration,
    },
    /// No uncollected run was measured to split.
    NoCleanRun,
    /// A launch timed alone serializes the pass, so its total is not a pipelined pass's.
    TimedAlone,
    /// Nothing was timed on the device.
    NothingTimed,
    /// The kernels measured more than the pass took: two clocks disagreeing, not a negative
    /// host.
    ClocksDisagree,
}

impl HostSplit {
    pub(crate) fn new(pass: &CollectedPass) -> Self {
        let Some(wall) = pass.clean_wall else {
            return Self::NoCleanRun;
        };
        let device = pass.breakdown.total_time();
        if pass.breakdown.timed_alone {
            Self::TimedAlone
        } else if device.is_zero() {
            Self::NothingTimed
        } else if device > wall {
            Self::ClocksDisagree
        } else {
            Self::Measured {
                host: wall - device,
                waited: pass.breakdown.total_wait(),
            }
        }
    }

    /// Of the clean pass, the time the device spent waiting on the host — the
    /// launches, the op recording, everything between kernels — where the pass splits.
    pub(crate) fn host(self) -> Option<Duration> {
        match self {
            Self::Measured { host, .. } => Some(host),
            Self::NoCleanRun | Self::TimedAlone | Self::NothingTimed | Self::ClocksDisagree => None,
        }
    }

    /// The clean pass's host time charged to `waited` of the collected pass's: a
    /// component's share, split by how long the device waited before its kernels.
    /// `None` where the pass does not split.
    ///
    /// Split rather than read off directly, because the collected pass's host
    /// pays a round trip per launch to time it, so its idle stretches are
    /// longer than the clean pass's by a cost every launch shares. As shares
    /// they still say which components the device waited on; as absolutes they
    /// would describe the instrumentation. Where that shared cost is most of
    /// every stretch the split falls back to counting launches, which is the
    /// honest reading of a device waiting on the same work each time.
    pub(crate) fn share(self, waited: Duration) -> Option<Duration> {
        let Self::Measured {
            host,
            waited: total,
        } = self
        else {
            return None;
        };
        (!total.is_zero()).then(|| {
            Duration::from_secs_f64(host.as_secs_f64() * waited.as_secs_f64() / total.as_secs_f64())
        })
    }
}
