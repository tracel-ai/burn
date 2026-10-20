//! What an instrumented run's times mean: how its kernels were timed, and whether the clean run
//! splits into the device's time and the host's.

use std::time::Duration;

use burn_instrument::{Component, Measurement};

/// What a component's milliseconds are, given how the run timed its kernels and whether a
/// clean run was measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TimeReading {
    /// Read back after the run, so the kernels ran back to back: a component's time is its
    /// kernels' own device time.
    InPlace,
    /// Each read as it launched, so the kernels ran one at a time: a component's time is its
    /// share of the clean run, the one unperturbed number there is.
    ApportionedOnto {
        /// The clean run's wall time.
        wall: Duration,
        /// Every kernel's serialized time, which the shares are of.
        serialized: Duration,
    },
    /// Each read as it launched, and no clean run to apportion onto: a component's time
    /// is its kernels' serialized time, and only the proportions mean anything.
    Serialized,
}

impl TimeReading {
    pub(crate) fn new(measurement: &Measurement) -> Self {
        match (measurement.breakdown.timed_alone, measurement.clean_wall) {
            (false, _) => Self::InPlace,
            (true, Some(wall)) => Self::ApportionedOnto {
                wall,
                serialized: measurement.breakdown.total_time(),
            },
            (true, None) => Self::Serialized,
        }
    }

    /// This component's milliseconds: its kernels' own device time where they
    /// were measured in place, and its share of the clean run where any
    /// launch was timed alone.
    ///
    /// In place, nothing waited on a kernel during the run, so each
    /// timing is the kernel as it runs in a full queue and needs no
    /// correcting. The components then sum to the device's time, and the clean run's
    /// remainder is the host's — the `host` line.
    ///
    /// **Timed alone, the instrumented run has no absolute clock.** Its per-launch device
    /// times are each measured with the kernel running *alone*, since timing a
    /// launch blocks on it, so two instrumented runs of the same work can read twice
    /// apart against a clean run that barely moved.
    /// What the instrumented run measures reliably is the **proportions**,
    /// because every kernel in it is measured the same way.
    ///
    /// So the proportions are apportioned onto the one unperturbed number
    /// there is: the median wall time of the clean run. The result is
    /// absolute, sums to that wall, and is the model's time rather than the
    /// instrumentation's.
    ///
    /// What it assumes is that serialization inflates every kernel alike. It
    /// does not, quite — a fixed cost per launch over-weights a component that
    /// runs many small kernels against one that runs a few large ones — so a
    /// component is good to a few percent and not to the last digit.
    ///
    /// Timed alone with no clean run measured, there is nothing to apportion
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

    /// What a report says about this reading, wherever it says it.
    pub(crate) fn text(self) -> TimeReadingText {
        match self {
            Self::InPlace => TimeReadingText {
                timings: [
                    "in place — read back after the run, the kernels ran back to back: a component is",
                    "its kernels' device time, and `host` is the device waiting on the host",
                ],
                time_of: "on device",
                wall: "the clean run",
                on_device: "timed in place — what the components above sum to",
            },
            Self::ApportionedOnto { .. } => TimeReadingText {
                timings: [
                    "alone — each read as it launched, so the kernels ran one at a time: a component is",
                    "its share of the clean run, good to a few percent",
                ],
                time_of: "of wall",
                wall: "the clean run, and what the components above sum to",
                on_device: "each timed alone",
            },
            Self::Serialized => TimeReadingText {
                timings: [
                    "alone — each read as it launched, so the kernels ran one at a time: a component is",
                    "its kernels' serialized time, and only the proportions mean anything",
                ],
                time_of: "on device",
                wall: "the clean run",
                on_device: "each timed alone",
            },
        }
    }
}

/// What a report says about one [`TimeReading`], gathered so that a reading's words change
/// together.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TimeReadingText {
    /// How the kernels were timed and what a component's time is, one string per line.
    pub(crate) timings: [&'static str; 2],
    /// What the time column is a time of.
    pub(crate) time_of: &'static str,
    /// Beside the clean run's wall time.
    pub(crate) wall: &'static str,
    /// Beside the instrumented run's device time.
    pub(crate) on_device: &'static str,
}

/// Whether a run splits into the device's time and the time the device waited on the host,
/// and why not where it does not: the clean run less what its kernels took holds only when
/// that device time is a pipelined run's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HostSplit {
    /// The clean run splits.
    Measured {
        /// The time the clean run's device spent waiting on the host.
        host: Duration,
        /// How long the instrumented run's device waited on the host, which `host` is split
        /// across the components by.
        waited: Duration,
    },
    /// No clean run was measured to split.
    NoCleanRun,
    /// A launch timed alone serializes the run, so its total is not a pipelined run's.
    TimedAlone,
    /// Nothing was timed on the device.
    NothingTimed,
    /// The kernels measured more than the run took: two clocks disagreeing, not a negative
    /// host.
    ClocksDisagree,
}

impl HostSplit {
    pub(crate) fn new(measurement: &Measurement) -> Self {
        let Some(wall) = measurement.clean_wall else {
            return Self::NoCleanRun;
        };
        let device = measurement.breakdown.total_time();
        if measurement.breakdown.timed_alone {
            Self::TimedAlone
        } else if device.is_zero() {
            Self::NothingTimed
        } else if device > wall {
            Self::ClocksDisagree
        } else {
            Self::Measured {
                host: wall - device,
                waited: measurement.breakdown.total_wait(),
            }
        }
    }

    /// Of the clean run, the time the device spent waiting on the host — the
    /// launches, the op recording, everything between kernels — where the run splits.
    pub(crate) fn host(self) -> Option<Duration> {
        match self {
            Self::Measured { host, .. } => Some(host),
            Self::NoCleanRun | Self::TimedAlone | Self::NothingTimed | Self::ClocksDisagree => None,
        }
    }

    /// Why the run does not split, one string per printed line: empty where it does.
    pub(crate) fn why_not_separable(self) -> &'static [&'static str] {
        match self {
            Self::Measured { .. } => &[],
            Self::NoCleanRun => &["no clean run was measured"],
            Self::TimedAlone => &[
                "a launch timed alone serializes the",
                "run, so the measured total is not a pipelined run's",
            ],
            Self::NothingTimed => &["nothing was timed on the device"],
            Self::ClocksDisagree => &[
                "the kernels measured more than the",
                "run took, so the two clocks disagree",
            ],
        }
    }

    /// The clean run's host time charged to `waited` of the instrumented run's: a
    /// component's share, split by how long the device waited before its kernels.
    /// `None` where the run does not split.
    ///
    /// Split rather than read off directly, because the instrumented run's host
    /// pays a round trip per launch to time it, so its idle stretches are
    /// longer than the clean run's by a cost every launch shares. As shares
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
