//! What an instrumented run's times mean: how its kernels were timed, and whether the clean run
//! splits into the device's time and the host's.

use std::time::Duration;

use burn_instrument::{Bounds, Breakdown, Component, Kernel, Measurement, MemberSchedule};

/// What a component's milliseconds are, given how the run timed its kernels and whether a
/// clean run was measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TimeReading {
    /// Read back after the run, so the kernels ran back to back: a component's time is its
    /// kernels' own device time.
    InPlace,
    /// Read back after the run on members that ran at once, each kernel's part on each member
    /// kept: a component's time is its kernels' time on the lead, the first member, which stands
    /// for every member.
    Lead,
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
            (false, _) if lead_readable(&measurement.breakdown) => Self::Lead,
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
        self.apportion(self.component_time(component))
    }

    /// `kernel`'s measured time as this reading counts it: the lead's part of it, or all of it.
    pub(crate) fn kernel_time(self, kernel: &Kernel) -> Duration {
        match self {
            Self::Lead => kernel.members.first().copied().unwrap_or_default(),
            Self::InPlace | Self::ApportionedOnto { .. } | Self::Serialized => kernel.total,
        }
    }

    /// `kernel`'s launches as this reading counts them, alongside
    /// [`kernel_time`](Self::kernel_time): the lead's, or every one.
    pub(crate) fn kernel_launches(self, kernel: &Kernel) -> usize {
        match self {
            Self::Lead => kernel.member_launches.first().copied().unwrap_or_default(),
            Self::InPlace | Self::ApportionedOnto { .. } | Self::Serialized => kernel.launches,
        }
    }

    /// The measured time of every kernel under `component`, each as
    /// [`kernel_time`](Self::kernel_time) counts it.
    pub(crate) fn component_time(self, component: &Component) -> Duration {
        component
            .kernels
            .iter()
            .map(|kernel| self.kernel_time(kernel))
            .sum::<Duration>()
            + component
                .children
                .iter()
                .map(|child| self.component_time(child))
                .sum::<Duration>()
    }

    /// The measured time of every kernel of the run, each as
    /// [`kernel_time`](Self::kernel_time) counts it: what every share is of.
    pub(crate) fn device_time(self, breakdown: &Breakdown) -> Duration {
        breakdown
            .kernels()
            .map(|kernel| self.kernel_time(kernel))
            .sum()
    }

    /// One measured duration in a component's milliseconds. See
    /// [`apportioned`](Self::apportioned).
    pub(crate) fn apportion(self, measured: Duration) -> Duration {
        let (wall, serialized) = match self {
            Self::ApportionedOnto { wall, serialized } => (wall, serialized),
            Self::InPlace | Self::Lead | Self::Serialized => return measured,
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
            Self::Lead => TimeReadingText {
                timings: [
                    "in place, at the lead: read back after the run, the members ran at once, and a",
                    "component is its kernels' time on the first member, which stands for every member",
                ],
                time_of: "on lead",
                wall: "the clean run",
                on_device: "on the lead, timed in place: what the components above sum to",
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
    /// The members ran in turn, so what their kernels leave of the clean run holds the hops
    /// between them as well as the host, and no timing tells the two apart.
    MembersInTurn {
        /// The clean run less every member's kernels.
        idle: Duration,
    },
    /// The members ran at once and were read at the lead, so what the lead's kernels leave of
    /// the clean run holds its collectives as well as the host.
    MembersAtOnce {
        /// The clean run less the lead's kernels.
        idle: Duration,
        /// The clean run less the slowest member's kernels, which no waiting removed gets under.
        recoverable: Duration,
    },
    /// No clean run was measured to split.
    NoCleanRun,
    /// The run was on several members at once that could not be read at the lead, whose kernels
    /// sum past any one device's time.
    SeveralMembers,
    /// A launch timed alone serializes the run, so its total is not a pipelined run's.
    TimedAlone,
    /// Nothing was timed on the device.
    NothingTimed,
    /// The kernels measured more than the run took: two clocks disagreeing, not a negative
    /// host.
    ClocksDisagree,
}

impl HostSplit {
    /// How `measurement`'s clean run splits, its device time read as `reading` reads it.
    pub(crate) fn new(measurement: &Measurement, reading: TimeReading) -> Self {
        let Some(wall) = measurement.clean_wall else {
            return Self::NoCleanRun;
        };
        let schedule = schedule(measurement);
        if schedule == Some(MemberSchedule::Concurrent) && reading != TimeReading::Lead {
            return Self::SeveralMembers;
        }
        let device = reading.device_time(&measurement.breakdown);
        if measurement.breakdown.timed_alone {
            Self::TimedAlone
        } else if device.is_zero() {
            Self::NothingTimed
        } else if device > wall {
            Self::ClocksDisagree
        } else {
            match schedule {
                Some(MemberSchedule::Sequential) => Self::MembersInTurn {
                    idle: wall - device,
                },
                Some(MemberSchedule::Concurrent) => Self::MembersAtOnce {
                    idle: wall - device,
                    recoverable: wall.saturating_sub(slowest_member(measurement).max(device)),
                },
                None => Self::Measured {
                    host: wall - device,
                    waited: measurement.breakdown.total_wait(),
                },
            }
        }
    }

    /// Of the clean run, the time the device spent on no kernel where the run splits: the
    /// host's, the hops between members in turn, and the lead's collectives.
    pub(crate) fn idle(self) -> Option<Duration> {
        match self {
            Self::Measured { host: idle, .. }
            | Self::MembersInTurn { idle }
            | Self::MembersAtOnce { idle, .. } => Some(idle),
            Self::NoCleanRun
            | Self::SeveralMembers
            | Self::TimedAlone
            | Self::NothingTimed
            | Self::ClocksDisagree => None,
        }
    }

    /// Of [`idle`](Self::idle), what the run would save with nothing waited on.
    pub(crate) fn recoverable(self) -> Option<Duration> {
        match self {
            Self::MembersAtOnce { recoverable, .. } => Some(recoverable),
            Self::Measured { .. }
            | Self::MembersInTurn { .. }
            | Self::NoCleanRun
            | Self::SeveralMembers
            | Self::TimedAlone
            | Self::NothingTimed
            | Self::ClocksDisagree => self.idle(),
        }
    }

    /// The ladder's rung with the device never idle, and the lever that reaches it.
    pub(crate) fn never_idle(self) -> [&'static str; 2] {
        match self {
            Self::MembersInTurn { .. } => ["the members never waiting", "host, hops"],
            Self::MembersAtOnce { .. } => ["the members never waiting", "host, collectives"],
            Self::Measured { .. }
            | Self::NoCleanRun
            | Self::SeveralMembers
            | Self::TimedAlone
            | Self::NothingTimed
            | Self::ClocksDisagree => ["the device never waiting", "fewer launches"],
        }
    }

    /// Why the run does not split, one string per printed line: empty where it does.
    pub(crate) fn why_not_separable(self) -> &'static [&'static str] {
        match self {
            Self::Measured { .. } => &[],
            Self::MembersInTurn { .. } => &[
                "the members ran in turn, so the time",
                "between their kernels holds the hops between them too",
            ],
            Self::MembersAtOnce { .. } => &[
                "the members ran at once, so the time",
                "between the lead's kernels holds its collectives too",
            ],
            Self::NoCleanRun => &["no clean run was measured"],
            Self::SeveralMembers => &[
                "the members ran at once, so their kernels",
                "sum past any one device's time: see each member below",
            ],
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

/// Whether the run's members worked at once or in turn, where it had several.
pub(crate) fn schedule(measurement: &Measurement) -> Option<MemberSchedule> {
    breakdown_schedule(&measurement.breakdown)
}

fn breakdown_schedule(breakdown: &Breakdown) -> Option<MemberSchedule> {
    (breakdown.per_member.as_ref()).map(|per_member| per_member.schedule)
}

/// Whether members that ran at once can be read at the lead: the lead has a lane of its own,
/// every timing was placed and measured some time, and every kernel's time is on the members'
/// lanes, so the lead's lane holds all of its work and none of another member's.
fn lead_readable(breakdown: &Breakdown) -> bool {
    let Some(per_member) = &breakdown.per_member else {
        return false;
    };
    per_member.schedule == MemberSchedule::Concurrent
        && per_member.member_count == per_member.members.len()
        && breakdown.unplaced.timings == 0
        && breakdown.untimed().next().is_none()
        && breakdown.kernels().all(|kernel| {
            kernel.members.iter().sum::<Duration>() == kernel.total
                && kernel.member_launches.iter().sum::<usize>() == kernel.launches
        })
}

/// The device time of the busiest member of `measurement`, zero on a device of one.
fn slowest_member(measurement: &Measurement) -> Duration {
    (measurement.breakdown.per_member.iter())
        .flat_map(|per_member| &per_member.members)
        .map(|member| member.on_device)
        .max()
        .unwrap_or_default()
}

/// Which of a component's declared bounds a run holds it to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HeldBounds {
    /// Its own, against the device's peaks: a device of one, or members in turn.
    Device,
    /// The lead's, against the lead's peaks, as its time is the lead's.
    Lead,
    /// None: the members ran at once and could not be read at the lead, so its time is every
    /// member's and the peaks one member's.
    Neither,
}

impl HeldBounds {
    pub(crate) fn new(breakdown: &Breakdown, reading: TimeReading) -> Self {
        match (reading, breakdown_schedule(breakdown)) {
            (TimeReading::Lead, _) => Self::Lead,
            (_, Some(MemberSchedule::Concurrent)) => Self::Neither,
            _ => Self::Device,
        }
    }

    /// What `component` declared it costs at best, as this run holds it.
    pub(crate) fn bounds(self, component: &Component) -> Option<Bounds> {
        match self {
            Self::Device => component.bounds,
            Self::Lead => component.lead_bounds,
            Self::Neither => None,
        }
    }

    /// The sum of `component`'s children's [`bounds`](Self::bounds).
    pub(crate) fn parts_bounds(self, component: &Component) -> Option<Bounds> {
        match self {
            Self::Device => component.parts_bounds(),
            Self::Lead => component.lead_parts_bounds(),
            Self::Neither => None,
        }
    }
}
