//! One component's floor and score, and the ladder of a whole pass.

use std::fmt::{self, Display};
use std::time::Duration;

use burn_instrument::{Bounds, Peaks};

/// Which of the two quantities a component declares sets its floor here.
///
/// Named for what it counts rather than for a class of hardware limit: a `flops` floor says
/// arithmetic set the floor on this device, not that the kernel is held by its ALUs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    /// Traffic through global memory, against the device's measured bandwidth.
    Bytes,
    /// Inner-loop arithmetic, against the device's measured peak.
    Flops,
}

/// The word a report prints, padded to the column it is printed in.
impl Display for Bound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(match self {
            Bound::Bytes => "bytes",
            Bound::Flops => "flops",
        })
    }
}

/// What a component would take at the roof, and how close it got.
#[derive(Clone, Copy, Debug)]
pub struct Score {
    /// The wall it is bound by — whichever gives the longer floor.
    pub bound: Bound,
    /// The time it cannot go below on this device.
    pub floor: Duration,
    /// Achieved against the peak of its binding wall, in `[0, 1]` for anything
    /// that is not beating the probe.
    pub achieved: f64,
}

impl Score {
    /// `bounds` scored against `peaks`, given what it measured.
    ///
    /// `None` when the device could not be probed, or when nothing was measured: a
    /// percentage against a denominator nobody measured, or of no time at all, is
    /// worse than no percentage.
    pub fn new(bounds: Bounds, measured: Duration, peaks: &Peaks) -> Option<Self> {
        let seconds = measured.as_secs_f64();
        if seconds <= 0.0 {
            return None;
        }
        let floor = Floor::new(bounds, peaks)?;
        Some(Self {
            bound: floor.bound,
            floor: floor.time,
            achieved: floor.time.as_secs_f64() / seconds,
        })
    }

    /// Time the component could still give back: what it took, less its floor.
    ///
    /// An **upper** bound on what any amount of work can recover, and at a
    /// composite it is not all reachable by faster kernels — part of it is the
    /// fuser's. [`Ladder`] is what splits the two.
    pub fn gain(&self, measured: Duration) -> Duration {
        measured.saturating_sub(self.floor)
    }
}

/// The time a declaration cannot go below on a device, and the wall that sets
/// it.
///
/// Apart from [`Score`] because a floor belongs to the declaration and the
/// device, not to what ran. A component whose work fused into the one after it
/// measured nothing and still has a floor — and the ladder's rungs are sums of
/// floors, so dropping it there would pull the rung below what the pass's
/// declarations actually cost.
#[derive(Clone, Copy, Debug)]
pub struct Floor {
    /// The wall it is bound by — whichever gives the longer floor.
    pub bound: Bound,
    /// The time itself.
    pub time: Duration,
}

impl Floor {
    /// The floor of `bounds` against `peaks`, or `None` when the device could
    /// not be probed for either wall.
    pub fn new(bounds: Bounds, peaks: &Peaks) -> Option<Self> {
        // Each wall's floor, where the device reported a peak for it.
        //
        // The bytes wall charges each direction at its own ceiling and sums them:
        // the two share one bus, so a component that both reads and writes waits
        // for each in turn.
        //
        // Where a device reports no per-direction peaks the copy ceiling over the
        // total is the fallback, which is the older, coarser answer rather than
        // none.
        let bytes = match (peaks.read_bytes_per_s, peaks.write_bytes_per_s) {
            (Some(read), Some(write)) if read > 0.0 && write > 0.0 => {
                Some(bounds.read as f64 / read + bounds.write as f64 / write)
            }
            _ => peaks
                .copy_bytes_per_s
                .filter(|peak| *peak > 0.0)
                .map(|peak| bounds.bytes() as f64 / peak),
        };
        let flops = peaks
            .flops
            .map(|peak| peak.per_s)
            .filter(|peak| *peak > 0.0)
            .filter(|_| bounds.flops > 0)
            .map(|peak| bounds.flops as f64 / peak);

        // The binding constraint is the longer floor, which is also what names
        // the wall — no ridge point to compute, and no way for the two to
        // disagree.
        let (bound, seconds) = match (bytes, flops) {
            (Some(bytes), Some(flops)) if flops > bytes => (Bound::Flops, flops),
            (Some(bytes), _) => (Bound::Bytes, bytes),
            (None, Some(flops)) => (Bound::Flops, flops),
            (None, None) => return None,
        };
        Some(Self {
            bound,
            time: Duration::from_secs_f64(seconds),
        })
    }
}

/// Where a pass's time could go, one rung per lever.
///
/// The rungs come from the *declarations*, not from what actually fused, so the
/// two facts a report carries stay independent: this says what fusion could
/// buy, and the kernels a pass ran say what it did.
///
/// The first lever is not a kernel's at all. Where the device time was
/// measured in place, the pass splits into the kernels and the time the device
/// waited on the host between them — which no faster kernel and no fusion
/// recovers, and fewer launches does.
#[derive(Clone, Copy, Debug)]
pub struct Ladder {
    /// What the pass took: the clean pass's wall time where it was measured, and the device's
    /// time otherwise.
    pub measured: Duration,
    /// Of that, what the device spent waiting on the host. Zero where the
    /// split is not known, which leaves the whole pass to the kernels.
    pub host: Duration,
    /// Every component at its own floor, taken as deep as the tree declares
    /// one — what faster kernels alone could reach.
    pub kernels: Duration,
    /// The top-level components at their floors, which assume perfect fusion
    /// *within* each — what fewer round-trips inside them would add.
    pub fused: Duration,
}

impl Ladder {
    /// Whether the rungs are in the order the arithmetic requires.
    ///
    /// A floor cannot exceed the time it is a floor for, and perfect fusion
    /// cannot cost more than each part at its own roof. Either would mean a
    /// broken denominator rather than a fast run — and saturating the
    /// subtractions would report it as "nothing on the table", which reads as
    /// the opposite of what happened.
    pub fn coherent(&self) -> bool {
        self.fused <= self.kernels && self.kernels <= self.device()
    }

    /// What the pass took with the device never waiting: the rung the kernels
    /// are measured from.
    pub fn device(&self) -> Duration {
        self.measured.saturating_sub(self.host)
    }

    /// What faster kernels could recover, with the fusion as it is.
    pub fn from_kernels(&self) -> Duration {
        self.device().saturating_sub(self.kernels)
    }

    /// What fusing inside the components could recover on top of that.
    pub fn from_fusion(&self) -> Duration {
        self.kernels.saturating_sub(self.fused)
    }

    /// Everything between what the pass took and its deepest floor.
    pub fn total(&self) -> Duration {
        self.measured.saturating_sub(self.fused)
    }
}

#[cfg(test)]
mod tests {
    use burn_instrument::ComputePeak;
    use burn_tensor::FloatDType;

    use super::*;

    /// ~199 GB/s and ~49 TFLOP/s at f16.
    fn peaks() -> Peaks {
        Peaks {
            copy_bytes_per_s: Some(198.9e9),
            flops: Some(ComputePeak {
                per_s: 49.3e12,
                dtype: FloatDType::F16,
            }),
            ..Peaks::default()
        }
    }

    /// Each direction is charged at its own ceiling, not both at a mixed one.
    ///
    /// The case that motivated the split: a decode projection streams its
    /// weights and writes one short vector, so it is nearly pure read. Scored
    /// against a copy ceiling well below the read one it reads past 100% of
    /// peak, which says the probe failed when the kernel is simply doing the
    /// thing the copy figure does not describe.
    #[test]
    fn each_direction_is_charged_at_its_own_ceiling() {
        let peaks = Peaks {
            copy_bytes_per_s: Some(160.0e9),
            read_bytes_per_s: Some(230.0e9),
            write_bytes_per_s: Some(115.0e9),
            flops: Some(ComputePeak {
                per_s: 50.0e12,
                dtype: FloatDType::F16,
            }),
        };
        // 230 MB read, 11.5 MB written: 1 ms of reading and 0.1 ms of writing.
        let bounds = Bounds {
            read: 230_000_000,
            write: 11_500_000,
            flops: 0,
        };
        let scored = Score::new(bounds, Duration::from_micros(1_100), &peaks).expect("scored");
        assert_eq!(scored.bound, Bound::Bytes);
        assert!(
            (scored.floor.as_secs_f64() - 0.0011).abs() < 1e-9,
            "read at its own ceiling plus write at its own: {:?}",
            scored.floor
        );
        // The same bytes against the copy ceiling would have claimed 1.51 ms of
        // floor for a pass that took 1.1 -- 137% of a peak it never exceeded.
        assert!(scored.achieved <= 1.0, "achieved {}", scored.achieved);
    }

    /// A device that reports no per-direction peaks still gets the older,
    /// coarser answer rather than none.
    #[test]
    fn the_copy_ceiling_is_the_fallback_where_directions_are_unmeasured() {
        let peaks = Peaks {
            copy_bytes_per_s: Some(200.0e9),
            flops: None,
            ..Peaks::default()
        };
        let bounds = Bounds {
            read: 150_000_000,
            write: 50_000_000,
            flops: 0,
        };
        let scored = Score::new(bounds, Duration::from_millis(2), &peaks).expect("scored");
        // 200 MB over 200 GB/s is a millisecond, whatever the mix.
        assert!(
            (scored.floor.as_secs_f64() - 0.001).abs() < 1e-9,
            "{:?}",
            scored.floor
        );
    }

    /// A device that measured one direction and not the other falls back too.
    ///
    /// The two are scored as a pair — read at its ceiling plus write at its
    /// own — so half a pair is not half an answer: charging the measured
    /// direction and nothing for the other would declare a floor below
    /// anything the device can do.
    #[test]
    fn one_measured_direction_is_not_half_a_pair() {
        let half = Peaks {
            copy_bytes_per_s: Some(200.0e9),
            read_bytes_per_s: Some(230.0e9),
            flops: None,
            ..Peaks::default()
        };
        let bounds = Bounds {
            read: 150_000_000,
            write: 50_000_000,
            flops: 0,
        };
        let scored = Score::new(bounds, Duration::from_millis(2), &half).expect("scored");
        // 200 MB over the 200 GB/s copy ceiling, not 150 over 230.
        assert!(
            (scored.floor.as_secs_f64() - 0.001).abs() < 1e-9,
            "{:?}",
            scored.floor
        );
    }

    #[test]
    fn the_binding_wall_is_the_one_with_the_longer_floor() {
        // A projection: 25 GFLOP against 25 MB is far past this device's
        // ratio, so no amount of bandwidth saves it.
        let matmul = Bounds {
            read: 18_750_000,
            write: 6_250_000,
            flops: 25_000_000_000,
        };
        let matmul =
            Score::new(matmul, Duration::from_millis(1), &peaks()).expect("both peaks known");
        assert_eq!(matmul.bound, Bound::Flops);

        // A norm: the same bytes and almost no arithmetic.
        let norm = Bounds {
            read: 18_750_000,
            write: 6_250_000,
            flops: 1_000_000,
        };
        let norm = Score::new(norm, Duration::from_millis(1), &peaks()).expect("both peaks known");
        assert_eq!(norm.bound, Bound::Bytes);
    }

    #[test]
    fn a_component_at_its_floor_reads_as_full() {
        let bounds = Bounds {
            read: 149_175_000_000,
            write: 49_725_000_000,
            flops: 0,
        };
        // A second's worth of traffic, measured in a second.
        let scored = Score::new(bounds, Duration::from_secs(1), &peaks()).expect("a memory peak");
        assert!((scored.achieved - 1.0).abs() < 1e-9);
        assert_eq!(scored.gain(Duration::from_secs(1)), Duration::ZERO);
    }

    #[test]
    fn an_unprobed_device_scores_nothing_rather_than_dividing_by_a_guess() {
        let blank = Peaks::default();
        assert!(
            Score::new(
                Bounds {
                    read: 1,
                    write: 0,
                    flops: 1
                },
                Duration::from_millis(1),
                &blank
            )
            .is_none()
        );
    }

    #[test]
    fn a_component_with_no_arithmetic_is_never_held_against_flops() {
        // `0 / peak` is not a utilization, so the `flops` wall does not apply.
        let gather = Bounds {
            read: 750_000,
            write: 250_000,
            flops: 0,
        };
        let gather = Score::new(gather, Duration::from_millis(1), &peaks()).expect("a memory peak");
        assert_eq!(gather.bound, Bound::Bytes);
    }

    #[test]
    fn the_ladder_splits_the_two_levers() {
        let ladder = Ladder {
            measured: Duration::from_millis(1000),
            host: Duration::ZERO,
            kernels: Duration::from_millis(600),
            fused: Duration::from_millis(400),
        };
        // 400 ms is the kernels'; 200 ms more is only reachable by fusing, and
        // a single number would have promised the kernel author all 600.
        assert_eq!(ladder.from_kernels(), Duration::from_millis(400));
        assert_eq!(ladder.from_fusion(), Duration::from_millis(200));
        assert_eq!(ladder.total(), Duration::from_millis(600));
    }

    /// The time the device waited on the host is nobody's kernel: a faster
    /// kernel does not shorten it, so it is not promised to one.
    #[test]
    fn the_host_is_a_lever_of_its_own() {
        let ladder = Ladder {
            measured: Duration::from_millis(1000),
            host: Duration::from_millis(100),
            kernels: Duration::from_millis(600),
            fused: Duration::from_millis(400),
        };
        assert_eq!(ladder.device(), Duration::from_millis(900));
        assert_eq!(ladder.from_kernels(), Duration::from_millis(300));
        assert_eq!(
            ladder.total(),
            Duration::from_millis(600),
            "all of it is still on the table, by one lever or another"
        );

        // Floors that fit the pass but not the device's share of it are a
        // broken denominator, not a fast run.
        let inverted = Ladder {
            kernels: Duration::from_millis(950),
            ..ladder
        };
        assert!(!inverted.coherent());
    }

    #[test]
    fn a_ladder_whose_rungs_invert_reports_nothing_rather_than_a_negative() {
        // Cannot arise from declarations that hold the parent-below-parts
        // invariant, but a mispriced composite must not print a negative gain.
        let ladder = Ladder {
            measured: Duration::from_millis(100),
            host: Duration::ZERO,
            kernels: Duration::from_millis(200),
            fused: Duration::from_millis(300),
        };
        assert_eq!(ladder.from_kernels(), Duration::ZERO);
        assert_eq!(ladder.from_fusion(), Duration::ZERO);
    }
}
