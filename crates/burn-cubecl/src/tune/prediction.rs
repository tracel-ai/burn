//! A candidate's time predicted from its routine's estimate ([`Peaks::time_of`]), against the
//! problem's own roofline ([`Peaks::floor_of`]): what a family places a candidate in its round by.

use super::CandidatePriority;
use core::time::Duration;
use cubecl::client::Client;
use cubecl::std::throughput::{compute_bound, measure_launch_overhead, memory_bound};
use cubecl::throughput::{MemoryAccess, ResourceBound, ThroughputKey};
use cubecl::tune::Work;
use cubek::std::Hardware;
use cubek::std::microkernel::{GlobalStream, KernelEstimate};
use cubek::std::routine::Routine;

/// What a device sustains, as its probes measured it: what a prediction divides a kernel's
/// traffic and arithmetic by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peaks {
    /// Bytes a second a copy of the problem's footprint sustains.
    pub memory_bytes_per_s: f64,
    /// Operations a second the instruction the kernel computes on sustains.
    pub compute_ops_per_s: f64,
    /// What one launch costs on top of its kernel.
    pub launch_overhead: Duration,
}

impl Peaks {
    /// The peaks `client`'s device sustains for a problem of `demand`, computing on `arithmetic`.
    ///
    /// The first call for a footprint or an instruction runs cubecl's probe of it; later calls
    /// read its cache. A peak the device cannot measure is zero, which predicts nothing.
    pub fn measured(client: &Client, demand: Work, arithmetic: ThroughputKey) -> Self {
        // A threshold of one is the peak itself: only the rate is read.
        let memory = memory_bound(client, MemoryAccess::Copy, demand, 1.0);
        let compute = compute_bound(client, arithmetic, demand, 1.0);
        Self {
            memory_bytes_per_s: memory.resource.peak_per_s,
            compute_ops_per_s: compute.resource.peak_per_s,
            launch_overhead: measure_launch_overhead(client),
        }
    }

    /// The time no kernel of a problem of `demand` beats: its bytes moved once and its
    /// operations done once at these peaks, whichever takes longer, and a launch.
    ///
    /// `None` where neither peak was measured.
    pub fn floor_of(&self, demand: Work) -> Option<Duration> {
        let memory = self.memory_time(demand.bytes as f64);
        let compute = self.compute_time(demand.compute_ops as f64);
        let busy = memory.max(compute)?;
        Some(busy + self.launch_overhead)
    }

    /// The time the kernel `estimate` describes takes on `hw` at these peaks, or `None` where
    /// the estimate leaves its cubes or its staging unstated, or neither peak was measured.
    ///
    /// Every stream moves whole memory transactions, so a run shorter than one costs the
    /// transaction. The cubes' traffic and arithmetic overlap, so the longer of the two is the
    /// time the device is busy, stretched by the units a launch leaves idle: the slots of every
    /// wave the cubes take, over the units they hold.
    pub fn time_of(&self, estimate: &KernelEstimate, hw: &Hardware) -> Option<Duration> {
        let staging = estimate.staging.as_ref()?;
        let cubes = estimate.cubes?;
        let transaction = hw.transaction_bytes();
        let moved: f64 = staging
            .reads
            .iter()
            .chain(&staging.writes)
            .map(|stream| transacted_bytes(stream, transaction))
            .sum();
        let memory = self.memory_time(moved * cubes as f64);
        let compute = self.compute_time(staging.operations * cubes as f64);
        let busy = memory.max(compute)?;
        let units = cubes.saturating_mul(u64::from(estimate.cube.units));
        Some(busy.mul_f64(idle_stretch(units, hw.units_to_fill())) + self.launch_overhead)
    }

    /// Where the kernel `estimate` describes goes in a round for a problem of `demand` on `hw`:
    /// how far its predicted time sits above the problem's floor
    /// ([`CandidatePriority::above_floor`]), or first where either cannot be predicted.
    pub fn priority_of(
        &self,
        estimate: &KernelEstimate,
        demand: Work,
        hw: &Hardware,
    ) -> CandidatePriority {
        match (self.time_of(estimate, hw), self.floor_of(demand)) {
            (Some(predicted), Some(floor)) => CandidatePriority::above_floor(predicted, floor),
            _ => CandidatePriority::FIRST,
        }
    }

    fn memory_time(&self, bytes: f64) -> Option<Duration> {
        time_at(bytes, self.memory_bytes_per_s)
    }

    fn compute_time(&self, operations: f64) -> Option<Duration> {
        time_at(operations, self.compute_ops_per_s)
    }
}

/// The estimate of `R` at `settings` for `problem` on `hw`: its grid, then every part its owners
/// set. `None` where the routine declines the problem.
pub(crate) fn routine_estimate<R: Routine>(
    hw: &Hardware,
    problem: &R::Problem,
    settings: &R::Settings,
) -> Option<KernelEstimate> {
    let (blueprint, grid) = R::prepare(hw, problem, settings).ok()?;
    let mut estimate = KernelEstimate::from(&grid);
    R::estimate(problem, &blueprint, &grid, &mut estimate);
    Some(estimate)
}

/// `amount` at `per_s`, through cubecl's own reading of a peak: `None` for one it does not
/// measure.
fn time_at(amount: f64, per_s: f64) -> Option<Duration> {
    ResourceBound {
        amount: amount as usize,
        peak_per_s: per_s,
    }
    .time_at_peak()
}

/// Bytes `stream` moves in whole transactions of `transaction` bytes: each run rounded up to the
/// transactions it touches.
fn transacted_bytes(stream: &GlobalStream, transaction: usize) -> f64 {
    let run = stream.run_bytes.max(1);
    let touched = run.div_ceil(transaction) * transaction;
    stream.bytes as f64 * touched as f64 / run as f64
}

/// How much longer `units` take than a device kept full would: the slots of every wave of
/// `fill` units they take, over the units themselves. One where the device does not say how
/// wide it is.
fn idle_stretch(units: u64, fill: Option<u32>) -> f64 {
    let Some(fill) = fill.map(u64::from).filter(|&fill| fill > 0) else {
        return 1.0;
    };
    let units = units.max(1);
    (units.div_ceil(fill) * fill) as f64 / units as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek::std::devices;
    use cubek::std::microkernel::{CubeFootprint, CubeStaging};

    /// One byte a nanosecond and one operation a nanosecond, with a launch that costs nothing:
    /// a prediction reads in nanoseconds.
    const PEAKS: Peaks = Peaks {
        memory_bytes_per_s: 1e9,
        compute_ops_per_s: 1e9,
        launch_overhead: Duration::ZERO,
    };

    /// `cubes` of `units` units, each reading `bytes` in runs of `run_bytes` and computing
    /// `operations`.
    fn estimate(
        cubes: u64,
        units: u32,
        bytes: u64,
        run_bytes: usize,
        operations: f64,
    ) -> KernelEstimate {
        KernelEstimate {
            cubes: Some(cubes),
            cube: CubeFootprint {
                units,
                shared_memory: 0,
            },
            demand: None,
            staging: Some(CubeStaging {
                operations,
                reads: vec![GlobalStream { bytes, run_bytes }],
                writes: Vec::new(),
                stages: 1,
                stages_in_flight: 1,
                drain_atomics: 0,
            }),
            work: None,
        }
    }

    /// Units filling the device a whole number of times run at its peaks.
    fn filling(hw: &Hardware) -> (u64, u32) {
        let fill = hw.units_to_fill().expect("a GPU says how wide it is");
        (u64::from(fill / 256), 256)
    }

    #[test]
    fn a_full_device_runs_at_the_peaks() {
        let hw = devices::strix_halo();
        let (cubes, units) = filling(&hw);
        let predicted = PEAKS.time_of(&estimate(cubes, units, 1024, 1024, 0.0), &hw);
        assert_eq!(predicted, Some(Duration::from_nanos(cubes * 1024)));
    }

    /// The longer of traffic and arithmetic is the time: they overlap.
    #[test]
    fn the_slower_of_traffic_and_arithmetic_is_the_time() {
        let hw = devices::strix_halo();
        let (cubes, units) = filling(&hw);
        let predicted = PEAKS.time_of(&estimate(cubes, units, 1024, 1024, 4096.0), &hw);
        assert_eq!(predicted, Some(Duration::from_nanos(cubes * 4096)));
    }

    /// A run shorter than a transaction pays the whole transaction.
    #[test]
    fn a_short_run_moves_whole_transactions() {
        let hw = devices::strix_halo();
        let (cubes, units) = filling(&hw);
        let transaction = hw.transaction_bytes();
        let short = PEAKS.time_of(&estimate(cubes, units, 1024, transaction / 4, 0.0), &hw);
        let whole = PEAKS.time_of(&estimate(cubes, units, 1024, transaction, 0.0), &hw);
        assert_eq!(short, whole.map(|time| time * 4));
    }

    /// A grid of half the units the device wants leaves half of it idle, and takes as long as a
    /// full one.
    #[test]
    fn a_grid_that_leaves_the_device_idle_takes_as_long_as_a_full_one() {
        let hw = devices::strix_halo();
        let (cubes, units) = filling(&hw);
        let half = PEAKS.time_of(&estimate(cubes / 2, units, 1024, 1024, 0.0), &hw);
        let full = PEAKS.time_of(&estimate(cubes, units, 1024, 1024, 0.0), &hw);
        assert_eq!(half, full);
    }

    /// A grid a cube past a whole wave takes a second wave.
    #[test]
    fn a_grid_past_a_wave_takes_the_next_one() {
        let hw = devices::strix_halo();
        let (cubes, units) = filling(&hw);
        let past = PEAKS.time_of(&estimate(cubes + 1, units, 1024, 1024, 0.0), &hw);
        let two = PEAKS.time_of(&estimate(2 * cubes, units, 1024, 1024, 0.0), &hw);
        assert_eq!(past, two);
    }

    #[test]
    fn an_unmeasured_peak_predicts_nothing() {
        let hw = devices::strix_halo();
        let unmeasured = Peaks {
            memory_bytes_per_s: 0.0,
            compute_ops_per_s: 0.0,
            launch_overhead: Duration::ZERO,
        };
        assert_eq!(
            unmeasured.time_of(&estimate(1, 256, 1024, 1024, 0.0), &hw),
            None
        );
        assert_eq!(
            unmeasured.priority_of(
                &estimate(1, 256, 1024, 1024, 0.0),
                Work {
                    compute_ops: 0,
                    bytes: 1024,
                },
                &hw
            ),
            CandidatePriority::FIRST
        );
    }

    #[test]
    fn the_floor_is_the_slower_resource_and_a_launch() {
        let peaks = Peaks {
            launch_overhead: Duration::from_nanos(5),
            ..PEAKS
        };
        let demand = Work {
            compute_ops: 300,
            bytes: 200,
        };
        assert_eq!(peaks.floor_of(demand), Some(Duration::from_nanos(305)));
    }
}
