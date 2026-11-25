//! A candidate's time predicted from its routine's estimate ([`Peaks::time_of`]), against the
//! problem's own roofline ([`Peaks::floor_of`]): what a family places a candidate in its round by.
//!
//! Every part of a [`KernelEstimate`] enters the prediction:
//!
//! - the cubes and their units, registers and shared memory decide how many cubes are resident
//!   at once ([`WorkerLimits`]), and so how many waves the grid's arithmetic takes and how full
//!   the last is — its traffic takes none, since the cubes of a short last wave share the whole
//!   bandwidth between them;
//! - the resident cubes' reads in flight — a stage's bytes times the stages a cube holds — decide
//!   how much of the memory latency is hidden, and so what fraction of the bandwidth is reached;
//! - every stream is moved in whole transactions, a run shorter than one paying for it, and no
//!   more of what the cubes request comes from memory than the problem's own bytes: the rest is
//!   what one cube reads again after another, served by the cache;
//! - what the cubes stage moves through shared memory at its bandwidth, as far as the resident
//!   cubes fill the device;
//! - the operations, and the plane-wide drain atomics, run at the rate of the instruction the
//!   work names, as far as the resident units fill the device's lanes;
//! - traffic and arithmetic overlap where a worker holds more than one cube or a cube more than
//!   one stage, and add up where neither does;
//! - the cubes take no less than the device needs to start and retire as many cubes of their
//!   size, however little each moves or computes.

use super::{CandidatePriority, WorkerLimits};
use core::time::Duration;
use cubecl::client::Client;
use cubecl::std::throughput::{
    compute_bound, measure_cache_read, measure_cube_rate, measure_launch_overhead,
    measure_shared_read, memory_bound,
};
use cubecl::throughput::{MemoryAccess, ResourceBound, ThroughputKey};
use cubecl::tune::Work;
use cubek::std::Hardware;
use cubek::std::microkernel::{ArithmeticInstruction, GlobalStream, KernelEstimate};
use cubek::std::routine::Routine;

/// The instructions a family's kernels compute on: what their arithmetic is measured against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arithmetic {
    /// Plain arithmetic, one unit at a time.
    pub vector: ThroughputKey,
    /// The matrix instruction a plane issues together, where the family has one.
    pub matrix: Option<ThroughputKey>,
}

/// What a device sustains, as its probes measured it: what a prediction divides a kernel's
/// traffic and arithmetic by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peaks {
    /// Bytes a second a copy of the problem's footprint sustains.
    pub memory_bytes_per_s: f64,
    /// Bytes a second the device reads what its last-level cache holds at, every unit reading
    /// again what others read: the rate of a kernel's re-reads. `NaN` where not measured.
    pub cache_bytes_per_s: f64,
    /// Bytes a second the device reads its shared memory at. `NaN` where not measured.
    pub shared_bytes_per_s: f64,
    /// Operations a second plain arithmetic sustains.
    pub vector_ops_per_s: f64,
    /// Operations a second the family's matrix instruction sustains, where it has one.
    pub matrix_ops_per_s: Option<f64>,
    /// Cubes of the kernel's size a second the device starts and retires, `NaN` where it was
    /// not measured.
    pub cubes_per_s: f64,
    /// What one launch costs on top of its kernel.
    pub launch_overhead: Duration,
}

impl Peaks {
    /// The peaks `client`'s device, `hw`, sustains for the kernel `estimate` describes on a
    /// problem of `demand`, computing on `arithmetic`: its cubes' rate at their size, and its
    /// re-reads' at the footprint a wave of its resident cubes reads
    /// ([`reread_footprint`](Self::reread_footprint)).
    ///
    /// The first call for a footprint, an instruction or a cube size runs cubecl's probe of
    /// it; later calls read its cache. A peak the device cannot measure is zero, which predicts
    /// nothing, and an unmeasured cube rate bounds nothing.
    pub fn measured(
        client: &Client,
        hw: &Hardware,
        estimate: &KernelEstimate,
        demand: Work,
        arithmetic: Arithmetic,
    ) -> Self {
        let demand = estimate.demand.unwrap_or(demand);
        // A threshold of one is the peak itself: only the rate is read.
        let rate = |key| compute_bound(client, key, demand, 1.0).resource.peak_per_s;
        Self {
            memory_bytes_per_s: memory_bound(client, MemoryAccess::Copy, demand, 1.0)
                .resource
                .peak_per_s,
            cache_bytes_per_s: measure_cache_read(
                client,
                Self::reread_footprint(estimate, demand, hw),
            ),
            shared_bytes_per_s: measure_shared_read(client),
            vector_ops_per_s: rate(arithmetic.vector),
            matrix_ops_per_s: arithmetic.matrix.map(rate),
            cubes_per_s: measure_cube_rate(client, estimate.cube.units),
            launch_overhead: measure_launch_overhead(client),
        }
    }

    /// The bytes the cubes of `estimate` resident on `hw` at once read over their walk, no more
    /// than the problem of `demand` holds: the working set what one cube reads again is found in,
    /// which decides the level of the cache that serves it.
    pub fn reread_footprint(estimate: &KernelEstimate, demand: Work, hw: &Hardware) -> u64 {
        let reads = estimate.staging.as_ref().map_or(0.0, |staging| {
            transacted_bytes(&staging.reads, hw.transaction_bytes())
        });
        let resident = Residency::of(estimate, hw, WorkerLimits::of(hw))
            .map_or(1, |residency| residency.device_cubes);
        let active = estimate.cubes.unwrap_or(1).min(resident);
        ((active as f64 * reads) as u64).min(demand.bytes as u64)
    }

    /// The time no kernel of a problem of `demand` beats: its bytes moved once and its
    /// operations done once at the fastest instruction, whichever takes longer, and a launch.
    ///
    /// `None` where neither peak was measured.
    pub fn floor_of(&self, demand: Work) -> Option<Duration> {
        let fastest = self
            .matrix_ops_per_s
            .unwrap_or(0.0)
            .max(self.vector_ops_per_s);
        let memory = time_at(demand.bytes as f64, self.memory_bytes_per_s);
        let compute = time_at(demand.compute_ops as f64, fastest);
        Some(memory.max(compute)? + self.launch_overhead)
    }

    /// The time the kernel `estimate` describes takes on `hw` at these peaks, for a problem of
    /// `demand` — the estimate's own, where it states one.
    ///
    /// `None` where the estimate leaves its cubes or its staging unstated, where its work runs
    /// on a matrix instruction these peaks hold no rate for, or where a peak it is timed at was
    /// not measured.
    pub fn time_of(
        &self,
        estimate: &KernelEstimate,
        demand: Work,
        hw: &Hardware,
    ) -> Option<Duration> {
        self.predict(estimate, demand, hw, WorkerLimits::of(hw))
            .map(|prediction| prediction.time())
    }

    /// The parts [`time_of`](Self::time_of) adds up for the kernel `estimate` describes, each
    /// as the prediction weighed it: what a person reads to see why a candidate is placed where
    /// it is. `None` where `time_of` is.
    pub fn prediction_of(
        &self,
        estimate: &KernelEstimate,
        demand: Work,
        hw: &Hardware,
    ) -> Option<Prediction> {
        self.predict(estimate, demand, hw, WorkerLimits::of(hw))
    }

    /// Where the kernel `estimate` describes goes in a round for a problem of `demand` on `hw`:
    /// how far its predicted time sits above the problem's floor
    /// ([`CandidatePriority::above_floor`]), or first where either cannot be predicted. The
    /// estimate's own demand, where it states one, is the floor's.
    pub fn priority_of(
        &self,
        estimate: &KernelEstimate,
        demand: Work,
        hw: &Hardware,
    ) -> CandidatePriority {
        let demand = estimate.demand.unwrap_or(demand);
        match (self.time_of(estimate, demand, hw), self.floor_of(demand)) {
            (Some(predicted), Some(floor)) => CandidatePriority::above_floor(predicted, floor),
            _ => CandidatePriority::FIRST,
        }
    }
}

impl Peaks {
    /// [`time_of`](Self::time_of) for a worker that holds `limits`, `None` where its
    /// architecture is not known.
    fn predict(
        &self,
        estimate: &KernelEstimate,
        demand: Work,
        hw: &Hardware,
        limits: Option<WorkerLimits>,
    ) -> Option<Prediction> {
        let demand = estimate.demand.unwrap_or(demand);
        let staging = estimate.staging.as_ref()?;
        let cubes = estimate.cubes?.max(1);
        let residency = Residency::of(estimate, hw, limits)?;
        let active = cubes.min(residency.device_cubes) as f64;
        let transaction = hw.transaction_bytes();
        let read = transacted_bytes(&staging.reads, transaction);
        let written = transacted_bytes(&staging.writes, transaction);

        // Reads in flight: a stage's share of a cube's reads, times the stages it holds.
        let stage_reads = read / staging.stages.max(1) as f64;
        let in_flight = active * stage_reads * f64::from(staging.stages_in_flight.max(1));
        let bandwidth_share = latency_hidden(in_flight, hw);
        // What the cubes request past the problem's own bytes, one reading again what another
        // read, comes from the cache.
        let requested = cubes as f64 * (read + written);
        let from_memory = requested.min(demand.bytes as f64);
        let memory = time_at(from_memory / bandwidth_share, self.memory_bytes_per_s)?;
        let cache = optional_time(requested - from_memory, self.cache_bytes_per_s)?;

        // A drain atomic serializes its plane: every unit of it waits on one shared address.
        let plane = f64::from(hw.properties.hardware.plane_size_max.max(1));
        let operations = cubes as f64 * (staging.operations + staging.drain_atomics as f64 * plane);
        let lanes_filled = residency.lanes_filled(active, estimate.cube.units);
        let stretch = residency.last_wave_stretch(cubes);
        let compute = time_at(operations / lanes_filled, self.rate_of(estimate)?)?.mul_f64(stretch);
        let staged = cubes as f64 * (staging.shared_reads + staging.shared_writes) as f64;
        let shared =
            optional_time(staged / lanes_filled, self.shared_bytes_per_s)?.mul_f64(stretch);

        Some(Prediction {
            memory,
            cache,
            compute,
            shared,
            started: time_at(cubes as f64, self.cubes_per_s).unwrap_or_default(),
            launch_overhead: self.launch_overhead,
            overlapped: residency.overlaps(staging.stages_in_flight),
            bandwidth_share,
            lanes_filled,
            worker_cubes: residency.worker_cubes,
            device_cubes: residency.device_cubes,
        })
    }

    /// The rate the estimate's work computes at: its matrix instruction's, where it names one,
    /// else plain arithmetic's. `None` for a matrix instruction these peaks hold no rate for.
    fn rate_of(&self, estimate: &KernelEstimate) -> Option<f64> {
        match estimate.work.map(|work| work.instruction) {
            Some(ArithmeticInstruction::Matrix) => self.matrix_ops_per_s,
            _ => Some(self.vector_ops_per_s),
        }
    }
}

/// A predicted time, in the parts [`Peaks::time_of`] adds up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    /// The kernel's traffic from memory — no more than the problem's own bytes — at the share of
    /// the bandwidth its reads in flight reach.
    pub memory: Duration,
    /// What its cubes read again, one after another, at the cache's bandwidth.
    pub cache: Duration,
    /// Its arithmetic, at the share of the lanes its resident units fill, over every wave.
    pub compute: Duration,
    /// What its cubes stage, through shared memory, as far as they fill the device, over every
    /// wave.
    pub shared: Duration,
    /// What running its cubes takes at the device's cube rate, however little each does.
    pub started: Duration,
    /// What its launch costs on top.
    pub launch_overhead: Duration,
    /// Whether its traffic and its arithmetic overlap.
    pub overlapped: bool,
    /// The share of the bandwidth its reads in flight reach.
    pub bandwidth_share: f64,
    /// The share of the device's lanes its resident units fill.
    pub lanes_filled: f64,
    /// Its cubes resident on one worker.
    pub worker_cubes: u64,
    /// Its cubes resident on the whole device.
    pub device_cubes: u64,
}

impl Prediction {
    /// The predicted time, no less than running the cubes takes, and the launch.
    ///
    /// What a worker does itself — reading again what its cache holds, reading and writing
    /// what it staged, computing — issues from its own instructions one after another; what it
    /// waits on memory for overlaps all of it where it holds a second cube or a second stage,
    /// and adds to it where it holds neither.
    pub fn time(&self) -> Duration {
        let on_chip = self.cache + self.shared + self.compute;
        let busy = match self.overlapped {
            true => self.memory.max(on_chip),
            false => self.memory + on_chip,
        };
        busy.max(self.started) + self.launch_overhead
    }
}

/// How many cubes of one kernel a device runs at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Residency {
    /// Cubes resident on one worker.
    worker_cubes: u64,
    /// Cubes resident on the whole device.
    device_cubes: u64,
    /// Units issuing arithmetic each cycle across the device, where its architecture is known.
    device_lanes: Option<u64>,
}

impl Residency {
    /// The cubes of `estimate` resident on `hw` at once: as many as every limit of a worker
    /// allows — its units, its cubes, its registers and its shared memory — on every worker.
    /// Without a worker's limits, as many as fill the units the device wants in flight. `None`
    /// where the device says neither how wide it is nor how many workers it has.
    fn of(estimate: &KernelEstimate, hw: &Hardware, limits: Option<WorkerLimits>) -> Option<Self> {
        let units = u64::from(estimate.cube.units.max(1));
        let workers = hw
            .properties
            .hardware
            .num_streaming_multiprocessors
            .map(u64::from);
        let Some((limits, workers)) = limits.zip(workers) else {
            let device_cubes = (u64::from(hw.units_to_fill()?) / units).max(1);
            let worker_cubes = device_cubes / workers.unwrap_or(1).max(1);
            return Some(Self {
                worker_cubes,
                device_cubes,
                device_lanes: None,
            });
        };
        let registers = estimate.work.map_or(0, |work| u64::from(work.registers));
        let shared = estimate.cube.shared_memory as u64;
        let worker_cubes = [
            u64::from(limits.units_count) / units,
            u64::from(limits.cubes_count),
            u64::from(limits.registers_count)
                .checked_div(registers * units)
                .unwrap_or(u64::MAX),
            (limits.shared_memory_bytes as u64)
                .checked_div(shared)
                .unwrap_or(u64::MAX),
        ]
        .into_iter()
        .min()
        .unwrap_or(1)
        .max(1);
        Some(Self {
            worker_cubes,
            device_cubes: worker_cubes * workers,
            device_lanes: Some(u64::from(limits.lanes_count) * workers),
        })
    }

    /// The share of the device's lanes `active` cubes of `units` units keep issuing, one where
    /// the architecture is not known.
    fn lanes_filled(&self, active: f64, units: u32) -> f64 {
        self.device_lanes.map_or(1.0, |lanes| {
            (active * f64::from(units.max(1)) / lanes as f64).clamp(f64::MIN_POSITIVE, 1.0)
        })
    }

    /// Whether a kernel holding `stages_in_flight` stages overlaps its traffic with its
    /// arithmetic: where a worker holds a second cube or a cube a second stage to overlap them
    /// with.
    fn overlaps(&self, stages_in_flight: u32) -> bool {
        self.worker_cubes > 1 || stages_in_flight > 1
    }

    /// How much longer `cubes` take than whole waves of resident cubes would: the slots of
    /// every wave they take, over the cubes themselves.
    fn last_wave_stretch(&self, cubes: u64) -> f64 {
        let wave = self.device_cubes.max(1);
        let cubes = cubes.max(wave);
        (cubes.div_ceil(wave) * wave) as f64 / cubes as f64
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

/// The share of the bandwidth reads `in_flight` bytes at once reach: the bytes in flight over
/// those it takes to cover the memory's latency, which is what the units `hw` wants in flight
/// ([`Hardware::units_to_fill`]) each holding its widest load. One where the device says
/// nothing of its width, or nothing is read.
fn latency_hidden(in_flight: f64, hw: &Hardware) -> f64 {
    let load_bytes = f64::from(hw.properties.hardware.load_width / 8);
    let needed = hw
        .units_to_fill()
        .map(|units| f64::from(units) * load_bytes)
        .filter(|needed| *needed > 0.0 && in_flight > 0.0);
    needed.map_or(1.0, |needed| {
        (in_flight / needed).clamp(f64::MIN_POSITIVE, 1.0)
    })
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

/// [`time_at`] for an amount that may be nothing: no time whatever the rate, where `time_at`
/// would read an unmeasured peak as no time to give.
fn optional_time(amount: f64, per_s: f64) -> Option<Duration> {
    match amount > 0.0 {
        true => time_at(amount, per_s),
        false => Some(Duration::ZERO),
    }
}

/// Bytes `streams` move in whole transactions of `transaction` bytes: each run rounded up to the
/// transactions it touches.
fn transacted_bytes(streams: &[GlobalStream], transaction: usize) -> f64 {
    streams
        .iter()
        .map(|stream| {
            let run = stream.run_bytes.max(1);
            stream.bytes as f64 * (run.div_ceil(transaction) * transaction) as f64 / run as f64
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek::std::devices;
    use cubek::std::microkernel::{CubeFootprint, CubeStaging, UnitWork};

    /// One byte a nanosecond and one operation a nanosecond, with a launch that costs nothing:
    /// a prediction reads in nanoseconds.
    const PEAKS: Peaks = Peaks {
        memory_bytes_per_s: 1e9,
        cache_bytes_per_s: 4e9,
        shared_bytes_per_s: 8e9,
        vector_ops_per_s: 1e9,
        matrix_ops_per_s: None,
        cubes_per_s: f64::NAN,
        launch_overhead: Duration::ZERO,
    };

    /// A problem whose own bytes are as many as any kernel requests: every byte comes from
    /// memory.
    const ANY_DEMAND: Work = Work {
        compute_ops: 0,
        bytes: usize::MAX,
    };

    /// A worker holding 1024 units in at most 8 cubes, 64K registers, 64 KB of shared memory,
    /// 64 lanes.
    const LIMITS: WorkerLimits = WorkerLimits {
        units_count: 1024,
        cubes_count: 8,
        registers_count: 64 * 1024,
        shared_memory_bytes: 64 * 1024,
        lanes_count: 64,
    };

    /// A kernel of `cubes` cubes of 128 units, each reading `bytes` in full transactions, in
    /// one stage it holds alone, and computing nothing.
    fn reading(cubes: u64, bytes: u64) -> KernelEstimate {
        KernelEstimate {
            cubes: Some(cubes),
            cube: CubeFootprint {
                units: 128,
                shared_memory: 0,
            },
            demand: None,
            staging: Some(CubeStaging {
                operations: 0.0,
                reads: vec![GlobalStream {
                    bytes,
                    run_bytes: 1024,
                }],
                writes: Vec::new(),
                stages: 1,
                stages_in_flight: 1,
                drain_atomics: 0,
                shared_reads: 0,
                shared_writes: 0,
            }),
            work: Some(UnitWork {
                instruction: ArithmeticInstruction::Vector,
                registers: 16,
            }),
        }
    }

    /// Bytes a cube must read for the resident cubes of [`reading`] to hide the latency of a
    /// strix halo: what its wanted units each hold of their widest load, shared by every cube
    /// resident under [`LIMITS`].
    fn hiding_bytes(hw: &Hardware) -> u64 {
        let needed = u64::from(hw.units_to_fill().unwrap())
            * u64::from(hw.properties.hardware.load_width / 8);
        let resident = 8 * u64::from(
            hw.properties
                .hardware
                .num_streaming_multiprocessors
                .unwrap(),
        );
        needed.div_ceil(resident)
    }

    fn time(estimate: &KernelEstimate) -> Duration {
        let hw = devices::strix_halo();
        PEAKS
            .predict(estimate, ANY_DEMAND, &hw, Some(LIMITS))
            .expect("timed")
            .time()
    }

    /// `actual` within a percent of `expected`: a prediction is a float divided into a
    /// duration, and rounding the bytes a cube reads moves it by a little.
    fn assert_close(actual: Duration, expected: Duration) {
        let ratio = actual.as_secs_f64() / expected.as_secs_f64();
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "{actual:?} against {expected:?}"
        );
    }

    /// Cubes that fill every worker and hide the latency read at the bandwidth.
    #[test]
    fn a_full_device_hiding_its_latency_reads_at_the_bandwidth() {
        let hw = devices::strix_halo();
        let bytes = hiding_bytes(&hw);
        let cubes = 8 * 40;
        assert_close(
            time(&reading(cubes, bytes)),
            Duration::from_nanos(cubes * bytes),
        );
    }

    /// Too few bytes in flight leave the bus idle while the reads wait: half of them, half the
    /// bandwidth.
    #[test]
    fn too_few_reads_in_flight_reach_part_of_the_bandwidth() {
        let hw = devices::strix_halo();
        let bytes = hiding_bytes(&hw) / 2;
        let cubes = 8 * 40;
        assert_close(
            time(&reading(cubes, bytes)),
            Duration::from_nanos(2 * cubes * bytes),
        );
    }

    /// A cube holding a second stage of the same reads has twice the bytes in flight.
    #[test]
    fn stages_in_flight_hide_latency() {
        let hw = devices::strix_halo();
        let bytes = hiding_bytes(&hw);
        let mut staged = reading(8 * 40, bytes);
        let staging = staged.staging.as_mut().unwrap();
        staging.stages = 2;
        let single = time(&staged);
        staged.staging.as_mut().unwrap().stages_in_flight = 2;
        assert_close(time(&staged) * 2, single);
    }

    /// Shared memory a cube holds past a worker's share leaves fewer cubes resident, and so
    /// fewer reads in flight.
    #[test]
    fn shared_memory_limits_the_cubes_resident() {
        let hw = devices::strix_halo();
        let bytes = hiding_bytes(&hw);
        let free = reading(8 * 40, bytes);
        let mut staged = free.clone();
        staged.cube.shared_memory = LIMITS.shared_memory_bytes / 4;
        assert_close(time(&staged), time(&free) * 2);
    }

    /// Registers a cube's units hold past a worker's file leave fewer cubes resident.
    #[test]
    fn registers_limit_the_cubes_resident() {
        let hw = devices::strix_halo();
        let bytes = hiding_bytes(&hw);
        let free = reading(8 * 40, bytes);
        let mut heavy = free.clone();
        // 128 units of 128 registers fill a quarter of the file: four cubes a worker.
        heavy.work.as_mut().unwrap().registers = 128;
        assert_close(time(&heavy), time(&free) * 2);
    }

    /// A run shorter than a transaction pays the whole transaction.
    #[test]
    fn a_short_run_moves_whole_transactions() {
        let hw = devices::strix_halo();
        let bytes = hiding_bytes(&hw) * 4;
        let whole = reading(8 * 40, bytes);
        let mut short = whole.clone();
        short.staging.as_mut().unwrap().reads[0].run_bytes = hw.transaction_bytes() / 4;
        assert_close(time(&short), time(&whole) * 4);
    }

    /// `cubes` computing `operations` each and reading nothing.
    fn computing(cubes: u64, operations: f64) -> KernelEstimate {
        let mut kernel = reading(cubes, 0);
        kernel.staging.as_mut().unwrap().operations = operations;
        kernel
    }

    /// Arithmetic one cube past a whole wave of resident cubes takes a second wave.
    #[test]
    fn arithmetic_past_a_wave_takes_the_next_one() {
        let wave = 8 * 40;
        assert_close(
            time(&computing(wave + 1, 4096.0)),
            time(&computing(2 * wave, 4096.0)),
        );
    }

    /// Traffic one cube past a whole wave costs that cube's bytes alone: the short last wave
    /// shares the whole bandwidth.
    #[test]
    fn traffic_past_a_wave_costs_its_own_bytes() {
        let hw = devices::strix_halo();
        let bytes = hiding_bytes(&hw);
        let wave = 8 * 40;
        assert_close(
            time(&reading(wave + 1, bytes)),
            Duration::from_nanos((wave + 1) * bytes),
        );
    }

    /// A worker holding a single cube of a single stage cannot overlap its reads with its
    /// arithmetic: the two add up.
    #[test]
    fn one_cube_of_one_stage_adds_traffic_and_arithmetic() {
        let hw = devices::strix_halo();
        let alone = WorkerLimits {
            cubes_count: 1,
            ..LIMITS
        };
        let mut kernel = reading(40, 4096);
        kernel.staging.as_mut().unwrap().operations = 4096.0;
        let traffic_only = PEAKS
            .predict(&reading(40, 4096), ANY_DEMAND, &hw, Some(alone))
            .unwrap()
            .time();
        let both = PEAKS
            .predict(&kernel, ANY_DEMAND, &hw, Some(alone))
            .unwrap()
            .time();
        assert!(both > traffic_only);
        let overlapped = PEAKS
            .predict(&kernel, ANY_DEMAND, &hw, Some(LIMITS))
            .unwrap()
            .time();
        assert!(overlapped < both);
    }

    /// Drain atomics serialize their plane, so they cost arithmetic as a plane's worth each.
    #[test]
    fn drain_atomics_cost_a_plane_each() {
        let mut atomics = reading(8 * 40, 0);
        atomics.staging.as_mut().unwrap().drain_atomics = 1024;
        let mut plain = reading(8 * 40, 0);
        let plane = f64::from(devices::strix_halo().properties.hardware.plane_size_max);
        plain.staging.as_mut().unwrap().operations = 1024.0 * plane;
        assert_close(time(&atomics), time(&plain));
    }

    /// Cubes that each move little take as long as the device needs to start them all.
    #[test]
    fn many_small_cubes_take_as_long_as_starting_them() {
        let hw = devices::strix_halo();
        let started = Peaks {
            // A cube a microsecond.
            cubes_per_s: 1e6,
            ..PEAKS
        };
        let cubes = 8 * 40 * 4;
        let predicted = started
            .predict(&computing(cubes, 0.0), ANY_DEMAND, &hw, Some(LIMITS))
            .map(|p| p.time());
        assert_close(predicted.unwrap(), Duration::from_micros(cubes));
    }

    /// What cubes request past the problem's own bytes is read again from the cache, at its
    /// bandwidth: a quarter of the requests from memory, the rest four times as fast.
    #[test]
    fn reads_past_the_problems_bytes_come_from_the_cache() {
        let hw = devices::strix_halo();
        let bytes = hiding_bytes(&hw);
        let cubes = 8 * 40;
        let requested = cubes * bytes;
        let demand = Work {
            compute_ops: 0,
            bytes: (requested / 4) as usize,
        };
        let predicted = PEAKS
            .predict(&reading(cubes, bytes), demand, &hw, Some(LIMITS))
            .unwrap();
        assert_close(predicted.memory, Duration::from_nanos(requested / 4));
        assert_close(predicted.cache, Duration::from_nanos(requested * 3 / 4 / 4));
    }

    /// What a cube stages moves through shared memory at its bandwidth, and bounds the cube
    /// alongside its traffic and arithmetic.
    #[test]
    fn staging_moves_through_shared_memory() {
        let mut staged = computing(8 * 40, 0.0);
        let staging = staged.staging.as_mut().unwrap();
        staging.shared_reads = 6000;
        staging.shared_writes = 2000;
        // 8000 bytes a cube at 8 bytes a nanosecond.
        assert_close(time(&staged), Duration::from_nanos(8 * 40 * 1000));
    }

    /// What a worker does itself issues one after another: its arithmetic and its staging add
    /// up, where its memory traffic would overlap them.
    #[test]
    fn a_workers_own_work_adds_up() {
        let mut both = computing(8 * 40, 1000.0);
        both.staging.as_mut().unwrap().shared_reads = 8000;
        // 1000 operations and 8000 staged bytes a cube: a microsecond each.
        assert_close(time(&both), Duration::from_nanos(8 * 40 * 2000));
    }

    /// Work on a matrix instruction the peaks hold no rate for predicts nothing.
    #[test]
    fn a_matrix_instruction_without_its_peak_predicts_nothing() {
        let hw = devices::strix_halo();
        let mut matrix = reading(8 * 40, 1024);
        matrix.work.as_mut().unwrap().instruction = ArithmeticInstruction::Matrix;
        assert_eq!(PEAKS.predict(&matrix, ANY_DEMAND, &hw, Some(LIMITS)), None);
    }

    #[test]
    fn an_unmeasured_peak_predicts_first() {
        let hw = devices::strix_halo();
        let unmeasured = Peaks {
            memory_bytes_per_s: 0.0,
            cache_bytes_per_s: 0.0,
            shared_bytes_per_s: 0.0,
            vector_ops_per_s: 0.0,
            matrix_ops_per_s: None,
            cubes_per_s: f64::NAN,
            launch_overhead: Duration::ZERO,
        };
        let demand = Work {
            compute_ops: 0,
            bytes: 1024,
        };
        assert_eq!(
            unmeasured.priority_of(&reading(1, 1024), demand, &hw),
            CandidatePriority::FIRST
        );
    }

    /// The floor is the slower resource at the fastest instruction, and a launch.
    #[test]
    fn the_floor_is_the_slower_resource_and_a_launch() {
        let peaks = Peaks {
            matrix_ops_per_s: Some(3e9),
            launch_overhead: Duration::from_nanos(5),
            ..PEAKS
        };
        let demand = Work {
            compute_ops: 900,
            bytes: 200,
        };
        assert_close(peaks.floor_of(demand).unwrap(), Duration::from_nanos(305));
    }
}
