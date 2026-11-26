//! Probe of the prediction on matmul: per shape, every float routine's every setting, its
//! predicted time beside its time measured on device.
use burn_cubecl::{
    CubeDevice, cubecl,
    tune::{Arithmetic, Peaks, cold_read_bytes, touch},
};
use core::fmt::Display;
use cubecl::CubeTuneId;
use cubecl::client::Client;
use cubecl::ir::{ElemType, FloatKind};
use cubecl::throughput::{CmmaDims, ThroughputKey};
use cubek::matmul::kernels::plan::{MatmulCall, MatmulCallInput};
use cubek::matmul::kernels::tensor::PlainTensor;
use cubek::matmul::kernels::{
    CubeShuffleMatmulRoutine, CubeStagedAsyncCopyMatmulRoutine, CubeStagedMatmulRoutine,
    CubeStagedTmaFillerMatmulRoutine, CubeStagedTmaMatmulRoutine, CubeStagedTmaStoredMatmulRoutine,
    HypercubeLastArrivalMatmulRoutine, MatmulBlueprint, MatmulRoutine, MatmulRoutineCandidate,
    MatmulRoutineProblem, PlaneShuffleMatmulRoutine, UnitRegistersMatmulRoutine,
    UnitWholeKMatmulRoutine,
};
use cubek::microkernel::matmul::MatmulMicrokernel;
use cubek::std::microkernel::KernelEstimate;
use cubek::std::routine::{EstimatedProblem, Routine};
use cubek::std::{Hardware, Instructions};

const F16: ElemType = ElemType::Float(FloatKind::F16);
const F32: ElemType = ElemType::Float(FloatKind::F32);

struct Probe {
    client: Client,
    hw: Hardware,
    arithmetic: Arithmetic,
    shape: (usize, usize, usize),
    call: MatmulCall,
}

impl Probe {
    fn sweep<R, M, SM>(&self, name: &str, routine: R)
    where
        M: MatmulMicrokernel,
        SM: PartialEq,
        R: MatmulRoutine<M, SM>
            + Routine<
                Problem = MatmulRoutineProblem,
                Microkernel = M,
                Blueprint = MatmulBlueprint<M, SM>,
            > + Copy
            + Send
            + Sync,
        R::Settings: Display + Send + Sync,
        M: Send + Sync,
        SM: Send + Sync,
    {
        let key = self.call.key();
        let problem = MatmulRoutineProblem::from(&key.shape());
        let demand = problem.demand();
        let candidates = MatmulRoutineCandidate::list(routine, &self.hw);
        let prepared: Vec<_> = candidates.iter().map(|c| c.prepared(&key)).collect();
        for (i, candidate) in candidates.iter().enumerate() {
            let Some(this) = &prepared[i] else { continue };
            if prepared[..i].iter().any(|p| p.as_ref() == Some(this)) {
                continue;
            }
            let Ok((blueprint, grid)) = R::prepare(&self.hw, &problem, candidate.settings()) else {
                continue;
            };
            let mut estimate = KernelEstimate::from(&grid);
            R::estimate(&problem, &blueprint, &grid, &mut estimate);
            let peaks = Peaks::measured(&self.client, &self.hw, &estimate, demand, self.arithmetic);
            if i == 0 {
                eprintln!("peaks {name}: {peaks:?}");
            }
            let prio = peaks.priority_of(&estimate, demand, &self.hw).get();
            let us =
                |t: Option<core::time::Duration>| t.map_or(f64::NAN, |t| t.as_secs_f64() * 1e6);
            let pred = us(peaks.time_of(&estimate, demand, &self.hw));
            let floor = us(peaks.floor_of(demand));
            let Some(measured) = self.measure(|| candidate.launch(self.call.clone()).is_ok())
            else {
                continue;
            };
            let (m, n, k) = self.shape;
            let parts = peaks.prediction_of(&estimate, demand, &self.hw);
            let part =
                |f: fn(&burn_cubecl::tune::Prediction) -> f64| parts.as_ref().map_or(f64::NAN, f);
            let staging = estimate.staging.as_ref();
            println!(
                "{m},{n},{k},{name},\"{}\",{prio},{pred:.2},{floor:.2},{measured:.2},{},{},{:.2},{:.2},{:.2},{:.2},{:.3},{:.3},{},{},{:.2},{:?},{:.3e},{},{},{}",
                candidate.settings(),
                estimate.cubes.unwrap_or(0),
                estimate.cube.units,
                part(|p| p.memory.as_secs_f64() * 1e6),
                part(|p| p.cache.as_secs_f64() * 1e6),
                part(|p| p.shared.as_secs_f64() * 1e6),
                part(|p| p.compute.as_secs_f64() * 1e6),
                part(|p| p.bandwidth_share),
                part(|p| p.lanes_filled),
                parts.map_or(0, |p| p.worker_cubes),
                parts.map_or(0, |p| p.streams),
                part(|p| p.overlap),
                estimate.work.map(|w| w.instruction),
                staging.map_or(f64::NAN, |s| s.operations),
                staging.map_or(0, |s| s.stages),
                staging.map_or(0, |s| s.stages_in_flight),
                estimate.work.map_or(0, |w| w.registers),
            );
        }
    }

    /// The fastest of a few timed launches, each after a read past the last-level cache as the
    /// matmul autotune's own samples are, in microseconds, or `None` where it does not launch.
    fn measure(&self, launch: impl Fn() -> bool + Sync) -> Option<f64> {
        if std::env::var("PROBE_PREDICT_ONLY").is_ok() {
            return Some(f64::NAN);
        }
        for _ in 0..2 {
            if !launch() {
                return None;
            }
        }
        let cold = cold_read_bytes(&self.client);
        let time = || {
            touch(&self.client, cold);
            let (_, profile) = self.client.profile(&launch, "matmul_probe").ok()?;
            let ticks = cubecl::future::block_on(profile.resolve())?;
            Some(ticks.duration().as_secs_f64() * 1e6)
        };
        let samples = std::env::var("PROBE_SAMPLES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(9);
        (0..samples).filter_map(|_| time()).reduce(f64::min)
    }
}

fn main() {
    let device = CubeDevice::default();
    let client = device.client();
    let hw = Hardware::from_client(&client);
    let cmma = hw
        .properties
        .features
        .matmul
        .cmma
        .iter()
        .copied()
        .for_triple(F16, F16, F32)
        .largest()
        .map(|i| CmmaDims {
            m: i.m as usize,
            n: i.n as usize,
            k: i.k as usize,
        });
    eprintln!("cmma: {cmma:?}");
    for shift in 20..=28 {
        let bytes = 1u64 << shift;
        let rate = cubecl::std::throughput::measure_cache_read(&client, bytes);
        eprintln!("re-read {:>4} MB: {:7.0} GB/s", bytes >> 20, rate / 1e9);
    }
    if std::env::var("PROBE_CURVE_ONLY").is_ok() {
        return;
    }
    let arithmetic = Arithmetic {
        vector: ThroughputKey::compute(None, F16, F32),
        matrix: cmma.map(|dims| ThroughputKey::compute(Some(dims), F16, F32)),
    };
    let shapes: Vec<(usize, usize, usize)> = std::env::var("PROBE_SHAPES")
        .ok()
        .map(|v| {
            v.split(';')
                .map(|s| {
                    let d: Vec<usize> = s.split('x').map(|x| x.parse().unwrap()).collect();
                    (d[0], d[1], d[2])
                })
                .collect()
        })
        .unwrap_or(vec![
            (1, 4096, 4096),
            (16, 4096, 4096),
            (256, 4096, 4096),
            (2048, 4096, 4096),
        ]);
    println!(
        "m,n,k,routine,setting,priority,pred_us,floor_us,us,cubes,units,mem_us,cache_us,shared_us,comp_us,bw_share,lanes,worker_cubes,streams,overlap,instr,ops_per_cube,stages,in_flight,registers"
    );
    for (m, n, k) in shapes {
        eprintln!("=== {m}x{n}x{k} ===");
        let lhs = PlainTensor::zeros(&client, &[m, k], [k, 1], F16);
        let rhs = PlainTensor::zeros(&client, &[k, n], [n, 1], F16);
        let call = MatmulCall {
            hw: hw.clone(),
            client: client.clone(),
            tune_id: CubeTuneId::new(&client, &device),
            lhs: MatmulCallInput::plain(lhs),
            rhs: MatmulCallInput::plain(rhs),
            out: F16,
            destination: None,
        }
        .with_destination();
        let probe = Probe {
            client: client.clone(),
            hw: hw.clone(),
            arithmetic,
            shape: (m, n, k),
            call,
        };
        probe.sweep("plane_shuffle", PlaneShuffleMatmulRoutine);
        probe.sweep("cube_shuffle", CubeShuffleMatmulRoutine);
        probe.sweep("unit_whole_k", UnitWholeKMatmulRoutine);
        probe.sweep("cube_staged", CubeStagedMatmulRoutine);
        probe.sweep("cube_staged_async", CubeStagedAsyncCopyMatmulRoutine);
        probe.sweep("cube_staged_tma", CubeStagedTmaMatmulRoutine);
        probe.sweep("cube_staged_tma_filler", CubeStagedTmaFillerMatmulRoutine);
        probe.sweep("cube_staged_tma_stored", CubeStagedTmaStoredMatmulRoutine);
        probe.sweep("hypercube_last_arrival", HypercubeLastArrivalMatmulRoutine);
        probe.sweep("unit_registers", UnitRegistersMatmulRoutine);
    }
}
