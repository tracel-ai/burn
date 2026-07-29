//! Measures what the adaptive autotune scheduler costs and what it buys.
//!
//! One process = one configuration. The interesting comparison is between processes, so the
//! toggles all come from the environment (`CUBECL_AUTOTUNE_BENCH_ADAPTIVE`,
//! `CUBECL_AUTOTUNE_CACHE`, `CUBECL_AUTOTUNE_SHORT_CIRCUIT`) and the driver script runs the
//! same binary under each combination.
//!
//! For each matmul problem it reports two numbers:
//!
//! - `tune_ms`: wall clock of the first call on a cold key, i.e. compilation plus the whole
//!   benchmarking pass. This is the time the adaptive scheduler is trying to cut.
//! - `steady_us`: median of `STEADY_SAMPLES` calls afterwards, once the winner is cached. This
//!   is what a wrong pick costs, and is the precision half of the comparison.

use std::time::{Duration, Instant};

use burn::tensor::{Device, DeviceConfig, DeviceIndex, Distribution, FloatDType, Shape, Tensor};

/// Calls timed after tuning, to see how fast the kernel autotune actually settled on is.
const STEADY_SAMPLES: usize = 20;
/// Untimed calls between tuning and the steady-state measurement, to let clocks settle.
const STEADY_WARMUP: usize = 5;

#[derive(Clone, Copy)]
struct Problem {
    tag: &'static str,
    b: usize,
    m: usize,
    n: usize,
    k: usize,
}

impl Problem {
    fn shapes(&self) -> (Shape, Shape) {
        (
            [self.b, self.m, self.k].into(),
            [self.b, self.k, self.n].into(),
        )
    }

    fn flops(&self) -> u64 {
        2 * self.b as u64 * self.m as u64 * self.n as u64 * self.k as u64
    }
}

/// Sized for an integrated GPU: every problem below stays well under a gigabyte of working set
/// and a few GFLOP of work, so the whole sweep runs on hardware that cannot touch the 4096³ and
/// 8192³ shapes the normal matmul benches use.
const PROBLEMS: &[Problem] = &[
    // Squares, small to moderate.
    Problem {
        tag: "square_1x256",
        b: 1,
        m: 256,
        n: 256,
        k: 256,
    },
    Problem {
        tag: "square_1x512",
        b: 1,
        m: 512,
        n: 512,
        k: 512,
    },
    Problem {
        tag: "square_1x1024",
        b: 1,
        m: 1024,
        n: 1024,
        k: 1024,
    },
    Problem {
        tag: "square_4x512",
        b: 4,
        m: 512,
        n: 512,
        k: 512,
    },
    Problem {
        tag: "square_8x256",
        b: 8,
        m: 256,
        n: 256,
        k: 256,
    },
    // Non-power-of-two, to exercise the masked/bounds-checked paths.
    Problem {
        tag: "square_1x768",
        b: 1,
        m: 768,
        n: 768,
        k: 768,
    },
    // Rectangular.
    Problem {
        tag: "rect_1x1024x1024x256",
        b: 1,
        m: 1024,
        n: 1024,
        k: 256,
    },
    Problem {
        tag: "rect_1x2048x512x512",
        b: 1,
        m: 2048,
        n: 512,
        k: 512,
    },
    // Degenerate dims: these pick different tunable groups (MatVec / VecMat / Outer).
    Problem {
        tag: "matvec_1x2048x1x2048",
        b: 1,
        m: 2048,
        n: 1,
        k: 2048,
    },
    Problem {
        tag: "vecmat_1x1x2048x2048",
        b: 1,
        m: 1,
        n: 2048,
        k: 2048,
    },
    Problem {
        tag: "outer_1x1024x1024x1",
        b: 1,
        m: 1024,
        n: 1024,
        k: 1,
    },
    // Skinny with batch, the shapes where per-candidate launch overhead dominates.
    Problem {
        tag: "skinny_64x64x1024x64",
        b: 64,
        m: 64,
        n: 1024,
        k: 64,
    },
    Problem {
        tag: "skinny_64x1024x64x64",
        b: 64,
        m: 1024,
        n: 64,
        k: 64,
    },
    Problem {
        tag: "skinny_64x64x64x1024",
        b: 64,
        m: 64,
        n: 64,
        k: 1024,
    },
];

fn median(mut durations: Vec<Duration>) -> Duration {
    durations.sort_unstable();
    durations[durations.len() / 2]
}

fn main() {
    let dtype = match std::env::var("BENCH_DTYPE").as_deref() {
        Ok("f16") => FloatDType::F16,
        Ok("bf16") => FloatDType::BF16,
        Ok("flex32") => FloatDType::Flex32,
        _ => FloatDType::F32,
    };

    let mut device = Device::cuda(DeviceIndex::Default);
    // let mut device = Device::vulkan(DeviceKind::DefaultDevice);
    device
        .configure(DeviceConfig::default().float_dtype(dtype))
        .expect("device should accept the requested float dtype");

    let label = std::env::var("BENCH_LABEL").unwrap_or_else(|_| "run".to_string());

    eprintln!(
        "config: label={label} dtype={dtype:?} adaptive={} cache={} short_circuit={}",
        std::env::var("CUBECL_AUTOTUNE_BENCH_ADAPTIVE").unwrap_or_else(|_| "<default>".into()),
        std::env::var("CUBECL_AUTOTUNE_CACHE").unwrap_or_else(|_| "<default>".into()),
        std::env::var("CUBECL_AUTOTUNE_SHORT_CIRCUIT").unwrap_or_else(|_| "<default>".into()),
    );

    // One header per process; the driver concatenates the CSVs.
    println!("label,problem,b,m,n,k,tune_ms,steady_us,gflops");

    let mut total_tune = Duration::ZERO;

    for problem in PROBLEMS {
        let (shape_lhs, shape_rhs) = problem.shapes();
        let lhs: Tensor<3> = Tensor::random(shape_lhs, Distribution::Default, &device);
        let rhs: Tensor<3> = Tensor::random(shape_rhs, Distribution::Default, &device);
        // Tensor creation is not part of what is being measured, and `random` is itself a
        // kernel launch, so it is drained before the clock starts.
        device.sync().unwrap();

        // Cold call: compiles the candidates and runs the whole tuning pass.
        let start = Instant::now();
        let out = lhs.clone().matmul(rhs.clone());
        device.sync().unwrap();
        let tune = start.elapsed();
        drop(out);
        total_tune += tune;

        for _ in 0..STEADY_WARMUP {
            let out = lhs.clone().matmul(rhs.clone());
            device.sync().unwrap();
            drop(out);
        }

        let mut samples = Vec::with_capacity(STEADY_SAMPLES);
        for _ in 0..STEADY_SAMPLES {
            let start = Instant::now();
            let out = lhs.clone().matmul(rhs.clone());
            device.sync().unwrap();
            samples.push(start.elapsed());
            drop(out);
        }
        let steady = median(samples);

        let gflops = problem.flops() as f64 / steady.as_secs_f64() / 1e9;
        println!(
            "{label},{},{},{},{},{},{:.3},{:.1},{gflops:.2}",
            problem.tag,
            problem.b,
            problem.m,
            problem.n,
            problem.k,
            tune.as_secs_f64() * 1e3,
            steady.as_secs_f64() * 1e6,
        );
    }

    eprintln!(
        "total tuning wall clock for {label}: {:.3}s",
        total_tune.as_secs_f64()
    );
}
