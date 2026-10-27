//! Test-only switches for comparing the automatic solver with its fallbacks.
//!
//! Run with `BURN_SOLVE_BENCH_DEVICE=1 cargo test -p burn-linalg --release
//! --no-default-features --features std,cuda --lib compare_solve_paths --
//! --ignored --nocapture`. The device index defaults to zero.

#[derive(Clone, Copy)]
pub(super) struct Options {
    pub shared_panel: bool,
    pub shared_rhs: bool,
    pub plane_ops: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            shared_panel: true,
            shared_rhs: true,
            plane_ops: true,
        }
    }
}

#[cfg(all(feature = "std", feature = "cuda"))]
mod cuda {
    use super::Options;
    use crate::solve_cubecl::solve_impl;
    use burn_core::backend::{Backend, ops::FloatTensorOps};
    use burn_cubecl::{CubeBackend, CubeDevice, ops::into_data_sync, tensor::CubeTensor};
    use burn_std::{DType, TensorData};
    use std::{hint::black_box, time::Instant};

    // Row-permuted, diagonally dominant matrices exercise pivoting while keeping
    // the known solution well conditioned for both F32 and F64.
    fn inputs(
        device: &CubeDevice,
        batch: usize,
        n: usize,
        rhs: usize,
        dtype: DType,
    ) -> (CubeTensor, CubeTensor, Vec<f64>) {
        let mut a = vec![0.0; batch * n * n];
        let mut b = vec![0.0; batch * n * rhs];
        let mut expected = vec![0.0; batch * n * rhs];
        for matrix in 0..batch {
            for row in 0..n {
                for col in 0..rhs {
                    expected[(matrix * n + row) * rhs + col] =
                        1.0 + (row % 7) as f64 * 0.125 + col as f64 * 0.25;
                }
                let permuted_row = n - 1 - row;
                for col in 0..n {
                    let value = if row == col {
                        n as f64 + 1.0
                    } else {
                        ((row * 13 + col * 7 + matrix) % 17) as f64 / 32.0 - 0.25
                    };
                    a[(matrix * n + permuted_row) * n + col] = value;
                    for k in 0..rhs {
                        let x = 1.0 + (col % 7) as f64 * 0.125 + k as f64 * 0.25;
                        b[(matrix * n + permuted_row) * rhs + k] += value * x;
                    }
                }
            }
        }
        let a = TensorData::new(a, [batch, n, n]).convert_dtype(dtype);
        let b = TensorData::new(b, [batch, n, rhs]).convert_dtype(dtype);
        (
            CubeBackend::float_from_data(a, device),
            CubeBackend::float_from_data(b, device),
            expected,
        )
    }

    #[test]
    #[ignore = "manual CUDA performance comparison; run in release mode with --nocapture"]
    fn compare_solve_paths() {
        let index = std::env::var("BURN_SOLVE_BENCH_DEVICE")
            .map(|value| value.parse::<usize>().expect("invalid CUDA device index"))
            .unwrap_or(0);
        let device = CubeDevice::cuda(index).expect("CUDA device unavailable");
        let variants = [
            ("auto", Options::default()),
            (
                "register_panel",
                Options {
                    shared_panel: false,
                    ..Options::default()
                },
            ),
            (
                "blocked_rhs",
                Options {
                    shared_rhs: false,
                    ..Options::default()
                },
            ),
            (
                "no_subgroups",
                Options {
                    plane_ops: false,
                    ..Options::default()
                },
            ),
            (
                "fallback",
                Options {
                    shared_panel: false,
                    shared_rhs: false,
                    plane_ops: false,
                },
            ),
        ];
        let cases = [
            (1, 16, 1),
            (256, 16, 8),
            (64, 31, 8),
            (64, 32, 8),
            (1, 64, 1),
            (64, 65, 8),
            (1, 128, 32),
            (16, 129, 8),
            (1, 256, 1),
            (16, 256, 32),
            (1, 512, 8),
            (1, 1024, 8),
        ];
        println!("backend: {}", CubeBackend::name(&device));
        println!("dtype,batch,n,rhs,variant,median_us,p10_us,p90_us");
        for dtype in [DType::F32, DType::F64] {
            for (batch, n, rhs) in cases {
                let (a, b, expected) = inputs(&device, batch, n, rhs, dtype);
                let active: Vec<_> = variants
                    .iter()
                    .copied()
                    .filter(|(name, _)| {
                        (*name != "blocked_rhs" || n < 32)
                            && (*name != "no_subgroups" || dtype == DType::F32)
                    })
                    .collect();
                // Check every variant independently before warming up and timing.
                for (name, options) in &active {
                    let output = solve_impl(a.clone(), b.clone(), *options);
                    let actual = into_data_sync(output).convert_dtype(DType::F64);
                    let tolerance = if dtype == DType::F32 { 2e-4 } else { 1e-10 };
                    for (&actual, &expected) in
                        actual.as_slice::<f64>().unwrap().iter().zip(&expected)
                    {
                        assert!(
                            (actual - expected).abs() < tolerance,
                            "{name}: {dtype:?}, batch={batch}, n={n}, rhs={rhs}: {actual} != {expected}"
                        );
                    }
                    for _ in 0..5 {
                        black_box(solve_impl(a.clone(), b.clone(), *options));
                        CubeBackend::sync(&device).unwrap();
                    }
                }
                // Interleave variants and rotate their order to reduce clock and
                // thermal bias. Include allocations, launches, singularity readback
                // and completion; input construction/transfers and JIT are excluded.
                let mut samples = vec![Vec::new(); active.len()];
                for round in 0..40 {
                    for offset in 0..active.len() {
                        let index = (round + offset) % active.len();
                        let options = active[index].1;
                        CubeBackend::sync(&device).unwrap();
                        let start = Instant::now();
                        let output = solve_impl(a.clone(), b.clone(), options);
                        CubeBackend::sync(&device).unwrap();
                        samples[index].push(start.elapsed().as_secs_f64() * 1e6);
                        black_box(output);
                    }
                }
                for ((name, _), mut values) in active.into_iter().zip(samples) {
                    values.sort_by(f64::total_cmp);
                    println!(
                        "{dtype:?},{batch},{n},{rhs},{name},{:.2},{:.2},{:.2}",
                        (values[19] + values[20]) / 2.0,
                        values[4],
                        values[35]
                    );
                }
            }
        }
    }
}
