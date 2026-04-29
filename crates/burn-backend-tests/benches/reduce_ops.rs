//! Benchmarks for reduction operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench reduce_ops --features simd
//! ```

#[path = "common/mod.rs"]
mod common;
use common::{BencherExt, TestBackend};

use burn_tensor::{FloatDType, Tensor, TensorData, backend::Backend};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Reduction ops Benchmarks");
    println!();
    divan::main();
    common::report_failures();
}

fn make_tensor_1d<B: Backend>(size: usize) -> Tensor<B, 1> {
    let data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
    Tensor::from_data(TensorData::new(data, [size]), &Default::default())
}

fn make_tensor_2d<B: Backend>(rows: usize, cols: usize) -> Tensor<B, 2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

fn make_tensor_3d<B: Backend>(d0: usize, d1: usize, d2: usize) -> Tensor<B, 3> {
    let size = d0 * d1 * d2;
    let data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
    Tensor::from_data(TensorData::new(data, [d0, d1, d2]), &Default::default())
}

macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            // Sum all elements
            #[divan::bench_group(name = "sum")]
            mod sum {
                use super::*;

                #[divan::bench]
                fn _1k(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024);
                    bencher.bench_synced(|| t.clone().sum());
                }

                #[divan::bench]
                fn _64k(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(64 * 1024);
                    bencher.bench_synced(|| t.clone().sum());
                }

                #[divan::bench]
                fn _1m(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024 * 1024);
                    bencher.bench_synced(|| t.clone().sum());
                }
            }

            // Sum along dimension
            #[divan::bench_group(name = "sum_dim")]
            mod sum_dim {
                use super::*;

                #[divan::bench]
                fn _256x256_dim0(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().sum_dim(0));
                }

                #[divan::bench]
                fn _256x256_dim1(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().sum_dim(1));
                }

                #[divan::bench]
                fn _1024x1024_dim0(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024);
                    bencher.bench_synced(|| t.clone().sum_dim(0));
                }

                #[divan::bench]
                fn _1024x1024_dim1(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024);
                    bencher.bench_synced(|| t.clone().sum_dim(1));
                }
            }

            // Mean along dimension
            #[divan::bench_group(name = "mean_dim")]
            mod mean_dim {
                use super::*;

                #[divan::bench]
                fn _256x256_dim1(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().mean_dim(1));
                }

                #[divan::bench]
                fn _1024x1024_dim1(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024);
                    bencher.bench_synced(|| t.clone().mean_dim(1));
                }
            }

            // Argmax
            #[divan::bench_group(name = "argmax")]
            mod argmax {
                use super::*;

                #[divan::bench]
                fn _1k(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024);
                    bencher.bench_synced(|| t.clone().argmax(0));
                }

                #[divan::bench]
                fn _256x256_dim1(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().argmax(1));
                }

                #[divan::bench]
                fn _1024x1024_dim1(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024);
                    bencher.bench_synced(|| t.clone().argmax(1));
                }
            }

            // Sum on transposed (non-contiguous)
            #[divan::bench_group(name = "sum_transposed")]
            mod sum_transposed {
                use super::*;

                #[divan::bench]
                fn _256x256(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256).transpose();
                    bencher.bench_synced(|| t.clone().sum());
                }

                #[divan::bench]
                fn _1024x1024(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024).transpose();
                    bencher.bench_synced(|| t.clone().sum());
                }
            }

            // Sum_dim on transposed tensor (tests dim_stride=1 optimization)
            #[divan::bench_group(name = "sum_dim_transposed")]
            mod sum_dim_transposed {
                use super::*;

                #[divan::bench]
                fn _256x256_dim0(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256).transpose();
                    bencher.bench_synced(|| t.clone().sum_dim(0));
                }

                #[divan::bench]
                fn _1024x1024_dim0(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024).transpose();
                    bencher.bench_synced(|| t.clone().sum_dim(0));
                }
            }

            // 3D reductions (batch-like)
            #[divan::bench_group(name = "sum_3d")]
            mod sum_3d {
                use super::*;

                #[divan::bench]
                fn _batch32_256x256_dim2(bencher: Bencher) {
                    let t = make_tensor_3d::<B>(32, 256, 256);
                    bencher.bench_synced(|| t.clone().sum_dim(2));
                }

                #[divan::bench]
                fn _batch32_256x256_dim1(bencher: Bencher) {
                    let t = make_tensor_3d::<B>(32, 256, 256);
                    bencher.bench_synced(|| t.clone().sum_dim(1));
                }
            }
        }
    };
}

bench_backend!(TestBackend, backend, "backend");

// ============================================================================
// f16 mean benches
//
// These exercise the half-precision mean / mean_dim fast path. Each tensor
// is cast to f16 once during setup so the bench measures only the reduction,
// not the cast. Sizes mirror the f32 sum_dim / mean_dim groups so before/after
// numbers can be compared apples-to-apples.
// ============================================================================

#[divan::bench_group(name = "f16")]
mod f16 {
    use super::*;

    type B = TestBackend;

    fn make_f16_2d(rows: usize, cols: usize) -> Tensor<B, 2> {
        make_tensor_2d::<B>(rows, cols).cast(FloatDType::F16)
    }

    fn make_f16_3d(d0: usize, d1: usize, d2: usize) -> Tensor<B, 3> {
        make_tensor_3d::<B>(d0, d1, d2).cast(FloatDType::F16)
    }

    #[divan::bench_group(name = "mean")]
    mod mean {
        use super::*;

        #[divan::bench]
        fn _64k(bencher: Bencher) {
            let t = make_f16_2d(256, 256);
            bencher.bench_synced(|| t.clone().mean());
        }

        #[divan::bench]
        fn _1m(bencher: Bencher) {
            let t = make_f16_2d(1024, 1024);
            bencher.bench_synced(|| t.clone().mean());
        }
    }

    #[divan::bench_group(name = "mean_dim")]
    mod mean_dim {
        use super::*;

        // Last-dim path (sum_rows_f32)
        #[divan::bench]
        fn _256x256_dim1(bencher: Bencher) {
            let t = make_f16_2d(256, 256);
            bencher.bench_synced(|| t.clone().mean_dim(1));
        }

        #[divan::bench]
        fn _1024x1024_dim1(bencher: Bencher) {
            let t = make_f16_2d(1024, 1024);
            bencher.bench_synced(|| t.clone().mean_dim(1));
        }

        // First-dim path (scatter_add_f32)
        #[divan::bench]
        fn _256x256_dim0(bencher: Bencher) {
            let t = make_f16_2d(256, 256);
            bencher.bench_synced(|| t.clone().mean_dim(0));
        }

        #[divan::bench]
        fn _1024x1024_dim0(bencher: Bencher) {
            let t = make_f16_2d(1024, 1024);
            bencher.bench_synced(|| t.clone().mean_dim(0));
        }

        // Middle-dim path (scatter_add_batched)
        #[divan::bench]
        fn _batch32_256x256_dim1(bencher: Bencher) {
            let t = make_f16_3d(32, 256, 256);
            bencher.bench_synced(|| t.clone().mean_dim(1));
        }
    }
}
