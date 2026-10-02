//! Benchmarks for topk, argtopk, and topk_with_indices.
//!
//! Run with:
//! ```bash
//! cargo bench --bench topk_ops --features ndarray
//! ```

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{Int, Tensor, TensorData};
use divan::Bencher;

#[cfg(not(feature = "bench-disable-alloc"))]
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    println!("Benchmarks for topk, argtopk, topk_with_indices");
    println!("Memory allocation tracking enabled");
    println!();
    divan::main();
    common::report_failures();
}

// === Helpers ===

fn make_f32_1d(size: usize) -> Tensor<1> {
    let data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
    Tensor::from_data(TensorData::new(data, [size]), &Default::default())
}

fn make_f32_2d(rows: usize, cols: usize) -> Tensor<2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

fn make_int_2d(rows: usize, cols: usize) -> Option<Tensor<2, Int>> {
    common::try_setup(|| {
        let data: Vec<i32> = (0..rows * cols).map(|i| (i % 10) as i32 + 1).collect();
        Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
    })
}

// =============================================================================
// Topk
// =============================================================================

macro_rules! bench_topk {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            #[divan::bench_group(name = "topk")]
            mod topk {
                use super::*;

                // Large axis, small k: the case partial selection wins on
                // (vocab-style logits). O(n + k log k) vs a full O(n log n)
                // sort of each lane.
                #[divan::bench]
                fn s_1d_1m_k5(bencher: Bencher) {
                    let t = make_f32_1d(1024 * 1024);
                    bencher.bench_synced(|| t.clone().topk(5, 0));
                }

                #[divan::bench]
                fn s_256x50k_k5_dim1(bencher: Bencher) {
                    let t = make_f32_2d(256, 50_000);
                    bencher.bench_synced(|| t.clone().topk(5, 1));
                }

                // Break-even-ish: k approaches the axis length.
                #[divan::bench]
                fn s_256x256_k128_dim1(bencher: Bencher) {
                    let t = make_f32_2d(256, 256);
                    bencher.bench_synced(|| t.clone().topk(128, 1));
                }

                // Outer-dim reduction (non-contiguous lanes).
                #[divan::bench]
                fn s_50kx256_k5_dim0(bencher: Bencher) {
                    let t = make_f32_2d(50_000, 256);
                    bencher.bench_synced(|| t.clone().topk(5, 0));
                }
            }

            #[divan::bench_group(name = "argtopk")]
            mod argtopk {
                use super::*;

                #[divan::bench]
                fn s_1d_1m_k5(bencher: Bencher) {
                    let t = make_f32_1d(1024 * 1024);
                    bencher.bench_synced(|| t.clone().argtopk(5, 0));
                }

                #[divan::bench]
                fn s_256x50k_k5_dim1(bencher: Bencher) {
                    let t = make_f32_2d(256, 50_000);
                    bencher.bench_synced(|| t.clone().argtopk(5, 1));
                }
            }

            #[divan::bench_group(name = "topk_with_indices")]
            mod topk_with_indices {
                use super::*;

                #[divan::bench]
                fn s_1d_1m_k5(bencher: Bencher) {
                    let t = make_f32_1d(1024 * 1024);
                    bencher.bench_synced(|| t.clone().topk_with_indices(5, 0));
                }

                #[divan::bench]
                fn s_256x50k_k5_dim1(bencher: Bencher) {
                    let t = make_f32_2d(256, 50_000);
                    bencher.bench_synced(|| t.clone().topk_with_indices(5, 1));
                }
            }

            #[divan::bench_group(name = "int_topk")]
            mod int_topk {
                use super::*;

                #[divan::bench]
                fn s_256x50k_k5_dim1(bencher: Bencher) {
                    let Some(t) = make_int_2d(256, 50_000) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| t.clone().topk(5, 1));
                }
            }
        }
    };
}

bench_topk!(backend, "backend");
