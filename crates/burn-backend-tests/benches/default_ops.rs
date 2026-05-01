//! Benchmarks for default ops.
//!
//! These ops previously used burn's default trait implementations (TensorData
//! round-trips, slice_assign loops, etc.) and now have direct implementations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench default_ops --features simd
//! ```

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{Int, Tensor, TensorData};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Default ops Benchmarks");
    println!();
    divan::main();
    common::report_failures();
}

const SMALL: usize = 64 * 64;
const MEDIUM: usize = 256 * 256;
const LARGE: usize = 1024 * 1024;

fn make_tensor_1d(size: usize) -> Tensor<1> {
    let data: Vec<f32> = (0..size)
        .map(|i| (i % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    Tensor::from_data(TensorData::new(data, [size]), &Default::default())
}

fn make_tensor_2d(rows: usize, cols: usize) -> Tensor<2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

macro_rules! bench_backend {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;
            use burn_tensor::Shape;

            // ---------------------------------------------------------------
            // sort (1D)
            // ---------------------------------------------------------------

            #[divan::bench_group(name = "sort_1d")]
            mod sort_1d {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor_1d(SMALL);
                    bencher.bench_synced(|| a.clone().sort(0));
                }

                #[divan::bench]
                fn medium(bencher: Bencher) {
                    let a = make_tensor_1d(MEDIUM);
                    bencher.bench_synced(|| a.clone().sort(0));
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor_1d(LARGE);
                    bencher.bench_synced(|| a.clone().sort(0));
                }
            }

            // ---------------------------------------------------------------
            // sort (2D along dim 1)
            // ---------------------------------------------------------------

            #[divan::bench_group(name = "sort_2d_dim1")]
            mod sort_2d {
                use super::*;

                #[divan::bench]
                fn rows64_cols64(bencher: Bencher) {
                    let a = make_tensor_2d(64, 64);
                    bencher.bench_synced(|| a.clone().sort(1));
                }

                #[divan::bench]
                fn rows256_cols256(bencher: Bencher) {
                    let a = make_tensor_2d(256, 256);
                    bencher.bench_synced(|| a.clone().sort(1));
                }

                #[divan::bench]
                fn rows1024_cols1024(bencher: Bencher) {
                    let a = make_tensor_2d(1024, 1024);
                    bencher.bench_synced(|| a.clone().sort(1));
                }
            }

            // ---------------------------------------------------------------
            // argsort (1D)
            // ---------------------------------------------------------------

            #[divan::bench_group(name = "argsort_1d")]
            mod argsort_1d {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let a = make_tensor_1d(SMALL);
                    bencher.bench_synced(|| a.clone().argsort(0));
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let a = make_tensor_1d(LARGE);
                    bencher.bench_synced(|| a.clone().argsort(0));
                }
            }

            // ---------------------------------------------------------------
            // repeat_dim
            // ---------------------------------------------------------------

            #[divan::bench_group(name = "repeat_dim")]
            mod repeat_dim {
                use super::*;

                #[divan::bench]
                fn repeat_dim0_4x(bencher: Bencher) {
                    let a = make_tensor_2d(256, 256);
                    bencher.bench_synced(|| a.clone().repeat_dim(0, 4));
                }

                #[divan::bench]
                fn repeat_dim1_4x(bencher: Bencher) {
                    let a = make_tensor_2d(256, 256);
                    bencher.bench_synced(|| a.clone().repeat_dim(1, 4));
                }

                #[divan::bench]
                fn repeat_large_dim0_8x(bencher: Bencher) {
                    let a = make_tensor_2d(512, 512);
                    bencher.bench_synced(|| a.clone().repeat_dim(0, 8));
                }
            }

            // ---------------------------------------------------------------
            // zeros / ones / full
            // ---------------------------------------------------------------

            #[divan::bench_group(name = "creation")]
            mod creation {
                use super::*;

                #[divan::bench]
                fn zeros_large(bencher: Bencher) {
                    let dev = Default::default();
                    bencher.bench_synced(|| Tensor::<1>::zeros(Shape::new([LARGE]), &dev));
                }

                #[divan::bench]
                fn ones_large(bencher: Bencher) {
                    let dev = Default::default();
                    bencher.bench_synced(|| Tensor::<1>::ones(Shape::new([LARGE]), &dev));
                }

                #[divan::bench]
                fn full_large(bencher: Bencher) {
                    let dev = Default::default();
                    bencher.bench_synced(|| Tensor::<1>::full(Shape::new([LARGE]), 3.14, &dev));
                }
            }

            // ---------------------------------------------------------------
            // arange
            // ---------------------------------------------------------------

            #[divan::bench_group(name = "arange")]
            mod arange {
                use super::*;

                #[divan::bench]
                fn small(bencher: Bencher) {
                    let dev = Default::default();
                    bencher.bench_synced(|| Tensor::<1, Int>::arange(0..SMALL as i64, &dev));
                }

                #[divan::bench]
                fn large(bencher: Bencher) {
                    let dev = Default::default();
                    bencher.bench_synced(|| Tensor::<1, Int>::arange(0..LARGE as i64, &dev));
                }
            }

            // ---------------------------------------------------------------
            // embedding
            // ---------------------------------------------------------------

            #[divan::bench_group(name = "embedding")]
            mod embedding {
                use super::*;
                use burn_tensor::module;

                fn make_weights(vocab: usize, dim: usize) -> Tensor<2> {
                    let data: Vec<f32> = (0..vocab * dim)
                        .map(|i| (i % 1000) as f32 / 1000.0)
                        .collect();
                    Tensor::from_data(TensorData::new(data, [vocab, dim]), &Default::default())
                }

                fn make_indices(batch: usize, seq: usize, vocab: usize) -> Option<Tensor<2, Int>> {
                    common::try_setup(|| {
                        let data: Vec<i32> = (0..batch * seq).map(|i| (i % vocab) as i32).collect();
                        Tensor::from_data(TensorData::new(data, [batch, seq]), &Default::default())
                    })
                }

                #[divan::bench]
                fn vocab30k_dim512_batch8_seq128(bencher: Bencher) {
                    let weights = make_weights(30000, 512);
                    let Some(indices) = make_indices(8, 128, 30000) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| module::embedding(weights.clone(), indices.clone()));
                }

                #[divan::bench]
                fn vocab50k_dim768_batch4_seq256(bencher: Bencher) {
                    let weights = make_weights(50000, 768);
                    let Some(indices) = make_indices(4, 256, 50000) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| module::embedding(weights.clone(), indices.clone()));
                }
            }

            // ---------------------------------------------------------------
            // is_nan / is_inf
            // ---------------------------------------------------------------

            #[divan::bench_group(name = "predicates")]
            mod predicates {
                use super::*;

                #[divan::bench]
                fn is_nan_large(bencher: Bencher) {
                    let a = make_tensor_1d(LARGE);
                    bencher.bench_synced(|| a.clone().is_nan());
                }

                #[divan::bench]
                fn is_inf_large(bencher: Bencher) {
                    let a = make_tensor_1d(LARGE);
                    bencher.bench_synced(|| a.clone().is_inf());
                }
            }
        }
    };
}

bench_backend!(backend, "backend");
