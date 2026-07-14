//! Benchmarks for cat, max, min, int power, bool select, and grid_sample_2d.
//!
//! Run with:
//! ```bash
//! cargo bench --bench cat_max_min_ops --features simd,rayon
//! ```

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{Bool, Int, Tensor, TensorData, ops::GridSampleOptions};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Benchmarks for cat, max/min, int power, bool select, grid_sample_2d");
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

fn make_int_exp_2d(rows: usize, cols: usize) -> Option<Tensor<2, Int>> {
    common::try_setup(|| {
        let data: Vec<i32> = (0..rows * cols).map(|i| (i % 4) as i32 + 1).collect();
        Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
    })
}

fn make_bool_2d(rows: usize, cols: usize) -> Tensor<2, Bool> {
    let data: Vec<bool> = (0..rows * cols).map(|i| i % 3 != 0).collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

fn make_indices_1d(size: usize, max_idx: usize) -> Option<Tensor<1, Int>> {
    common::try_setup(|| {
        let data: Vec<i32> = (0..size).map(|i| (i % max_idx) as i32).collect();
        Tensor::from_data(TensorData::new(data, [size]), &Default::default())
    })
}

fn make_grid_4d(batch: usize, h_out: usize, w_out: usize) -> Tensor<4> {
    let size = batch * h_out * w_out * 2;
    // Grid values in [-1, 1]
    let data: Vec<f32> = (0..size)
        .map(|i| (i as f32 / size as f32) * 2.0 - 1.0)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [batch, h_out, w_out, 2]),
        &Default::default(),
    )
}

fn make_image_4d(batch: usize, channels: usize, h: usize, w: usize) -> Tensor<4> {
    let size = batch * channels * h * w;
    let data: Vec<f32> = (0..size).map(|i| (i % 256) as f32 / 255.0).collect();
    Tensor::from_data(
        TensorData::new(data, [batch, channels, h, w]),
        &Default::default(),
    )
}

// =============================================================================
// Cat
// =============================================================================

macro_rules! bench_cat {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            #[divan::bench_group(name = "cat")]
            mod cat {
                use super::*;

                // Cat along dim 0 (fast path: contiguous memcpy)
                #[divan::bench]
                fn dim0_4x_256x256(bencher: Bencher) {
                    let t = make_f32_2d(256, 256);
                    let tensors = vec![t.clone(), t.clone(), t.clone(), t.clone()];
                    bencher.bench_synced(|| Tensor::cat(tensors.clone(), 0));
                }

                #[divan::bench]
                fn dim0_4x_1024x256(bencher: Bencher) {
                    let t = make_f32_2d(1024, 256);
                    let tensors = vec![t.clone(), t.clone(), t.clone(), t.clone()];
                    bencher.bench_synced(|| Tensor::cat(tensors.clone(), 0));
                }

                // Cat along dim 1 (general path)
                #[divan::bench]
                fn dim1_4x_256x64(bencher: Bencher) {
                    let t = make_f32_2d(256, 64);
                    let tensors = vec![t.clone(), t.clone(), t.clone(), t.clone()];
                    bencher.bench_synced(|| Tensor::cat(tensors.clone(), 1));
                }

                #[divan::bench]
                fn dim1_4x_1024x64(bencher: Bencher) {
                    let t = make_f32_2d(1024, 64);
                    let tensors = vec![t.clone(), t.clone(), t.clone(), t.clone()];
                    bencher.bench_synced(|| Tensor::cat(tensors.clone(), 1));
                }

                // Many small tensors
                #[divan::bench]
                fn dim0_16x_64x64(bencher: Bencher) {
                    let t = make_f32_2d(64, 64);
                    let tensors: Vec<_> = (0..16).map(|_| t.clone()).collect();
                    bencher.bench_synced(|| Tensor::cat(tensors.clone(), 0));
                }

                // 1D cat
                #[divan::bench]
                fn dim0_4x_16k_1d(bencher: Bencher) {
                    let t = make_f32_1d(16 * 1024);
                    let tensors = vec![t.clone(), t.clone(), t.clone(), t.clone()];
                    bencher.bench_synced(|| Tensor::cat(tensors.clone(), 0));
                }
            }

            #[divan::bench_group(name = "max")]
            mod max {
                use super::*;

                #[divan::bench]
                fn s_1k(bencher: Bencher) {
                    let t = make_f32_1d(1024);
                    bencher.bench_synced(|| t.clone().max());
                }

                #[divan::bench]
                fn s_64k(bencher: Bencher) {
                    let t = make_f32_1d(64 * 1024);
                    bencher.bench_synced(|| t.clone().max());
                }

                #[divan::bench]
                fn s_1m(bencher: Bencher) {
                    let t = make_f32_1d(1024 * 1024);
                    bencher.bench_synced(|| t.clone().max());
                }
            }

            #[divan::bench_group(name = "min")]
            mod min {
                use super::*;

                #[divan::bench]
                fn s_1k(bencher: Bencher) {
                    let t = make_f32_1d(1024);
                    bencher.bench_synced(|| t.clone().min());
                }

                #[divan::bench]
                fn s_64k(bencher: Bencher) {
                    let t = make_f32_1d(64 * 1024);
                    bencher.bench_synced(|| t.clone().min());
                }

                #[divan::bench]
                fn s_1m(bencher: Bencher) {
                    let t = make_f32_1d(1024 * 1024);
                    bencher.bench_synced(|| t.clone().min());
                }
            }

            #[divan::bench_group(name = "int_max")]
            mod int_max {
                use super::*;

                #[divan::bench]
                fn s_256x256(bencher: Bencher) {
                    let Some(t) = make_int_2d(256, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| t.clone().max());
                }

                #[divan::bench]
                fn s_1024x1024(bencher: Bencher) {
                    let Some(t) = make_int_2d(1024, 1024) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| t.clone().max());
                }
            }

            #[divan::bench_group(name = "int_powi")]
            mod int_powi {
                use super::*;

                #[divan::bench]
                fn s_256x256(bencher: Bencher) {
                    let Some(base) = make_int_2d(256, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(exp) = make_int_exp_2d(256, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| base.clone().powi(exp.clone()));
                }

                #[divan::bench]
                fn s_1024x256(bencher: Bencher) {
                    let Some(base) = make_int_2d(1024, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let Some(exp) = make_int_exp_2d(1024, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| base.clone().powi(exp.clone()));
                }
            }

            #[divan::bench_group(name = "bool_select")]
            mod bool_select {
                use super::*;

                #[divan::bench]
                fn s_256x256_128idx(bencher: Bencher) {
                    let t = make_bool_2d(256, 256);
                    let Some(idx) = make_indices_1d(128, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| t.clone().select(0, idx.clone()));
                }

                #[divan::bench]
                fn s_1024x256_512idx(bencher: Bencher) {
                    let t = make_bool_2d(1024, 256);
                    let Some(idx) = make_indices_1d(512, 1024) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| t.clone().select(0, idx.clone()));
                }
            }

            #[divan::bench_group(name = "grid_sample_2d")]
            mod grid_sample {
                use super::*;

                #[divan::bench]
                fn s_b1_c3_32x32(bencher: Bencher) {
                    let img = make_image_4d(1, 3, 32, 32);
                    let grid = make_grid_4d(1, 32, 32);
                    bencher.bench_synced(|| {
                        img.clone()
                            .grid_sample_2d(grid.clone(), GridSampleOptions::default())
                    });
                }

                #[divan::bench]
                fn s_b1_c3_64x64(bencher: Bencher) {
                    let img = make_image_4d(1, 3, 64, 64);
                    let grid = make_grid_4d(1, 64, 64);
                    bencher.bench_synced(|| {
                        img.clone()
                            .grid_sample_2d(grid.clone(), GridSampleOptions::default())
                    });
                }

                #[divan::bench]
                fn s_b4_c3_32x32(bencher: Bencher) {
                    let img = make_image_4d(4, 3, 32, 32);
                    let grid = make_grid_4d(4, 32, 32);
                    bencher.bench_synced(|| {
                        img.clone()
                            .grid_sample_2d(grid.clone(), GridSampleOptions::default())
                    });
                }

                #[divan::bench]
                fn s_b1_c16_64x64(bencher: Bencher) {
                    let img = make_image_4d(1, 16, 64, 64);
                    let grid = make_grid_4d(1, 64, 64);
                    bencher.bench_synced(|| {
                        img.clone()
                            .grid_sample_2d(grid.clone(), GridSampleOptions::default())
                    });
                }
            }
        }
    };
}

bench_cat!(backend, "backend");
