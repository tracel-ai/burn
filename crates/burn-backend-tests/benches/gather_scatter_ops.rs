//! Benchmarks for gather/scatter operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench gather_scatter_ops --features simd
//! ```

#[path = "common/mod.rs"]
mod common;
use common::{BencherExt, TestBackend};

use burn_tensor::{IndexingUpdateOp, Int, Tensor, TensorData, backend::Backend};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Gather/scatter ops Benchmarks");
    println!();
    divan::main();
    common::report_failures();
}

fn make_tensor_2d<B: Backend>(rows: usize, cols: usize) -> Tensor<B, 2> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
}

fn make_indices_2d<B: Backend>(
    rows: usize,
    cols: usize,
    max_idx: usize,
) -> Option<Tensor<B, 2, Int>> {
    common::try_setup(|| {
        let data: Vec<i32> = (0..rows * cols).map(|i| (i % max_idx) as i32).collect();
        Tensor::from_data(TensorData::new(data, [rows, cols]), &Default::default())
    })
}

fn make_indices_1d<B: Backend>(size: usize, max_idx: usize) -> Option<Tensor<B, 1, Int>> {
    common::try_setup(|| {
        let data: Vec<i32> = (0..size).map(|i| (i % max_idx) as i32).collect();
        Tensor::from_data(TensorData::new(data, [size]), &Default::default())
    })
}

macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            #[divan::bench_group(name = "gather")]
            mod gather {
                use super::*;

                #[divan::bench]
                fn _256x256_dim1(bencher: Bencher) {
                    let tensor = make_tensor_2d::<B>(256, 256);
                    let Some(indices) = make_indices_2d::<B>(256, 128, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| tensor.clone().gather(1, indices.clone()));
                }

                #[divan::bench]
                fn _1024x1024_dim1(bencher: Bencher) {
                    let tensor = make_tensor_2d::<B>(1024, 1024);
                    let Some(indices) = make_indices_2d::<B>(1024, 512, 1024) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| tensor.clone().gather(1, indices.clone()));
                }

                #[divan::bench]
                fn _256x256_dim0(bencher: Bencher) {
                    let tensor = make_tensor_2d::<B>(256, 256);
                    let Some(indices) = make_indices_2d::<B>(128, 256, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| tensor.clone().gather(0, indices.clone()));
                }
            }

            #[divan::bench_group(name = "scatter_add")]
            mod scatter_add {
                use super::*;

                #[divan::bench]
                fn _256x256_dim1(bencher: Bencher) {
                    let tensor = make_tensor_2d::<B>(256, 256);
                    let Some(indices) = make_indices_2d::<B>(256, 128, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let values = make_tensor_2d::<B>(256, 128);
                    bencher.bench_synced(|| {
                        tensor.clone().scatter(
                            1,
                            indices.clone(),
                            values.clone(),
                            IndexingUpdateOp::Add,
                        )
                    });
                }

                #[divan::bench]
                fn _1024x1024_dim1(bencher: Bencher) {
                    let tensor = make_tensor_2d::<B>(1024, 1024);
                    let Some(indices) = make_indices_2d::<B>(1024, 512, 1024) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let values = make_tensor_2d::<B>(1024, 512);
                    bencher.bench_synced(|| {
                        tensor.clone().scatter(
                            1,
                            indices.clone(),
                            values.clone(),
                            IndexingUpdateOp::Add,
                        )
                    });
                }
            }

            #[divan::bench_group(name = "select")]
            mod select {
                use super::*;

                #[divan::bench]
                fn _256x256_dim0(bencher: Bencher) {
                    let tensor = make_tensor_2d::<B>(256, 256);
                    let Some(indices) = make_indices_1d::<B>(128, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| tensor.clone().select(0, indices.clone()));
                }

                #[divan::bench]
                fn _1024x1024_dim0(bencher: Bencher) {
                    let tensor = make_tensor_2d::<B>(1024, 1024);
                    let Some(indices) = make_indices_1d::<B>(512, 1024) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| tensor.clone().select(0, indices.clone()));
                }

                #[divan::bench]
                fn _256x256_dim1(bencher: Bencher) {
                    let tensor = make_tensor_2d::<B>(256, 256);
                    let Some(indices) = make_indices_1d::<B>(128, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    bencher.bench_synced(|| tensor.clone().select(1, indices.clone()));
                }
            }

            #[divan::bench_group(name = "select_add")]
            mod select_add {
                use super::*;

                #[divan::bench]
                fn _256x256_dim0(bencher: Bencher) {
                    let tensor = make_tensor_2d::<B>(256, 256);
                    let Some(indices) = make_indices_1d::<B>(128, 256) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let values = make_tensor_2d::<B>(128, 256);
                    bencher.bench_synced(|| {
                        tensor.clone().select_assign(
                            0,
                            indices.clone(),
                            values.clone(),
                            IndexingUpdateOp::Add,
                        )
                    });
                }

                #[divan::bench]
                fn _1024x1024_dim0(bencher: Bencher) {
                    let tensor = make_tensor_2d::<B>(1024, 1024);
                    let Some(indices) = make_indices_1d::<B>(512, 1024) else {
                        bencher.bench(|| ());
                        return;
                    };
                    let values = make_tensor_2d::<B>(512, 1024);
                    bencher.bench_synced(|| {
                        tensor.clone().select_assign(
                            0,
                            indices.clone(),
                            values.clone(),
                            IndexingUpdateOp::Add,
                        )
                    });
                }
            }
        }
    };
}

bench_backend!(TestBackend, backend, "backend");
