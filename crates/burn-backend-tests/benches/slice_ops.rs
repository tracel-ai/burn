//! Benchmarks for slice operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench slice_ops --features simd
//! ```
//!
//! Memory allocation tracking is enabled via divan's AllocProfiler.

#[path = "common/mod.rs"]
mod common;
use common::{BencherExt, TestBackend};

use burn_tensor::{Tensor, TensorData, backend::Backend, s};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Slice Operations Benchmarks");
    println!("Memory allocation tracking enabled");
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
    let data: Vec<f32> = (0..d0 * d1 * d2)
        .map(|i| (i % 1000) as f32 / 1000.0)
        .collect();
    Tensor::from_data(TensorData::new(data, [d0, d1, d2]), &Default::default())
}

macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            // Basic slice (contiguous, step=1)
            #[divan::bench_group(name = "basic_slice")]
            mod basic_slice {
                use super::*;

                #[divan::bench]
                fn slice_1d_1k(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024);
                    bencher.bench_synced(|| t.clone().slice([256..768]));
                }

                #[divan::bench]
                fn slice_1d_1m(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024 * 1024);
                    bencher.bench_synced(|| t.clone().slice([256 * 1024..768 * 1024]));
                }

                #[divan::bench]
                fn slice_2d_256x256(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().slice([64..192, 64..192]));
                }

                #[divan::bench]
                fn slice_2d_1024x1024(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024);
                    bencher.bench_synced(|| t.clone().slice([256..768, 256..768]));
                }

                #[divan::bench]
                fn slice_3d_64x64x64(bencher: Bencher) {
                    let t = make_tensor_3d::<B>(64, 64, 64);
                    bencher.bench_synced(|| t.clone().slice([16..48, 16..48, 16..48]));
                }
            }

            // Slice with step > 1
            #[divan::bench_group(name = "slice_with_step")]
            mod slice_with_step {
                use super::*;

                #[divan::bench]
                fn step2_1d_1k(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024);
                    bencher.bench_synced(|| t.clone().slice(s![0..1024;2]));
                }

                #[divan::bench]
                fn step2_1d_1m(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024 * 1024);
                    bencher.bench_synced(|| t.clone().slice(s![0..1024 * 1024;2]));
                }

                #[divan::bench]
                fn step4_2d_256x256(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().slice(s![0..256;4, 0..256;4]));
                }

                #[divan::bench]
                fn step2_2d_1024x1024(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024);
                    bencher.bench_synced(|| t.clone().slice(s![0..1024;2, 0..1024;2]));
                }
            }

            // Slice on transposed tensor (non-contiguous)
            #[divan::bench_group(name = "slice_transposed")]
            mod slice_transposed {
                use super::*;

                #[divan::bench]
                fn transposed_256x256(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256).transpose();
                    bencher.bench_synced(|| t.clone().slice([64..192, 64..192]));
                }

                #[divan::bench]
                fn transposed_1024x1024(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024).transpose();
                    bencher.bench_synced(|| t.clone().slice([256..768, 256..768]));
                }
            }

            // Slice assign
            #[divan::bench_group(name = "slice_assign")]
            mod slice_assign {
                use super::*;

                #[divan::bench]
                fn assign_1d_1k(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024);
                    let v = make_tensor_1d::<B>(512);
                    bencher.bench_synced(|| t.clone().slice_assign([256..768], v.clone()));
                }

                #[divan::bench]
                fn assign_2d_256x256(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    let v = make_tensor_2d::<B>(128, 128);
                    bencher.bench_synced(|| t.clone().slice_assign([64..192, 64..192], v.clone()));
                }

                #[divan::bench]
                fn assign_2d_1024x1024(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024);
                    let v = make_tensor_2d::<B>(512, 512);
                    bencher.bench_synced(|| t.clone().slice_assign([256..768, 256..768], v.clone()));
                }
            }

            // Slice fill (scalar fill over a slice region)
            //
            // This is the pattern the user hit in issue #64 item 3:
            // `tensor.slice_fill(slice, scalar)`. burn-tensor's default impl
            // allocates a 1-element tensor, calls expand() to broadcast it
            // over the slice shape, and then hands that view to
            // slice_assign. A naive backend materializes the broadcast (one
            // full alloc + fill) before doing the actual strided write, so
            // the slice fill ends up doing ~3x the memory traffic it should.
            #[divan::bench_group(name = "slice_fill")]
            mod slice_fill {
                use super::*;

                #[divan::bench]
                fn fill_1d_512_of_1k(bencher: Bencher) {
                    let t = make_tensor_1d::<B>(1024);
                    bencher.bench_synced(|| t.clone().slice_fill([256..768], 1.0f32));
                }

                #[divan::bench]
                fn fill_2d_128x128_of_256x256(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().slice_fill([64..192, 64..192], 1.0f32));
                }

                #[divan::bench]
                fn fill_2d_512x512_of_1024x1024(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024);
                    bencher.bench_synced(|| t.clone().slice_fill([256..768, 256..768], 1.0f32));
                }

                // Segmentation-model-sized: 488x448 feature map, half the
                // spatial extent.
                #[divan::bench]
                fn fill_2d_244x224_of_488x448(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(488, 448);
                    bencher.bench_synced(|| t.clone().slice_fill([122..366, 112..336], 0.0f32));
                }

                // 3D feature map (B=1,C=64,H,W) half-spatial fill.
                #[divan::bench]
                fn fill_3d_64x64x64_of_64x128x128(bencher: Bencher) {
                    let t = make_tensor_3d::<B>(64, 128, 128);
                    bencher.bench_synced(|| t.clone().slice_fill([0..64, 32..96, 32..96], 0.0f32));
                }
            }

            // Narrow (single dimension slice)
            #[divan::bench_group(name = "narrow")]
            mod narrow {
                use super::*;

                #[divan::bench]
                fn narrow_dim0_256x256(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().narrow(0, 64, 128));
                }

                #[divan::bench]
                fn narrow_dim1_256x256(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(256, 256);
                    bencher.bench_synced(|| t.clone().narrow(1, 64, 128));
                }

                #[divan::bench]
                fn narrow_dim0_1024x1024(bencher: Bencher) {
                    let t = make_tensor_2d::<B>(1024, 1024);
                    bencher.bench_synced(|| t.clone().narrow(0, 256, 512));
                }
            }
        }
    };
}

bench_backend!(TestBackend, backend, "backend");
