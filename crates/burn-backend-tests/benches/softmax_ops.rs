//! Benchmarks comparing `B::softmax` against a manual decomposition baseline.
//! Covers last-axis and non-last-axis cases.
//!
//! Run with:
//! ```bash
//! cargo bench --bench softmax_ops --features simd,rayon
//! ```

#[path = "common/mod.rs"]
mod common;
use common::{BencherExt, TestBackend};

use burn_tensor::{Element, Tensor, TensorData, backend::Backend};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Softmax Benchmarks: fused (via B::softmax) vs decomposed");
    println!();
    divan::main();
    common::report_failures();
}

fn make_tensor_3d<B: Backend, E: Element + From<f32>>(
    d0: usize,
    d1: usize,
    d2: usize,
) -> Tensor<B, 3> {
    let data: Vec<E> = (0..d0 * d1 * d2)
        .map(|i| E::from((((i % 997) as f32) / 997.0) - 0.5))
        .collect();
    Tensor::from_data(TensorData::new(data, [d0, d1, d2]), &Default::default())
}

/// `softmax(x, dim) = exp(x - max) / sum(exp(x - max))`, matching the
/// `ActivationOps::softmax` default in burn-backend.
fn decomposed_softmax<B: Backend, const D: usize>(x: Tensor<B, D>, dim: usize) -> Tensor<B, D> {
    let max = x.clone().detach().max_dim(dim);
    let shifted = x - max;
    let exp = shifted.exp();
    let sum = exp.clone().sum_dim(dim);
    exp / sum
}

macro_rules! bench_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            #[divan::bench_group(name = "last_axis_f32")]
            mod last_axis_f32 {
                use super::*;

                #[divan::bench]
                fn trait_hook_bert_seq512_h12(bencher: Bencher) {
                    let x = make_tensor_3d::<B, f32>(12, 512, 512);
                    bencher.bench_synced(|| burn_tensor::activation::softmax(x.clone(), 2));
                }

                #[divan::bench]
                fn decomposed_bert_seq512_h12(bencher: Bencher) {
                    let x = make_tensor_3d::<B, f32>(12, 512, 512);
                    bencher.bench_synced(|| decomposed_softmax(x.clone(), 2));
                }

                #[divan::bench]
                fn trait_hook_wide_d2048(bencher: Bencher) {
                    let x = make_tensor_3d::<B, f32>(2, 256, 2048);
                    bencher.bench_synced(|| burn_tensor::activation::softmax(x.clone(), 2));
                }

                #[divan::bench]
                fn decomposed_wide_d2048(bencher: Bencher) {
                    let x = make_tensor_3d::<B, f32>(2, 256, 2048);
                    bencher.bench_synced(|| decomposed_softmax(x.clone(), 2));
                }
            }

            #[divan::bench_group(name = "non_last_axis_f32")]
            mod non_last_axis_f32 {
                use super::*;

                #[divan::bench]
                fn trait_hook_dim1_of_3(bencher: Bencher) {
                    let x = make_tensor_3d::<B, f32>(4, 1024, 512);
                    bencher.bench_synced(|| burn_tensor::activation::softmax(x.clone(), 1));
                }

                #[divan::bench]
                fn decomposed_dim1_of_3(bencher: Bencher) {
                    let x = make_tensor_3d::<B, f32>(4, 1024, 512);
                    bencher.bench_synced(|| decomposed_softmax(x.clone(), 1));
                }
            }
        }
    };
}

bench_backend!(TestBackend, backend, "backend");
