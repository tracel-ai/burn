//! Softmax, log-softmax, and hard-target cross-entropy benchmarks.
//! Covers forward and backward execution, confidence, and class-axis layouts.
//!
//! Run from `crates/burn-backend-tests`:
//! ```bash
//! cargo bench-flex --bench softmax_ops --features flex-simd,flex-rayon
//! ```

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{Device, Element, Int, Tensor, TensorData, activation};
use divan::Bencher;
use std::fmt;

#[cfg(not(feature = "bench-disable-alloc"))]
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    println!("Softmax, log-softmax, and hard-target cross-entropy benchmarks");
    println!();
    divan::main();
    common::report_failures();
}

fn make_tensor_3d<E: Element + From<f32>>(d0: usize, d1: usize, d2: usize) -> Tensor<3> {
    let data: Vec<E> = (0..d0 * d1 * d2)
        .map(|i| E::from((((i % 997) as f32) / 997.0) - 0.5))
        .collect();
    Tensor::from_data(TensorData::new(data, [d0, d1, d2]), &Default::default())
}

/// Legacy max-shift decomposition retained as a performance reference.
fn decomposed_softmax<const D: usize>(x: Tensor<D>, dim: usize) -> Tensor<D> {
    let max = x.clone().detach().max_dim(dim);
    let shifted = x - max;
    let exp = shifted.exp();
    let sum = exp.clone().sum_dim(dim);
    exp / sum
}

macro_rules! bench_backend {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            #[divan::bench_group(name = "last_axis_f32")]
            mod last_axis_f32 {
                use super::*;

                #[divan::bench]
                fn trait_hook_bert_seq512_h12(bencher: Bencher) {
                    let x = make_tensor_3d::<f32>(12, 512, 512);
                    bencher.bench_synced(|| burn_tensor::activation::softmax(x.clone(), 2));
                }

                #[divan::bench]
                fn decomposed_bert_seq512_h12(bencher: Bencher) {
                    let x = make_tensor_3d::<f32>(12, 512, 512);
                    bencher.bench_synced(|| decomposed_softmax(x.clone(), 2));
                }

                #[divan::bench]
                fn trait_hook_wide_d2048(bencher: Bencher) {
                    let x = make_tensor_3d::<f32>(2, 256, 2048);
                    bencher.bench_synced(|| burn_tensor::activation::softmax(x.clone(), 2));
                }

                #[divan::bench]
                fn decomposed_wide_d2048(bencher: Bencher) {
                    let x = make_tensor_3d::<f32>(2, 256, 2048);
                    bencher.bench_synced(|| decomposed_softmax(x.clone(), 2));
                }
            }

            #[divan::bench_group(name = "non_last_axis_f32")]
            mod non_last_axis_f32 {
                use super::*;

                #[divan::bench]
                fn trait_hook_dim1_of_3(bencher: Bencher) {
                    let x = make_tensor_3d::<f32>(4, 1024, 512);
                    bencher.bench_synced(|| burn_tensor::activation::softmax(x.clone(), 1));
                }

                #[divan::bench]
                fn decomposed_dim1_of_3(bencher: Bencher) {
                    let x = make_tensor_3d::<f32>(4, 1024, 512);
                    bencher.bench_synced(|| decomposed_softmax(x.clone(), 1));
                }
            }
        }
    };
}

bench_backend!(backend, "backend");

#[derive(Clone, Copy, Debug)]
enum InputCase {
    Short,
    ShortHighConfidence,
    ShortTransposed,
    ShortHighConfidenceTransposed,
    Wide,
    WideHighConfidence,
    WideTransposed,
    WideHighConfidenceTransposed,
}

impl fmt::Display for InputCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

const INPUT_CASES: [InputCase; 8] = [
    InputCase::Short,
    InputCase::ShortHighConfidence,
    InputCase::ShortTransposed,
    InputCase::ShortHighConfidenceTransposed,
    InputCase::Wide,
    InputCase::WideHighConfidence,
    InputCase::WideTransposed,
    InputCase::WideHighConfidenceTransposed,
];

fn collapse_inputs(case: InputCase, tracked: bool) -> (Tensor<2>, Tensor<2, Int>, usize) {
    let (classes, high_confidence, transposed) = match case {
        InputCase::Short => (3, false, false),
        InputCase::ShortHighConfidence => (3, true, false),
        InputCase::ShortTransposed => (3, false, true),
        InputCase::ShortHighConfidenceTransposed => (3, true, true),
        InputCase::Wide => (2048, false, false),
        InputCase::WideHighConfidence => (2048, true, false),
        InputCase::WideTransposed => (2048, false, true),
        InputCase::WideHighConfidenceTransposed => (2048, true, true),
    };
    let device = if tracked {
        Device::default().autodiff()
    } else {
        Device::default()
    };
    let values: Vec<f32> = (0..32 * classes)
        .map(|index| {
            if high_confidence {
                if index % classes == 0 { 20.0 } else { -20.0 }
            } else {
                (index % 97) as f32 / 48.0 - 1.0
            }
        })
        .collect();
    let input = Tensor::<2>::from_data(TensorData::new(values, [32, classes]), &device);
    let indices = Tensor::<2, Int>::zeros([32, 1], &device);
    let (input, indices, dim) = if transposed {
        (input.transpose(), indices.transpose(), 0)
    } else {
        (input, indices, 1)
    };
    let input = if tracked { input.require_grad() } else { input };
    (input, indices, dim)
}

/// Default hard-target loss formula. The `burn-nn` benchmark measures the public loss module.
fn hard_target_cross_entropy(input: Tensor<2>, indices: Tensor<2, Int>, dim: usize) -> Tensor<1> {
    activation::log_softmax(input, dim)
        .gather(dim, indices)
        .mean()
        .neg()
}

#[divan::bench_group]
mod collapse {
    use super::*;

    #[divan::bench(args = INPUT_CASES)]
    fn softmax_forward(bencher: Bencher, case: InputCase) {
        let (input, _, dim) = collapse_inputs(case, false);
        bencher.bench_synced(|| activation::softmax(input.clone(), dim));
    }

    #[divan::bench(args = INPUT_CASES)]
    fn softmax_forward_backward(bencher: Bencher, case: InputCase) {
        let (input, indices, dim) = collapse_inputs(case, true);
        bencher.bench_synced(|| {
            let output = activation::softmax(input.clone(), dim).gather(dim, indices.clone());
            let grads = output.sum().backward();
            input.grad(&grads).unwrap()
        });
    }

    #[divan::bench(args = INPUT_CASES)]
    fn log_softmax_forward(bencher: Bencher, case: InputCase) {
        let (input, _, dim) = collapse_inputs(case, false);
        bencher.bench_synced(|| activation::log_softmax(input.clone(), dim));
    }

    #[divan::bench(args = INPUT_CASES)]
    fn log_softmax_forward_backward(bencher: Bencher, case: InputCase) {
        let (input, indices, dim) = collapse_inputs(case, true);
        bencher.bench_synced(|| {
            let output = activation::log_softmax(input.clone(), dim).gather(dim, indices.clone());
            let grads = output.sum().backward();
            input.grad(&grads).unwrap()
        });
    }

    #[divan::bench(args = INPUT_CASES)]
    fn cross_entropy_forward(bencher: Bencher, case: InputCase) {
        let (input, indices, dim) = collapse_inputs(case, false);
        bencher.bench_synced(|| hard_target_cross_entropy(input.clone(), indices.clone(), dim));
    }

    #[divan::bench(args = INPUT_CASES)]
    fn cross_entropy_forward_backward(bencher: Bencher, case: InputCase) {
        let (input, indices, dim) = collapse_inputs(case, true);
        bencher.bench_synced(|| {
            let output = hard_target_cross_entropy(input.clone(), indices.clone(), dim);
            let grads = output.backward();
            input.grad(&grads).unwrap()
        });
    }
}
