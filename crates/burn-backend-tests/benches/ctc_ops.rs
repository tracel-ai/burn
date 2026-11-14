//! Flex CPU CTC forward and training versus the default autodiff decomposition.
//!
//! From this crate, run `RAYON_NUM_THREADS=1 cargo bench-flex --bench ctc_ops`.
//! Add `--features flex-simd,flex-rayon` to enable Flex SIMD and parallel execution.
//! Add `bench-disable-alloc` to those features for timing without allocation profiling.
//! Training includes log-softmax, summed CTC loss, backward, and gradient access;
//! forward measures only CTC on retained log-probabilities. Input creation is excluded.
//! Divan tracks allocations on the benchmark thread. Leave `flex-rayon` disabled
//! and pass `--sample-size 1` for peak allocation measurements; enable Rayon for
//! parallel timing runs.

#[cfg(not(feature = "bench-disable-alloc"))]
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

#[cfg(feature = "flex")]
#[divan::bench_group(name = "flex")]
mod flex {
    use burn_autodiff::Autodiff;
    use burn_backend::{
        AutodiffBackend,
        ops::{FloatTensorOps, ModuleOps, ctc::ctc_loss_default},
    };
    use burn_dispatch::backends::flex::{Flex, FlexTensor};
    use burn_tensor::FloatDType;
    use divan::Bencher;

    mod scenarios {
        include!("common/ctc.rs");
    }
    use scenarios::{BATCH, MIXED_LENGTHS, SHORT, STRIDED, Scenario};

    type Ad = Autodiff<Flex>;

    fn bench_forward(
        bencher: Bencher,
        scenario: &Scenario,
        float_dtype: FloatDType,
        decomposed: bool,
    ) {
        let [logits, targets, input_lengths, target_lengths] = scenario.fixture(float_dtype);
        let scores = scenario.log_probs::<Flex>(logits);
        let forward = if decomposed {
            ctc_loss_default::<Flex>
        } else {
            <Flex as ModuleOps<Flex>>::ctc_loss
        };
        bencher.bench(|| {
            let output = forward(
                scores.clone(),
                targets.clone(),
                input_lengths.clone(),
                target_lengths.clone(),
                0,
            );
            // Flex executes eagerly; include output access and disposal on both paths.
            std::hint::black_box(output.bytes());
        });
    }

    fn bench_training(
        bencher: Bencher,
        scenario: &Scenario,
        float_dtype: FloatDType,
        decomposed: bool,
    ) {
        let [logits, targets, input_lengths, target_lengths] = scenario.fixture(float_dtype);
        let forward = if decomposed {
            ctc_loss_default::<Ad>
        } else {
            <Ad as ModuleOps<Ad>>::ctc_loss
        };
        bencher.bench(|| {
            // Each iteration builds a fresh graph; backward consumes its steps.
            let logits = Ad::float_set_require_grad(Ad::from_inner(logits.clone()), true);
            let loss = forward(
                scenario.log_probs::<Ad>(logits.clone()),
                targets.clone(),
                input_lengths.clone(),
                target_lengths.clone(),
                0,
            );
            let grads = Ad::backward(Ad::float_sum(loss));
            let gradient =
                Ad::grad(&logits, &grads).expect("CTC must propagate gradients to logits");
            std::hint::black_box(gradient.bytes());
        });
    }

    macro_rules! scenario {
        ($name:ident, $fixture:ident, $float:ident) => {
            mod $name {
                use super::*;

                #[divan::bench]
                fn forward_native(bencher: Bencher) {
                    bench_forward(bencher, &$fixture, FloatDType::$float, false);
                }

                #[divan::bench]
                fn forward_default(bencher: Bencher) {
                    bench_forward(bencher, &$fixture, FloatDType::$float, true);
                }

                #[divan::bench]
                fn training_native(bencher: Bencher) {
                    bench_training(bencher, &$fixture, FloatDType::$float, false);
                }

                #[divan::bench]
                fn training_default(bencher: Bencher) {
                    bench_training(bencher, &$fixture, FloatDType::$float, true);
                }
            }
        };
    }

    scenario!(short_f32, SHORT, F32);
    scenario!(batch_f32, BATCH, F32);
    scenario!(batch_f64, BATCH, F64);
    scenario!(batch_f16, BATCH, F16);
    scenario!(batch_bf16, BATCH, BF16);
    scenario!(mixed_lengths_f32, MIXED_LENGTHS, F32);
    scenario!(strided_f32, STRIDED, F32);
}
