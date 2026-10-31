//! Flex CPU CTC forward: backend hook versus the default decomposition.
//!
//! From this crate, run `RAYON_NUM_THREADS=1 cargo bench-flex --bench ctc_ops`.
//! Add `--features flex-simd,flex-rayon` to enable Flex SIMD and parallel execution.
//! Add `bench-disable-alloc` to those features for timing without allocation profiling.

#[cfg(not(feature = "bench-disable-alloc"))]
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    divan::main();
}

#[cfg(feature = "flex")]
#[divan::bench_group(name = "flex")]
mod flex {
    use burn_backend::ops::{ModuleOps, ctc::ctc_loss_default};
    use burn_dispatch::backends::flex::{Flex, FlexTensor};
    use burn_tensor::FloatDType;
    use divan::Bencher;

    mod scenarios {
        include!("common/ctc.rs");
    }
    use scenarios::{SCENARIOS, Scenario};

    fn bench_ctc(bencher: Bencher, scenario: &Scenario, float_dtype: FloatDType, decomposed: bool) {
        std::hint::black_box((scenario.id, scenario.name));
        let ([scores, targets, input_lengths, target_lengths], blank) =
            scenario.fixture(float_dtype);
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
                blank,
            );
            // Flex executes eagerly; both paths include output access and disposal.
            std::hint::black_box(output.bytes());
        });
    }

    macro_rules! scenario {
        ($name:ident, $index:expr, $float:ident) => {
            mod $name {
                use super::*;

                #[divan::bench]
                fn backend_hook(bencher: Bencher) {
                    bench_ctc(bencher, &SCENARIOS[$index], FloatDType::$float, false);
                }

                #[divan::bench]
                fn default(bencher: Bencher) {
                    bench_ctc(bencher, &SCENARIOS[$index], FloatDType::$float, true);
                }
            }
        };
    }

    macro_rules! dtype_group {
        ($name:ident, $float:ident) => {
            #[divan::bench_group]
            mod $name {
                use super::*;

                scenario!(s01_tiny, 0, $float);
                scenario!(s02_short, 1, $float);
                scenario!(s03_empty, 2, $float);
                scenario!(s04_medium, 3, $float);
                scenario!(s05_representative, 4, $float);
                scenario!(s06_long, 5, $float);
                scenario!(s07_long_targets, 6, $float);
                scenario!(s08_long_short_targets, 7, $float);
                scenario!(s09_single, 8, $float);
                scenario!(s10_wide_vocabulary, 9, $float);
                scenario!(s11_mixed_lengths, 10, $float);
                scenario!(s12_repeated_labels, 11, $float);
                scenario!(s13_strided_scores, 12, $float);
                scenario!(s14_unreachable, 13, $float);
                scenario!(s15_padded_targets, 14, $float);
                scenario!(s16_nan_fallback, 15, $float);
            }
        };
    }

    dtype_group!(f32, F32);
    dtype_group!(f64, F64);
    dtype_group!(f16, F16);
    dtype_group!(bf16, BF16);
}
