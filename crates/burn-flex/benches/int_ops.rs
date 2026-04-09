//! Benchmarks for integer operations: cast and random.
//!
//! Run with:
//! ```bash
//! cargo bench --bench int_ops --features simd,rayon
//! ```

use burn_flex::Flex;
use burn_ndarray::NdArray;
use burn_tensor::{DType, Distribution, Int, Tensor, TensorData, backend::Backend};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Integer Operations Benchmarks: Flex vs NdArray");
    println!("Memory allocation tracking enabled");
    println!();
    divan::main();
}

fn make_int_tensor<B: Backend>(shape: &[usize]) -> Tensor<B, 2, Int> {
    let size: usize = shape.iter().product();
    // Use values that fit in i8 (-128 to 127) so all casts work
    let data: Vec<i64> = (0..size).map(|i| (i % 200) as i64 - 100).collect();
    Tensor::from_data(TensorData::new(data, shape.to_vec()), &Default::default())
}

// =============================================================================
// Int Cast Benchmarks
// =============================================================================

macro_rules! bench_cast_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            #[divan::bench_group(name = "cast")]
            mod cast {
                use super::*;

                // Small tensor cast
                #[divan::bench]
                fn cast_i64_to_i32_64x64(bencher: Bencher) {
                    let t = make_int_tensor::<B>(&[64, 64]);
                    bencher.bench(|| t.clone().cast(DType::I32));
                }

                // Medium tensor cast
                #[divan::bench]
                fn cast_i64_to_i32_256x256(bencher: Bencher) {
                    let t = make_int_tensor::<B>(&[256, 256]);
                    bencher.bench(|| t.clone().cast(DType::I32));
                }

                // Large tensor cast
                #[divan::bench]
                fn cast_i64_to_i32_1024x1024(bencher: Bencher) {
                    let t = make_int_tensor::<B>(&[1024, 1024]);
                    bencher.bench(|| t.clone().cast(DType::I32));
                }

                // Cast to smaller type (i8)
                #[divan::bench]
                fn cast_i64_to_i8_256x256(bencher: Bencher) {
                    let t = make_int_tensor::<B>(&[256, 256]);
                    bencher.bench(|| t.clone().cast(DType::I8));
                }
            }
        }
    };
}

bench_cast_backend!(Flex, flex_cast, "Flex_Cast");
bench_cast_backend!(NdArray, ndarray_cast, "NdArray_Cast");

// =============================================================================
// Int Random Benchmarks
// =============================================================================

macro_rules! bench_random_backend {
    ($backend:ty, $mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            type B = $backend;

            #[divan::bench_group(name = "random")]
            mod random {
                use super::*;

                // Small random tensor
                #[divan::bench]
                fn random_uniform_64x64(bencher: Bencher) {
                    bencher.bench(|| {
                        Tensor::<B, 2, Int>::random(
                            [64, 64],
                            Distribution::Uniform(0.0, 100.0),
                            &Default::default(),
                        )
                    });
                }

                // Medium random tensor
                #[divan::bench]
                fn random_uniform_256x256(bencher: Bencher) {
                    bencher.bench(|| {
                        Tensor::<B, 2, Int>::random(
                            [256, 256],
                            Distribution::Uniform(0.0, 100.0),
                            &Default::default(),
                        )
                    });
                }

                // Large random tensor
                #[divan::bench]
                fn random_uniform_1024x1024(bencher: Bencher) {
                    bencher.bench(|| {
                        Tensor::<B, 2, Int>::random(
                            [1024, 1024],
                            Distribution::Uniform(0.0, 100.0),
                            &Default::default(),
                        )
                    });
                }

                // 3D random tensor (batch)
                #[divan::bench]
                fn random_uniform_batch_16x128x128(bencher: Bencher) {
                    bencher.bench(|| {
                        Tensor::<B, 3, Int>::random(
                            [16, 128, 128],
                            Distribution::Uniform(-1000.0, 1000.0),
                            &Default::default(),
                        )
                    });
                }
            }
        }
    };
}

bench_random_backend!(Flex, flex_random, "Flex_Random");
bench_random_backend!(NdArray, ndarray_random, "NdArray_Random");
