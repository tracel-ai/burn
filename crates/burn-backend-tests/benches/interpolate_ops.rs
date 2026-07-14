//! Benchmarks for interpolation operations.
//!
//! Run with:
//! ```bash
//! cargo bench --bench interpolate_ops --features simd,rayon
//! ```
//!
//! Memory allocation tracking is enabled via divan's AllocProfiler.

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{
    Tensor, TensorData, module,
    ops::{InterpolateMode, InterpolateOptions},
};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Interpolate Benchmarks");
    println!("Memory allocation tracking enabled");
    println!();
    divan::main();
    common::report_failures();
}

fn make_input(batch: usize, channels: usize, height: usize, width: usize) -> Tensor<4> {
    let data: Vec<f32> = (0..batch * channels * height * width)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(
        TensorData::new(data, [batch, channels, height, width]),
        &Default::default(),
    )
}

macro_rules! bench_backend {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            #[divan::bench_group(name = "nearest")]
            mod nearest {
                use super::*;

                #[divan::bench]
                fn upsample_2x_64x64_to_128x128(bencher: Bencher) {
                    let x = make_input(1, 3, 64, 64);
                    let opts = InterpolateOptions::new(InterpolateMode::Nearest);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [128, 128], opts.clone()));
                }

                #[divan::bench]
                fn upsample_4x_32x32_to_128x128(bencher: Bencher) {
                    let x = make_input(1, 3, 32, 32);
                    let opts = InterpolateOptions::new(InterpolateMode::Nearest);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [128, 128], opts.clone()));
                }

                #[divan::bench]
                fn downsample_2x_256x256_to_128x128(bencher: Bencher) {
                    let x = make_input(1, 3, 256, 256);
                    let opts = InterpolateOptions::new(InterpolateMode::Nearest);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [128, 128], opts.clone()));
                }

                #[divan::bench]
                fn batch8_upsample_2x_64x64_to_128x128(bencher: Bencher) {
                    let x = make_input(8, 3, 64, 64);
                    let opts = InterpolateOptions::new(InterpolateMode::Nearest);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [128, 128], opts.clone()));
                }

                #[divan::bench]
                fn channels64_upsample_2x_32x32_to_64x64(bencher: Bencher) {
                    let x = make_input(1, 64, 32, 32);
                    let opts = InterpolateOptions::new(InterpolateMode::Nearest);
                    bencher.bench_synced(|| module::interpolate(x.clone(), [64, 64], opts.clone()));
                }
            }

            #[divan::bench_group(name = "bilinear")]
            mod bilinear {
                use super::*;

                #[divan::bench]
                fn upsample_2x_64x64_to_128x128(bencher: Bencher) {
                    let x = make_input(1, 3, 64, 64);
                    let opts = InterpolateOptions::new(InterpolateMode::Bilinear);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [128, 128], opts.clone()));
                }

                #[divan::bench]
                fn upsample_4x_32x32_to_128x128(bencher: Bencher) {
                    let x = make_input(1, 3, 32, 32);
                    let opts = InterpolateOptions::new(InterpolateMode::Bilinear);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [128, 128], opts.clone()));
                }

                #[divan::bench]
                fn downsample_2x_256x256_to_128x128(bencher: Bencher) {
                    let x = make_input(1, 3, 256, 256);
                    let opts = InterpolateOptions::new(InterpolateMode::Bilinear);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [128, 128], opts.clone()));
                }

                #[divan::bench]
                fn batch8_upsample_2x_64x64_to_128x128(bencher: Bencher) {
                    let x = make_input(8, 3, 64, 64);
                    let opts = InterpolateOptions::new(InterpolateMode::Bilinear);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [128, 128], opts.clone()));
                }

                #[divan::bench]
                fn channels64_upsample_2x_32x32_to_64x64(bencher: Bencher) {
                    let x = make_input(1, 64, 32, 32);
                    let opts = InterpolateOptions::new(InterpolateMode::Bilinear);
                    bencher.bench_synced(|| module::interpolate(x.clone(), [64, 64], opts.clone()));
                }

                // Segmentation-model sized shapes mirroring the user's
                // 488x448 input + UNet up/down pattern from issue #64
                // item 4. Covers the typical spatial dims across
                // ConvNeXt-style downsample stages.
                #[divan::bench]
                fn model_downsample_488x448_to_244x224(bencher: Bencher) {
                    let x = make_input(1, 3, 488, 448);
                    let opts = InterpolateOptions::new(InterpolateMode::Bilinear);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [244, 224], opts.clone()));
                }

                #[divan::bench]
                fn model_upsample_244x224_to_488x448(bencher: Bencher) {
                    let x = make_input(1, 48, 244, 224);
                    let opts = InterpolateOptions::new(InterpolateMode::Bilinear);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [488, 448], opts.clone()));
                }

                #[divan::bench]
                fn model_upsample_61x56_to_122x112(bencher: Bencher) {
                    let x = make_input(1, 192, 61, 56);
                    let opts = InterpolateOptions::new(InterpolateMode::Bilinear);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [122, 112], opts.clone()));
                }

                // Explicit align_corners=false variant at a model size,
                // since the default is `true` everywhere else.
                #[divan::bench]
                fn model_upsample_61x56_to_122x112_halfpixel(bencher: Bencher) {
                    let x = make_input(1, 192, 61, 56);
                    let opts = InterpolateOptions::new(InterpolateMode::Bilinear)
                        .with_align_corners(false);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [122, 112], opts.clone()));
                }
            }

            #[divan::bench_group(name = "bicubic")]
            mod bicubic {
                use super::*;

                #[divan::bench]
                fn upsample_2x_64x64_to_128x128(bencher: Bencher) {
                    let x = make_input(1, 3, 64, 64);
                    let opts = InterpolateOptions::new(InterpolateMode::Bicubic);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [128, 128], opts.clone()));
                }

                #[divan::bench]
                fn upsample_4x_32x32_to_128x128(bencher: Bencher) {
                    let x = make_input(1, 3, 32, 32);
                    let opts = InterpolateOptions::new(InterpolateMode::Bicubic);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [128, 128], opts.clone()));
                }

                #[divan::bench]
                fn downsample_2x_256x256_to_128x128(bencher: Bencher) {
                    let x = make_input(1, 3, 256, 256);
                    let opts = InterpolateOptions::new(InterpolateMode::Bicubic);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [128, 128], opts.clone()));
                }

                #[divan::bench]
                fn batch8_upsample_2x_64x64_to_128x128(bencher: Bencher) {
                    let x = make_input(8, 3, 64, 64);
                    let opts = InterpolateOptions::new(InterpolateMode::Bicubic);
                    bencher
                        .bench_synced(|| module::interpolate(x.clone(), [128, 128], opts.clone()));
                }

                #[divan::bench]
                fn channels64_upsample_2x_32x32_to_64x64(bencher: Bencher) {
                    let x = make_input(1, 64, 32, 32);
                    let opts = InterpolateOptions::new(InterpolateMode::Bicubic);
                    bencher.bench_synced(|| module::interpolate(x.clone(), [64, 64], opts.clone()));
                }
            }
        }
    };
}

bench_backend!(backend, "backend");
