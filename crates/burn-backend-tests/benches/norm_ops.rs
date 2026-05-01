//! Benchmarks for normalization ops.
//!
//! Covers `burn::nn::LayerNorm::forward`'s decomposed primitive-ops path
//! (what you get by default from burn-nn), the backend `layer_norm` fast
//! path, and a `reshape -> layer_norm -> reshape-back` composite that
//! mirrors a ConvNeXt-style channels-last layer norm. See issue #64 item 2.
//!
//! Run with:
//! ```bash
//! cargo bench --bench norm_ops --features simd,rayon
//! ```

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::{Tensor, TensorData};
use divan::{AllocProfiler, Bencher};

/// Route through the `B::layer_norm` backend hook. On Flex this hits the
/// fused override; on NdArray this hits the `ModuleOps` default decomposition.
fn trait_layer_norm<const D: usize>(
    input: Tensor<D>,
    gamma: Tensor<1>,
    beta: Tensor<1>,
    epsilon: f64,
) -> Tensor<D> {
    burn_tensor::module::layer_norm(input, gamma, Some(beta), epsilon)
}

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Normalization Benchmarks");
    println!();
    divan::main();
    common::report_failures();
}

fn make_tensor_3d(d0: usize, d1: usize, d2: usize) -> Tensor<3> {
    let data: Vec<f32> = (0..d0 * d1 * d2)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(TensorData::new(data, [d0, d1, d2]), &Default::default())
}

fn make_tensor_4d(d0: usize, d1: usize, d2: usize, d3: usize) -> Tensor<4> {
    let data: Vec<f32> = (0..d0 * d1 * d2 * d3)
        .map(|i| ((i % 1000) as f32 / 1000.0) - 0.5)
        .collect();
    Tensor::from_data(TensorData::new(data, [d0, d1, d2, d3]), &Default::default())
}

fn make_gamma_beta(d_model: usize) -> (Tensor<1>, Tensor<1>) {
    let gamma: Vec<f32> = (0..d_model)
        .map(|i| 0.5 + (i as f32 / d_model as f32))
        .collect();
    let beta: Vec<f32> = (0..d_model).map(|i| i as f32 / d_model as f32).collect();
    (
        Tensor::from_data(TensorData::new(gamma, [d_model]), &Default::default()),
        Tensor::from_data(TensorData::new(beta, [d_model]), &Default::default()),
    )
}

/// Decomposed layer_norm, mirroring what `burn::nn::LayerNorm::forward`
/// does via primitive tensor ops. Applies along the last dim.
fn decomposed_layer_norm<const D: usize>(
    input: Tensor<D>,
    gamma: Tensor<1>,
    beta: Tensor<1>,
    epsilon: f32,
) -> Tensor<D> {
    let dim = D - 1;
    let mean = input.clone().mean_dim(dim);
    let centered = input - mean;
    let var = centered.clone().powi_scalar(2).mean_dim(dim);
    let normalized = centered / (var + epsilon).sqrt();
    normalized * gamma.unsqueeze() + beta.unsqueeze()
}

macro_rules! bench_backend {
    ($mod_name:ident, $backend_name:literal) => {
        #[divan::bench_group(name = $backend_name)]
        mod $mod_name {
            use super::*;

            // ConvNeXt-style: small last-dim C (48, 96, 192, 384), large
            // N*H*W. This is the shape the user's segmentation model hits
            // at the channels-last layer norm between convolution blocks.
            #[divan::bench_group(name = "convnext_layer_norm_decomposed")]
            mod convnext_layer_norm_decomposed {
                use super::*;

                #[divan::bench]
                fn c48_hw244x224(bencher: Bencher) {
                    let x = make_tensor_3d(1, 244 * 224, 48);
                    let (g, b) = make_gamma_beta(48);
                    bencher.bench_synced(|| {
                        decomposed_layer_norm(x.clone(), g.clone(), b.clone(), 1e-5)
                    });
                }

                #[divan::bench]
                fn c96_hw122x112(bencher: Bencher) {
                    let x = make_tensor_3d(1, 122 * 112, 96);
                    let (g, b) = make_gamma_beta(96);
                    bencher.bench_synced(|| {
                        decomposed_layer_norm(x.clone(), g.clone(), b.clone(), 1e-5)
                    });
                }

                #[divan::bench]
                fn c192_hw61x56(bencher: Bencher) {
                    let x = make_tensor_3d(1, 61 * 56, 192);
                    let (g, b) = make_gamma_beta(192);
                    bencher.bench_synced(|| {
                        decomposed_layer_norm(x.clone(), g.clone(), b.clone(), 1e-5)
                    });
                }

                #[divan::bench]
                fn c384_hw30x28(bencher: Bencher) {
                    let x = make_tensor_3d(1, 30 * 28, 384);
                    let (g, b) = make_gamma_beta(384);
                    bencher.bench_synced(|| {
                        decomposed_layer_norm(x.clone(), g.clone(), b.clone(), 1e-5)
                    });
                }

                // Transformer-style: typical `[batch=2, seq=512, d=768]`.
                #[divan::bench]
                fn d768_seq512_b2(bencher: Bencher) {
                    let x = make_tensor_3d(2, 512, 768);
                    let (g, b) = make_gamma_beta(768);
                    bencher.bench_synced(|| {
                        decomposed_layer_norm(x.clone(), g.clone(), b.clone(), 1e-5)
                    });
                }
            }

            // Routes through `B::layer_norm`; compare against the
            // `convnext_layer_norm_decomposed` group above.
            #[divan::bench_group(name = "convnext_layer_norm_fused_trait")]
            mod convnext_layer_norm_fused_trait {
                use super::*;

                #[divan::bench]
                fn c48_hw244x224(bencher: Bencher) {
                    let x = make_tensor_3d(1, 244 * 224, 48);
                    let (g, b) = make_gamma_beta(48);
                    bencher
                        .bench_synced(|| trait_layer_norm(x.clone(), g.clone(), b.clone(), 1e-5));
                }

                #[divan::bench]
                fn c96_hw122x112(bencher: Bencher) {
                    let x = make_tensor_3d(1, 122 * 112, 96);
                    let (g, b) = make_gamma_beta(96);
                    bencher
                        .bench_synced(|| trait_layer_norm(x.clone(), g.clone(), b.clone(), 1e-5));
                }

                #[divan::bench]
                fn c192_hw61x56(bencher: Bencher) {
                    let x = make_tensor_3d(1, 61 * 56, 192);
                    let (g, b) = make_gamma_beta(192);
                    bencher
                        .bench_synced(|| trait_layer_norm(x.clone(), g.clone(), b.clone(), 1e-5));
                }

                #[divan::bench]
                fn c384_hw30x28(bencher: Bencher) {
                    let x = make_tensor_3d(1, 30 * 28, 384);
                    let (g, b) = make_gamma_beta(384);
                    bencher
                        .bench_synced(|| trait_layer_norm(x.clone(), g.clone(), b.clone(), 1e-5));
                }

                #[divan::bench]
                fn d768_seq512_b2(bencher: Bencher) {
                    let x = make_tensor_3d(2, 512, 768);
                    let (g, b) = make_gamma_beta(768);
                    bencher
                        .bench_synced(|| trait_layer_norm(x.clone(), g.clone(), b.clone(), 1e-5));
                }
            }

            // Drill-down: time each primitive op used inside the
            // decomposition, so we can see which one drives the gap.
            // Shape is [1, 244*224, 48], matching the worst case above.
            #[divan::bench_group(name = "layer_norm_primitive_drilldown")]
            mod layer_norm_primitive_drilldown {
                use super::*;

                fn shape() -> Tensor<3> {
                    make_tensor_3d(1, 244 * 224, 48)
                }

                #[divan::bench]
                fn op1_mean_dim_last(bencher: Bencher) {
                    let x = shape();
                    bencher.bench_synced(|| x.clone().mean_dim(2));
                }

                #[divan::bench]
                fn op2_broadcast_sub(bencher: Bencher) {
                    let x = shape();
                    let m = x.clone().mean_dim(2);
                    bencher.bench_synced(|| x.clone() - m.clone());
                }

                #[divan::bench]
                fn op3_powi_scalar_2(bencher: Bencher) {
                    let x = shape();
                    bencher.bench_synced(|| x.clone().powi_scalar(2));
                }

                #[divan::bench]
                fn op4_broadcast_div(bencher: Bencher) {
                    let x = shape();
                    let m = x.clone().mean_dim(2);
                    bencher.bench_synced(|| x.clone() / m.clone());
                }

                #[divan::bench]
                fn op5_broadcast_mul_1d(bencher: Bencher) {
                    let x = shape();
                    let (g, _) = make_gamma_beta(48);
                    bencher.bench_synced(|| x.clone() * g.clone().unsqueeze());
                }

                #[divan::bench]
                fn op6_broadcast_add_1d(bencher: Bencher) {
                    let x = shape();
                    let (_, b) = make_gamma_beta(48);
                    bencher.bench_synced(|| x.clone() + b.clone().unsqueeze());
                }
            }

            // Drill-down for the "non-contig-input layer_norm" case.
            // Input is x.permute([0, 2, 3, 1]) from shape
            // [1, 48, 244, 224], so the permuted last dim has an
            // absurd stride (54656) and hurts every op that needs
            // row-contig access.
            #[divan::bench_group(name = "permuted_input_drilldown")]
            mod permuted_input_drilldown {
                use super::*;

                fn permuted() -> Tensor<4> {
                    let x = make_tensor_4d(1, 48, 244, 224);
                    x.permute([0, 2, 3, 1])
                }

                #[divan::bench]
                fn p1_permute_only(bencher: Bencher) {
                    let x = make_tensor_4d(1, 48, 244, 224);
                    bencher.bench_synced(|| x.clone().permute([0, 2, 3, 1]));
                }

                #[divan::bench]
                fn p2_mean_dim_last_on_permuted(bencher: Bencher) {
                    let p = permuted();
                    bencher.bench_synced(|| p.clone().mean_dim(3));
                }

                #[divan::bench]
                fn p3_broadcast_sub_on_permuted(bencher: Bencher) {
                    let p = permuted();
                    let m = p.clone().mean_dim(3);
                    bencher.bench_synced(|| p.clone() - m.clone());
                }
            }

            // ConvNeXt pattern: input is (N, C, H, W), reshape/permute to
            // (N, H, W, C), layer_norm over C, permute back. The full
            // composite is what the user's benchmark bundle called
            // "reshape + dims + layer_norm".
            #[divan::bench_group(name = "convnext_reshape_layer_norm_composite")]
            mod convnext_reshape_layer_norm_composite {
                use super::*;

                fn run<const D: usize>(x: Tensor<4>, g: Tensor<1>, b: Tensor<1>) -> Tensor<4> {
                    // (N, C, H, W) -> (N, H, W, C)
                    let y = x.permute([0, 2, 3, 1]);
                    let normed = decomposed_layer_norm(y, g, b, 1e-5);
                    // (N, H, W, C) -> (N, C, H, W)
                    normed.permute([0, 3, 1, 2])
                }

                #[divan::bench]
                fn c48_488x448(bencher: Bencher) {
                    let x = make_tensor_4d(1, 48, 244, 224);
                    let (g, b) = make_gamma_beta(48);
                    bencher.bench_synced(|| run::<3>(x.clone(), g.clone(), b.clone()));
                }

                #[divan::bench]
                fn c96_122x112(bencher: Bencher) {
                    let x = make_tensor_4d(1, 96, 122, 112);
                    let (g, b) = make_gamma_beta(96);
                    bencher.bench_synced(|| run::<3>(x.clone(), g.clone(), b.clone()));
                }

                #[divan::bench]
                fn c192_61x56(bencher: Bencher) {
                    let x = make_tensor_4d(1, 192, 61, 56);
                    let (g, b) = make_gamma_beta(192);
                    bencher.bench_synced(|| run::<3>(x.clone(), g.clone(), b.clone()));
                }

                #[divan::bench]
                fn c384_30x28(bencher: Bencher) {
                    let x = make_tensor_4d(1, 384, 30, 28);
                    let (g, b) = make_gamma_beta(384);
                    bencher.bench_synced(|| run::<3>(x.clone(), g.clone(), b.clone()));
                }
            }
        }
    };
}

bench_backend!(backend, "backend");
