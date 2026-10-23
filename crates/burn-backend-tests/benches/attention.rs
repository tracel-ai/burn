//! Scaled dot-product attention benchmarks.
//!
//! Run with:
//! ```bash
//! cargo bench --bench attention
//! ```

#[path = "common/mod.rs"]
mod common;
use common::BencherExt;

use burn_tensor::module::attention;
use burn_tensor::ops::AttentionModuleOptions;
use burn_tensor::{FloatDType, Tensor, TensorData};
use divan::Bencher;

#[cfg(not(feature = "bench-disable-alloc"))]
#[global_allocator]
static ALLOC: divan::AllocProfiler = divan::AllocProfiler::system();

fn main() {
    println!("Attention Benchmarks");
    println!("Memory allocation tracking enabled");
    println!();
    divan::main();
    common::report_failures();
}

/// Create Q, K, V tensors for attention benchmarking.
fn make_qkv(
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
    let dev = Default::default();
    let make = |rows: usize, cols: usize| -> Tensor<4> {
        let n = batch * heads * rows * cols;
        let data: Vec<f32> = (0..n).map(|i| ((i % 1000) as f32 / 1000.0) - 0.5).collect();
        Tensor::from_data(TensorData::new(data, [batch, heads, rows, cols]), &dev)
    };
    (
        make(seq_q, head_dim),
        make(seq_k, head_dim),
        make(seq_k, head_dim),
    )
}

/// Create an additive bias tensor [batch, heads, seq_q, seq_k].
fn make_bias(batch: usize, heads: usize, seq_q: usize, seq_k: usize) -> Tensor<4> {
    let n = batch * heads * seq_q * seq_k;
    let data: Vec<f32> = (0..n).map(|i| ((i % 500) as f32 / 500.0) - 0.5).collect();
    Tensor::from_data(
        TensorData::new(data, [batch, heads, seq_q, seq_k]),
        &Default::default(),
    )
}

macro_rules! bench_attention {
    ($mod_name:ident, $name:literal) => {
        #[divan::bench_group(name = $name)]
        mod $mod_name {
            use super::*;

            // Short decoding contexts expose scheduling overhead that the
            // larger self-attention benchmarks do not capture.
            #[divan::bench(args = [1, 64, 128, 256, 512])]
            fn decode_b1_h8_d64(bencher: Bencher, seq_kv: usize) {
                let (q, k, v) = make_qkv(1, 8, 1, seq_kv, 64);
                bencher.bench_synced(|| {
                    attention(
                        q.clone(),
                        k.clone(),
                        v.clone(),
                        None,
                        None,
                        Default::default(),
                    )
                });
            }

            #[divan::bench_group(name = "self_attention")]
            mod self_attention {
                use super::*;

                #[divan::bench]
                fn b1_h8_s64_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv(1, 8, 64, 64, 64);
                    bencher.bench_synced(|| {
                        attention(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            None,
                            None,
                            Default::default(),
                        )
                    });
                }

                #[divan::bench]
                fn b1_h12_s128_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv(1, 12, 128, 128, 64);
                    bencher.bench_synced(|| {
                        attention(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            None,
                            None,
                            Default::default(),
                        )
                    });
                }

                #[divan::bench]
                fn b1_h12_s256_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv(1, 12, 256, 256, 64);
                    bencher.bench_synced(|| {
                        attention(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            None,
                            None,
                            Default::default(),
                        )
                    });
                }

                #[divan::bench]
                fn b1_h12_s512_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv(1, 12, 512, 512, 64);
                    bencher.bench_synced(|| {
                        attention(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            None,
                            None,
                            Default::default(),
                        )
                    });
                }

                #[divan::bench]
                fn b1_h32_s256_d128(bencher: Bencher) {
                    let (q, k, v) = make_qkv(1, 32, 256, 256, 128);
                    bencher.bench_synced(|| {
                        attention(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            None,
                            None,
                            Default::default(),
                        )
                    });
                }

                #[divan::bench]
                fn b4_h12_s128_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv(4, 12, 128, 128, 64);
                    bencher.bench_synced(|| {
                        attention(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            None,
                            None,
                            Default::default(),
                        )
                    });
                }
            }

            #[divan::bench_group(name = "causal")]
            mod causal {
                use super::*;

                fn causal_opts() -> AttentionModuleOptions {
                    AttentionModuleOptions {
                        is_causal: true,
                        ..Default::default()
                    }
                }

                #[divan::bench]
                fn b1_h12_s128_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv(1, 12, 128, 128, 64);
                    bencher.bench_synced(|| {
                        attention(q.clone(), k.clone(), v.clone(), None, None, causal_opts())
                    });
                }

                #[divan::bench]
                fn b1_h12_s256_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv(1, 12, 256, 256, 64);
                    bencher.bench_synced(|| {
                        attention(q.clone(), k.clone(), v.clone(), None, None, causal_opts())
                    });
                }

                #[divan::bench]
                fn b1_h12_s512_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv(1, 12, 512, 512, 64);
                    bencher.bench_synced(|| {
                        attention(q.clone(), k.clone(), v.clone(), None, None, causal_opts())
                    });
                }
            }

            #[divan::bench_group(name = "with_bias")]
            mod with_bias {
                use super::*;

                #[divan::bench]
                fn b1_h12_s128_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv(1, 12, 128, 128, 64);
                    let bias = make_bias(1, 12, 128, 128);
                    bencher.bench_synced(|| {
                        attention(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            None,
                            Some(bias.clone()),
                            Default::default(),
                        )
                    });
                }

                #[divan::bench]
                fn b1_h12_s256_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv(1, 12, 256, 256, 64);
                    let bias = make_bias(1, 12, 256, 256);
                    bencher.bench_synced(|| {
                        attention(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            None,
                            Some(bias.clone()),
                            Default::default(),
                        )
                    });
                }
            }

            #[divan::bench_group(name = "cross_attention")]
            mod cross_attention {
                use super::*;

                #[divan::bench]
                fn b1_h12_sq128_sk512_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv(1, 12, 128, 512, 64);
                    bencher.bench_synced(|| {
                        attention(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            None,
                            None,
                            Default::default(),
                        )
                    });
                }

                #[divan::bench]
                fn b1_h12_sq32_sk1024_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv(1, 12, 32, 1024, 64);
                    bencher.bench_synced(|| {
                        attention(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            None,
                            None,
                            Default::default(),
                        )
                    });
                }
            }

            // Qwen3-8B's heads (32 query heads of 128 features sharing 8 key heads): a prompt of
            // 512 tokens and decode steps of 8 users against 4096 cached positions, in f16, and a
            // training step's causal prompts in f32.
            #[divan::bench_group(name = "transformer")]
            mod transformer {
                use super::*;

                fn causal_opts() -> AttentionModuleOptions {
                    AttentionModuleOptions {
                        is_causal: true,
                        ..Default::default()
                    }
                }

                /// The query of 32 heads and the keys and values of 8, `[batch, heads, seq, 128]`.
                fn grouped(
                    batch: usize,
                    seq_q: usize,
                    seq_k: usize,
                ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
                    let (q, _, _) = make_qkv(batch, 32, seq_q, seq_k, 128);
                    let (_, k, v) = make_qkv(batch, 8, seq_q, seq_k, 128);
                    (q, k, v)
                }

                fn half(
                    (q, k, v): (Tensor<4>, Tensor<4>, Tensor<4>),
                ) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
                    (
                        q.cast(FloatDType::F16),
                        k.cast(FloatDType::F16),
                        v.cast(FloatDType::F16),
                    )
                }

                #[divan::bench]
                fn prefill_b1_h32_kv8_s512_d128(bencher: Bencher) {
                    let (q, k, v) = half(grouped(1, 512, 512));
                    bencher.bench_synced(|| {
                        attention(q.clone(), k.clone(), v.clone(), None, None, causal_opts())
                    });
                }

                #[divan::bench]
                fn decode_b8_h32_kv8_s4096_d128(bencher: Bencher) {
                    let (q, k, v) = half(grouped(8, 1, 4096));
                    bencher.bench_synced(|| {
                        attention(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            None,
                            None,
                            Default::default(),
                        )
                    });
                }

                #[divan::bench]
                fn training_f32_b4_h32_kv8_s512_d128(bencher: Bencher) {
                    let (q, k, v) = grouped(4, 512, 512);
                    bencher.bench_synced(|| {
                        attention(q.clone(), k.clone(), v.clone(), None, None, causal_opts())
                    });
                }
            }
        }
    };
}

bench_attention!(backend, "backend");
