//! Scaled dot-product attention benchmarks.
//!
//! Run with:
//! ```bash
//! cargo bench --bench attention
//! ```

#[path = "common/mod.rs"]
mod common;
use common::{BencherExt, TestBackend};

use burn_tensor::module::attention;
use burn_tensor::ops::AttentionModuleOptions;
use burn_tensor::{Tensor, TensorData, backend::Backend};
use divan::{AllocProfiler, Bencher};

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

fn main() {
    println!("Attention Benchmarks");
    println!("Memory allocation tracking enabled");
    println!();
    divan::main();
    common::report_failures();
}

/// Create Q, K, V tensors for attention benchmarking.
fn make_qkv<B: Backend>(
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    let dev = Default::default();
    let make = |rows: usize, cols: usize| -> Tensor<B, 4> {
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
fn make_bias<B: Backend>(batch: usize, heads: usize, seq_q: usize, seq_k: usize) -> Tensor<B, 4> {
    let n = batch * heads * seq_q * seq_k;
    let data: Vec<f32> = (0..n).map(|i| ((i % 500) as f32 / 500.0) - 0.5).collect();
    Tensor::from_data(
        TensorData::new(data, [batch, heads, seq_q, seq_k]),
        &Default::default(),
    )
}

macro_rules! bench_attention {
    ($backend:ty, $mod_name:ident, $name:literal) => {
        #[divan::bench_group(name = $name)]
        mod $mod_name {
            use super::*;
            type B = $backend;

            #[divan::bench_group(name = "self_attention")]
            mod self_attention {
                use super::*;

                #[divan::bench]
                fn b1_h8_s64_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv::<B>(1, 8, 64, 64, 64);
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
                    let (q, k, v) = make_qkv::<B>(1, 12, 128, 128, 64);
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
                    let (q, k, v) = make_qkv::<B>(1, 12, 256, 256, 64);
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
                    let (q, k, v) = make_qkv::<B>(1, 12, 512, 512, 64);
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
                    let (q, k, v) = make_qkv::<B>(1, 32, 256, 256, 128);
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
                    let (q, k, v) = make_qkv::<B>(4, 12, 128, 128, 64);
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
                    let (q, k, v) = make_qkv::<B>(1, 12, 128, 128, 64);
                    bencher.bench_synced(|| {
                        attention(q.clone(), k.clone(), v.clone(), None, None, causal_opts())
                    });
                }

                #[divan::bench]
                fn b1_h12_s256_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv::<B>(1, 12, 256, 256, 64);
                    bencher.bench_synced(|| {
                        attention(q.clone(), k.clone(), v.clone(), None, None, causal_opts())
                    });
                }

                #[divan::bench]
                fn b1_h12_s512_d64(bencher: Bencher) {
                    let (q, k, v) = make_qkv::<B>(1, 12, 512, 512, 64);
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
                    let (q, k, v) = make_qkv::<B>(1, 12, 128, 128, 64);
                    let bias = make_bias::<B>(1, 12, 128, 128);
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
                    let (q, k, v) = make_qkv::<B>(1, 12, 256, 256, 64);
                    let bias = make_bias::<B>(1, 12, 256, 256);
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
                    let (q, k, v) = make_qkv::<B>(1, 12, 128, 512, 64);
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
                    let (q, k, v) = make_qkv::<B>(1, 12, 32, 1024, 64);
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
        }
    };
}

bench_attention!(TestBackend, backend, "backend");
