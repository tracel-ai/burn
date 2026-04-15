//! Attention (scaled dot-product) for CPU.
//!
//! Computes: softmax(Q @ K^T * scale + bias) @ V
//!
//! Two strategies, auto-selected by `attention()` based on sequence length:
//!
//! - **Naive** (seq_kv <= 8*TILE_KV): materializes full score matrix, two
//!   large gemm calls per (batch, head). Faster for short sequences.
//! - **Flash** (seq_kv > 8*TILE_KV): tiles over KV with online softmax,
//!   O(TILE_KV) scratch per row. Better cache behavior for long sequences.

use alloc::vec;
use alloc::vec::Vec;
use burn_backend::DType;
use burn_backend::ops::AttentionModuleOptions;
use burn_std::Bytes;
use bytemuck::Pod;
use num_traits::Float;

use crate::{FlexTensor, Layout};

/// KV tile size for flash attention.
///
/// Chosen so the score row (TILE_KV * 4 bytes for f32) and the V tile
/// [TILE_KV, val_dim] fit comfortably in L1. With val_dim=128 the V tile
/// is 32KB; this fits well on Apple Silicon (64-128KB L1) and modern x86
/// (48KB+ L1d since Golden Cove / Zen 4). On older x86 with 32KB L1d the
/// tile saturates L1 but still benefits from L2 residency.
/// WASM targets use a smaller tile to stay within tighter cache budgets.
#[cfg(target_family = "wasm")]
const TILE_KV: usize = 32;
#[cfg(not(target_family = "wasm"))]
const TILE_KV: usize = 64;

/// Max score matrix size (in elements) for the naive path.
///
/// Naive attention materializes a [seq_q, seq_kv] score matrix per head.
/// When this exceeds the budget, flash attention is used instead.
/// 256K elements = 1 MB for f32, fits comfortably in L2.
const NAIVE_SCORE_BUDGET: usize = 256 * 1024;

/// Auto-selecting attention: picks the fastest strategy based on sequence length.
///
/// Uses naive attention when the score matrix (seq_q * seq_kv) fits within
/// `NAIVE_SCORE_BUDGET`. Falls back to flash attention for larger shapes.
pub fn attention(
    query: FlexTensor,
    key: FlexTensor,
    value: FlexTensor,
    mask: Option<FlexTensor>,
    attn_bias: Option<FlexTensor>,
    options: AttentionModuleOptions,
) -> FlexTensor {
    debug_assert!(
        query.layout().shape().num_dims() == 4,
        "attention: query must be 4D, got {}D",
        query.layout().shape().num_dims()
    );
    debug_assert!(
        key.layout().shape().num_dims() == 4,
        "attention: key must be 4D, got {}D",
        key.layout().shape().num_dims()
    );
    let seq_q = query.layout().shape()[2];
    let seq_kv = key.layout().shape()[2];
    if seq_q * seq_kv <= NAIVE_SCORE_BUDGET {
        return attention_naive(query, key, value, mask, attn_bias, options);
    }
    attention_flash(query, key, value, mask, attn_bias, options)
}

/// Dispatch attention by dtype, casting f16/bf16 to f32 for computation.
macro_rules! dispatch_attention_dtype {
    ($query:expr, $key:expr, $value:expr, $mask:expr, $attn_bias:expr, $options:expr, $impl_fn:ident) => {{
        let query = $query;
        let key = $key;
        let value = $value;
        let mask = $mask;
        let attn_bias = $attn_bias;
        let options = $options;
        let dtype = query.dtype();
        debug_assert_eq!(key.dtype(), dtype, "attention: key dtype mismatch");
        debug_assert_eq!(value.dtype(), dtype, "attention: value dtype mismatch");
        if let Some(ref b) = attn_bias {
            debug_assert_eq!(b.dtype(), dtype, "attention: attn_bias dtype mismatch");
        }
        match dtype {
            DType::F32 => $impl_fn::<f32>(query, key, value, mask, attn_bias, options),
            DType::F64 => $impl_fn::<f64>(query, key, value, mask, attn_bias, options),
            DType::F16 => {
                use burn_std::f16;
                let r = $impl_fn::<f32>(
                    cast_to_f32(query, f16::to_f32),
                    cast_to_f32(key, f16::to_f32),
                    cast_to_f32(value, f16::to_f32),
                    mask,
                    attn_bias.map(|b| cast_to_f32(b, f16::to_f32)),
                    options,
                );
                cast_from_f32(r, f16::from_f32)
            }
            DType::BF16 => {
                use burn_std::bf16;
                let r = $impl_fn::<f32>(
                    cast_to_f32(query, bf16::to_f32),
                    cast_to_f32(key, bf16::to_f32),
                    cast_to_f32(value, bf16::to_f32),
                    mask,
                    attn_bias.map(|b| cast_to_f32(b, bf16::to_f32)),
                    options,
                );
                cast_from_f32(r, bf16::from_f32)
            }
            dtype => panic!("attention: unsupported dtype {:?}", dtype),
        }
    }};
}

/// Contiguous mask/bias tensor plus the per-batch and per-head element offsets the
/// inner loop should use to locate the `[seq_q, seq_kv]` tile for each `(batch, head)`
/// pair. When a leading dim (batch or heads) is `1` in the source, its step is `0`, so
/// the inner loop re-reads the same tile for every pair without allocating an expanded
/// copy. The tile length itself is always `seq_q * seq_kv` and is computed at the call
/// site, so it is not stored here.
struct BroadcastMaskBias {
    tensor: FlexTensor,
    batch_step: usize,
    head_step: usize,
}

/// Prepare an attention mask or bias for the inner loop, accepting ONNX Attention-23
/// broadcast shapes.
///
/// Stride-0 along the leading `[batch, heads]` dims is handled without materializing,
/// so the common ONNX patterns (`[1, 1, seq_q, seq_kv]`, `[batch, 1, seq_q, seq_kv]`,
/// `[1, heads, seq_q, seq_kv]`) stay zero-copy. That matters especially for the flash
/// path, where materializing an expanded mask/bias would allocate a full
/// `[batch, heads, seq_q, seq_kv]` buffer and negate flash attention's memory
/// efficiency. If the trailing `[seq_q, seq_kv]` dims are themselves broadcast (rare in
/// practice), we fall back to `expand` + `to_contiguous` so the tile stays contiguous
/// in memory for the inner loop's slice-based access.
fn broadcast_attn_mask_bias(
    tensor: FlexTensor,
    target: [usize; 4],
    name: &'static str,
) -> BroadcastMaskBias {
    let ndim = tensor.layout().shape().num_dims();
    assert!(ndim == 4, "attention: {name} must be 4D, got {ndim}D");
    let shape = tensor.layout().shape();
    let src = [shape[0], shape[1], shape[2], shape[3]];
    for i in 0..4 {
        assert!(
            src[i] == target[i] || src[i] == 1,
            "attention: {name} dim {i} must be {} or 1, got {}",
            target[i],
            src[i]
        );
    }

    let tile_len = target[2] * target[3];

    // Broadcast on seq_q or seq_kv: the source's trailing tile has fewer elements
    // than `tile_len`, so per-pair slice access would under-read. Materialize via
    // expand + to_contiguous in that case.
    if src[2] != target[2] || src[3] != target[3] {
        let expanded = crate::ops::expand::expand(tensor, burn_std::Shape::new(target));
        return BroadcastMaskBias {
            tensor: expanded.to_contiguous(),
            batch_step: target[1] * tile_len,
            head_step: tile_len,
        };
    }

    // Trailing dims match the target. Keep the source at its own shape (size
    // `src[0] * src[1] * tile_len`) and zero-out the step for any leading dim of 1.
    BroadcastMaskBias {
        tensor: tensor.to_contiguous(),
        batch_step: if src[0] == 1 { 0 } else { src[1] * tile_len },
        head_step: if src[1] == 1 { 0 } else { tile_len },
    }
}

/// Flash attention: tiled computation with online softmax. Use directly to bypass auto-selection.
pub fn attention_flash(
    query: FlexTensor,
    key: FlexTensor,
    value: FlexTensor,
    mask: Option<FlexTensor>,
    attn_bias: Option<FlexTensor>,
    options: AttentionModuleOptions,
) -> FlexTensor {
    dispatch_attention_dtype!(query, key, value, mask, attn_bias, options, attention_impl)
}

fn cast_to_f32<E: burn_backend::Element + Pod + Copy>(
    tensor: FlexTensor,
    to_f32: fn(E) -> f32,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let data: &[E] = tensor.storage();
    let f32_data: Vec<f32> = data.iter().map(|&v| to_f32(v)).collect();
    FlexTensor::new(
        Bytes::from_elems(f32_data),
        Layout::contiguous(shape),
        DType::F32,
    )
}

fn cast_from_f32<E: burn_backend::Element + Pod + Copy>(
    tensor: FlexTensor,
    from_f32: fn(f32) -> E,
) -> FlexTensor {
    let tensor = tensor.to_contiguous();
    let shape = tensor.layout().shape().clone();
    let data: &[f32] = tensor.storage();
    let half_data: Vec<E> = data.iter().map(|&v| from_f32(v)).collect();
    FlexTensor::new(
        Bytes::from_elems(half_data),
        Layout::contiguous(shape),
        E::dtype(),
    )
}

/// Flash attention: tiled computation that avoids materializing the full scores matrix.
///
/// Input shapes (all 4D):
///   query: \[batch, heads, seq_q, head_dim\]
///   key:   \[batch, heads, seq_kv, head_dim\]
///   value: \[batch, heads, seq_kv, val_dim\]
///   mask:  \[batch, heads, seq_q, seq_kv\] (optional, u8 where nonzero = masked out)
///   attn_bias: \[batch, heads, seq_q, seq_kv\] (optional)
///
/// Output: \[batch, heads, seq_q, val_dim\]
///
/// Algorithm per (batch, head):
///   For each KV tile of size TILE_KV:
///     1. Score matmul: scores\[seq_q, tile_kv\] = Q @ K_tile^T  (gemm)
///     2. Per query row: apply scale, softcap, mask, bias
///     3. Per query row: online softmax update (running max/sum, rescale accumulator)
///     4. Value matmul: output += P @ V_tile  (gemm)
///   Final: output\[qi\] /= row_sum\[qi\] for each query row
fn attention_impl<T>(
    query: FlexTensor,
    key: FlexTensor,
    value: FlexTensor,
    mask: Option<FlexTensor>,
    attn_bias: Option<FlexTensor>,
    options: AttentionModuleOptions,
) -> FlexTensor
where
    T: FlashGemm + burn_backend::Element,
{
    if let Some(softcap) = options.softcap {
        assert!(softcap > 0.0, "softcap must be positive, got {softcap}");
    }

    let query = query.to_contiguous();
    let key = key.to_contiguous();
    let value = value.to_contiguous();

    let q_shape = query.layout().shape();
    let k_shape = key.layout().shape();
    let v_shape = value.layout().shape();
    assert!(q_shape.num_dims() == 4, "attention: query must be 4D");
    assert!(k_shape.num_dims() == 4, "attention: key must be 4D");
    assert!(v_shape.num_dims() == 4, "attention: value must be 4D");

    let batch = q_shape[0];
    let heads = q_shape[1];
    let seq_q = q_shape[2];
    let head_dim = q_shape[3];
    assert!(head_dim > 0, "attention: head_dim must be non-zero");

    let seq_kv = k_shape[2];
    let val_dim = v_shape[3];

    assert_eq!(k_shape[0], batch, "attention: key batch mismatch");
    assert_eq!(k_shape[1], heads, "attention: key heads mismatch");
    assert_eq!(k_shape[3], head_dim, "attention: key head_dim mismatch");
    assert_eq!(v_shape[0], batch, "attention: value batch mismatch");
    assert_eq!(v_shape[1], heads, "attention: value heads mismatch");
    assert_eq!(v_shape[2], seq_kv, "attention: value seq_kv mismatch");

    let target = [batch, heads, seq_q, seq_kv];
    let mask_bcast = mask.map(|m| broadcast_attn_mask_bias(m, target, "mask"));
    let bias_bcast = attn_bias.map(|b| broadcast_attn_mask_bias(b, target, "bias"));

    let scale = T::from(
        options
            .scale
            .unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt()),
    )
    .unwrap();
    let softcap: Option<T> = options.softcap.map(|s| T::from(s).unwrap());
    let causal_offset = if options.is_causal {
        Some(seq_kv as isize - seq_q as isize)
    } else {
        None
    };

    let q_data: &[T] = query.storage();
    let k_data: &[T] = key.storage();
    let v_data: &[T] = value.storage();
    let mask_data: Option<&[u8]> = mask_bcast.as_ref().map(|b| b.tensor.bytes());
    let bias_data: Option<&[T]> = bias_bcast.as_ref().map(|b| b.tensor.storage());
    let (mask_batch_step, mask_head_step) = mask_bcast
        .as_ref()
        .map(|b| (b.batch_step, b.head_step))
        .unwrap_or((0, 0));
    let (bias_batch_step, bias_head_step) = bias_bcast
        .as_ref()
        .map(|b| (b.batch_step, b.head_step))
        .unwrap_or((0, 0));

    let mut output = vec![T::zero(); batch * heads * seq_q * val_dim];

    // 4D strides for contiguous layout
    let q_head_stride = seq_q * head_dim;
    let q_batch_stride = heads * q_head_stride;
    let k_head_stride = seq_kv * head_dim;
    let k_batch_stride = heads * k_head_stride;
    let v_head_stride = seq_kv * val_dim;
    let v_batch_stride = heads * v_head_stride;
    let o_head_stride = seq_q * val_dim;
    let o_batch_stride = heads * o_head_stride;
    let mask_tile_len = seq_q * seq_kv;

    let params = AttentionParams {
        scale,
        softcap,
        causal_offset,
        seq_q,
        seq_kv,
        head_dim,
        val_dim,
    };

    // Allocate scratch buffers once and reuse across all (batch, head) pairs
    let mut scratch = ScratchBuffers {
        row_max: vec![T::neg_infinity(); seq_q],
        row_sum: vec![T::zero(); seq_q],
        scores: vec![T::zero(); seq_q * TILE_KV],
    };

    for b in 0..batch {
        for h in 0..heads {
            let q_off = b * q_batch_stride + h * q_head_stride;
            let k_off = b * k_batch_stride + h * k_head_stride;
            let v_off = b * v_batch_stride + h * v_head_stride;
            let o_off = b * o_batch_stride + h * o_head_stride;
            let mask_off = b * mask_batch_step + h * mask_head_step;
            let bias_off = b * bias_batch_step + h * bias_head_step;

            flash_attention_head(
                &q_data[q_off..q_off + q_head_stride],
                &k_data[k_off..k_off + k_head_stride],
                &v_data[v_off..v_off + v_head_stride],
                &mut output[o_off..o_off + o_head_stride],
                mask_data.map(|m| &m[mask_off..mask_off + mask_tile_len]),
                bias_data.map(|b| &b[bias_off..bias_off + mask_tile_len]),
                &params,
                &mut scratch,
            );
        }
    }

    let shape = burn_std::Shape::from(vec![batch, heads, seq_q, val_dim]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        T::dtype(),
    )
}

/// Gemm dispatch for flash attention block matmuls.
///
/// Wraps `gemm::gemm` for f32 and f64 so `flash_attention_head` stays generic.
/// Convention: dst = alpha * dst + beta * (lhs @ rhs)
trait FlashGemm: Float + Pod + Copy + core::ops::AddAssign {
    /// Block matrix multiply used for score and value matmuls.
    ///
    /// # Safety
    /// All pointers must be valid for the given dimensions and strides.
    unsafe fn block_gemm(args: BlockGemmArgs<Self>);
}

/// Arguments for a block matrix multiply: dst = alpha * dst + beta * (lhs @ rhs).
struct BlockGemmArgs<T> {
    m: usize,
    n: usize,
    k: usize,
    dst: *mut T,
    dst_cs: isize,
    dst_rs: isize,
    read_dst: bool,
    lhs: *const T,
    lhs_cs: isize,
    lhs_rs: isize,
    rhs: *const T,
    rhs_cs: isize,
    rhs_rs: isize,
    alpha: T,
    beta: T,
}

macro_rules! impl_flash_gemm {
    ($ty:ty) => {
        impl FlashGemm for $ty {
            unsafe fn block_gemm(a: BlockGemmArgs<Self>) {
                unsafe {
                    gemm::gemm(
                        a.m,
                        a.n,
                        a.k,
                        a.dst,
                        a.dst_cs,
                        a.dst_rs,
                        a.read_dst,
                        a.lhs,
                        a.lhs_cs,
                        a.lhs_rs,
                        a.rhs,
                        a.rhs_cs,
                        a.rhs_rs,
                        a.alpha,
                        a.beta,
                        false,
                        false,
                        false,
                        gemm::Parallelism::None,
                    );
                }
            }
        }
    };
}

impl_flash_gemm!(f32);
impl_flash_gemm!(f64);

/// Scratch buffers reused across (batch, head) pairs to avoid per-head allocation.
struct ScratchBuffers<T> {
    row_max: Vec<T>,
    row_sum: Vec<T>,
    scores: Vec<T>,
}

/// Parameters for a single (batch, head) flash attention computation.
struct AttentionParams<T> {
    scale: T,
    softcap: Option<T>,
    causal_offset: Option<isize>,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    val_dim: usize,
}

#[allow(clippy::too_many_arguments)]
/// Process a single (batch, head) pair with flash attention.
///
/// Uses gemm for the two block matmuls per tile:
///   1. Score matmul: scores\[seq_q, tile_kv\] = Q @ K_tile^T
///   2. Value matmul: output += P @ V_tile (where P = exp(scores - max))
///
/// The online softmax (scale, mask, bias, exp, correction) is applied
/// row-by-row between the two gemm calls.
fn flash_attention_head<T: FlashGemm>(
    q: &[T],
    k: &[T],
    v: &[T],
    output: &mut [T],
    mask: Option<&[u8]>,
    bias: Option<&[T]>,
    p: &AttentionParams<T>,
    scratch: &mut ScratchBuffers<T>,
) {
    debug_assert_eq!(q.len(), p.seq_q * p.head_dim);
    debug_assert_eq!(k.len(), p.seq_kv * p.head_dim);
    debug_assert_eq!(v.len(), p.seq_kv * p.val_dim);
    debug_assert_eq!(output.len(), p.seq_q * p.val_dim);

    let neg_inf = T::neg_infinity();
    let AttentionParams {
        scale,
        softcap,
        causal_offset,
        seq_q,
        seq_kv,
        head_dim,
        val_dim,
    } = *p;

    // Reset scratch buffers for this (batch, head) pair
    let row_max = &mut scratch.row_max;
    row_max.fill(neg_inf);
    let row_sum = &mut scratch.row_sum;
    row_sum.fill(T::zero());
    let scores = &mut scratch.scores;

    let num_kv_tiles = seq_kv.div_ceil(TILE_KV);

    for tile_idx in 0..num_kv_tiles {
        let kv_start = tile_idx * TILE_KV;
        let kv_end = (kv_start + TILE_KV).min(seq_kv);
        let tile_kv = kv_end - kv_start;

        // Step 1: Score matmul via gemm
        // scores[seq_q, tile_kv] = Q[seq_q, head_dim] @ K_tile[tile_kv, head_dim]^T
        //
        // Q is row-major [seq_q, head_dim]: rs=head_dim, cs=1
        // K_tile^T: treat K[tile_kv, head_dim] row-major with swapped strides
        //   K row-major: rs=head_dim, cs=1 -> K^T: rs=1, cs=head_dim
        // scores row-major [seq_q, tile_kv]: rs=tile_kv, cs=1
        unsafe {
            T::block_gemm(BlockGemmArgs {
                m: seq_q,
                n: tile_kv,
                k: head_dim,
                dst: scores.as_mut_ptr(),
                dst_cs: 1,
                dst_rs: tile_kv as isize,
                read_dst: false,
                lhs: q.as_ptr(),
                lhs_cs: 1,
                lhs_rs: head_dim as isize,
                rhs: k.as_ptr().add(kv_start * head_dim),
                rhs_cs: head_dim as isize,
                rhs_rs: 1,
                alpha: T::zero(),
                beta: T::one(),
            });
        }

        // Step 2: Apply scale/softcap/mask/bias, online softmax, rescale output
        for qi in 0..seq_q {
            let score_row = &mut scores[qi * tile_kv..(qi + 1) * tile_kv];

            let mut tile_max = neg_inf;

            for (ki, score) in score_row.iter_mut().enumerate() {
                let kv_idx = kv_start + ki;
                let mut val = *score * scale;

                if let Some(cap) = softcap {
                    val = cap * (val / cap).tanh();
                }

                if let Some(m) = mask
                    && m[qi * seq_kv + kv_idx] != 0
                {
                    val = neg_inf;
                }

                if let Some(offset) = causal_offset
                    && (kv_idx as isize) > (qi as isize) + offset
                {
                    val = neg_inf;
                }

                if let Some(b) = bias {
                    val += b[qi * seq_kv + kv_idx];
                }

                *score = val;
                if val > tile_max {
                    tile_max = val;
                }
            }

            if tile_max == neg_inf {
                // All masked: zero out scores so gemm contributes nothing
                for score in score_row.iter_mut() {
                    *score = T::zero();
                }
                continue;
            }

            let new_max = if row_max[qi] > tile_max {
                row_max[qi]
            } else {
                tile_max
            };

            let mut tile_sum = T::zero();
            for score in score_row.iter_mut() {
                let e = (*score - new_max).exp();
                *score = e;
                tile_sum += e;
            }

            let correction = if row_max[qi] == neg_inf {
                T::zero()
            } else {
                (row_max[qi] - new_max).exp()
            };

            // Rescale existing output accumulator
            let out_row = &mut output[qi * val_dim..(qi + 1) * val_dim];
            for o in out_row.iter_mut() {
                *o = *o * correction;
            }

            row_sum[qi] = row_sum[qi] * correction + tile_sum;
            row_max[qi] = new_max;
        }

        // Step 3: Value matmul via gemm
        // output[seq_q, val_dim] += P[seq_q, tile_kv] @ V_tile[tile_kv, val_dim]
        //
        // P (scores buffer) row-major: rs=tile_kv, cs=1
        // V_tile row-major [tile_kv, val_dim]: rs=val_dim, cs=1
        // output row-major [seq_q, val_dim]: rs=val_dim, cs=1
        //
        // dst = 1.0 * dst + 1.0 * P @ V  (accumulate)
        unsafe {
            T::block_gemm(BlockGemmArgs {
                m: seq_q,
                n: val_dim,
                k: tile_kv,
                dst: output.as_mut_ptr(),
                dst_cs: 1,
                dst_rs: val_dim as isize,
                read_dst: true,
                lhs: scores.as_ptr(),
                lhs_cs: 1,
                lhs_rs: tile_kv as isize,
                rhs: v.as_ptr().add(kv_start * val_dim),
                rhs_cs: 1,
                rhs_rs: val_dim as isize,
                alpha: T::one(),
                beta: T::one(),
            });
        }
    }

    // Final normalization: output /= row_sum
    for qi in 0..seq_q {
        let sum = row_sum[qi];
        if sum > T::zero() {
            let inv_sum = T::one() / sum;
            let out_row = &mut output[qi * val_dim..(qi + 1) * val_dim];
            for o in out_row.iter_mut() {
                *o = *o * inv_sum;
            }
        }
        // sum == 0 means all positions masked; output stays zero
    }
}

/// Naive attention: mathematically equivalent to flash but without KV tiling.
///
/// Materializes the full [seq_q, seq_kv] score matrix, applies scale/softcap/mask/causal/bias,
/// row-wise softmax, then a single output matmul. Two large gemm calls per (batch, head).
///
/// Faster than flash for short sequences. Also useful as a benchmarking baseline.
pub fn attention_naive(
    query: FlexTensor,
    key: FlexTensor,
    value: FlexTensor,
    mask: Option<FlexTensor>,
    attn_bias: Option<FlexTensor>,
    options: AttentionModuleOptions,
) -> FlexTensor {
    dispatch_attention_dtype!(
        query,
        key,
        value,
        mask,
        attn_bias,
        options,
        attention_naive_impl
    )
}

fn attention_naive_impl<T>(
    query: FlexTensor,
    key: FlexTensor,
    value: FlexTensor,
    mask: Option<FlexTensor>,
    attn_bias: Option<FlexTensor>,
    options: AttentionModuleOptions,
) -> FlexTensor
where
    T: FlashGemm + burn_backend::Element,
{
    if let Some(softcap) = options.softcap {
        assert!(softcap > 0.0, "softcap must be positive, got {softcap}");
    }

    let query = query.to_contiguous();
    let key = key.to_contiguous();
    let value = value.to_contiguous();

    let q_shape = query.layout().shape();
    let k_shape = key.layout().shape();
    let v_shape = value.layout().shape();
    assert!(q_shape.num_dims() == 4, "attention_naive: query must be 4D");
    assert!(k_shape.num_dims() == 4, "attention_naive: key must be 4D");
    assert!(v_shape.num_dims() == 4, "attention_naive: value must be 4D");

    let batch = q_shape[0];
    let heads = q_shape[1];
    let seq_q = q_shape[2];
    let head_dim = q_shape[3];
    assert!(head_dim > 0, "attention_naive: head_dim must be non-zero");

    let seq_kv = k_shape[2];
    let val_dim = v_shape[3];

    assert_eq!(k_shape[0], batch, "attention_naive: key batch mismatch");
    assert_eq!(k_shape[1], heads, "attention_naive: key heads mismatch");
    assert_eq!(
        k_shape[3], head_dim,
        "attention_naive: key head_dim mismatch"
    );
    assert_eq!(v_shape[0], batch, "attention_naive: value batch mismatch");
    assert_eq!(v_shape[1], heads, "attention_naive: value heads mismatch");
    assert_eq!(v_shape[2], seq_kv, "attention_naive: value seq_kv mismatch");

    let target = [batch, heads, seq_q, seq_kv];
    let mask_bcast = mask.map(|m| broadcast_attn_mask_bias(m, target, "mask"));
    let bias_bcast = attn_bias.map(|b| broadcast_attn_mask_bias(b, target, "bias"));

    let scale = T::from(
        options
            .scale
            .unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt()),
    )
    .unwrap();
    let softcap: Option<T> = options.softcap.map(|s| T::from(s).unwrap());
    let causal_offset = if options.is_causal {
        Some(seq_kv as isize - seq_q as isize)
    } else {
        None
    };

    let q_data: &[T] = query.storage();
    let k_data: &[T] = key.storage();
    let v_data: &[T] = value.storage();
    let mask_data: Option<&[u8]> = mask_bcast.as_ref().map(|b| b.tensor.bytes());
    let bias_data: Option<&[T]> = bias_bcast.as_ref().map(|b| b.tensor.storage());
    let (mask_batch_step, mask_head_step) = mask_bcast
        .as_ref()
        .map(|b| (b.batch_step, b.head_step))
        .unwrap_or((0, 0));
    let (bias_batch_step, bias_head_step) = bias_bcast
        .as_ref()
        .map(|b| (b.batch_step, b.head_step))
        .unwrap_or((0, 0));

    let mut output = vec![T::zero(); batch * heads * seq_q * val_dim];
    let mut scores = vec![T::zero(); seq_q * seq_kv];

    let q_head_stride = seq_q * head_dim;
    let q_batch_stride = heads * q_head_stride;
    let k_head_stride = seq_kv * head_dim;
    let k_batch_stride = heads * k_head_stride;
    let v_head_stride = seq_kv * val_dim;
    let v_batch_stride = heads * v_head_stride;
    let o_head_stride = seq_q * val_dim;
    let o_batch_stride = heads * o_head_stride;
    let mask_tile_len = seq_q * seq_kv;

    let params = AttentionParams {
        scale,
        softcap,
        causal_offset,
        seq_q,
        seq_kv,
        head_dim,
        val_dim,
    };

    for b in 0..batch {
        for h in 0..heads {
            let q_off = b * q_batch_stride + h * q_head_stride;
            let k_off = b * k_batch_stride + h * k_head_stride;
            let v_off = b * v_batch_stride + h * v_head_stride;
            let o_off = b * o_batch_stride + h * o_head_stride;
            let mask_off = b * mask_batch_step + h * mask_head_step;
            let bias_off = b * bias_batch_step + h * bias_head_step;

            naive_attention_head(
                &q_data[q_off..q_off + q_head_stride],
                &k_data[k_off..k_off + k_head_stride],
                &v_data[v_off..v_off + v_head_stride],
                &mut output[o_off..o_off + o_head_stride],
                &mut scores,
                &params,
                (
                    mask_data.map(|m| &m[mask_off..mask_off + mask_tile_len]),
                    bias_data.map(|b| &b[bias_off..bias_off + mask_tile_len]),
                ),
            );
        }
    }

    let shape = burn_std::Shape::from(vec![batch, heads, seq_q, val_dim]);
    FlexTensor::new(
        Bytes::from_elems(output),
        Layout::contiguous(shape),
        T::dtype(),
    )
}

/// Process a single (batch, head) pair with naive (non-tiled) attention.
///
/// 1. scores[seq_q, seq_kv] = Q @ K^T              (one gemm)
/// 2. Apply scale/softcap/mask/causal/bias + softmax (per-row)
/// 3. output[seq_q, val_dim] = scores @ V            (one gemm)
fn naive_attention_head<T: FlashGemm>(
    q: &[T],
    k: &[T],
    v: &[T],
    output: &mut [T],
    scores: &mut [T],
    p: &AttentionParams<T>,
    mask_bias: (Option<&[u8]>, Option<&[T]>),
) {
    let (mask, bias) = mask_bias;
    let neg_inf = T::neg_infinity();
    let AttentionParams {
        scale,
        softcap,
        causal_offset,
        seq_q,
        seq_kv,
        head_dim,
        val_dim,
    } = *p;

    // scores = Q @ K^T
    unsafe {
        T::block_gemm(BlockGemmArgs {
            m: seq_q,
            n: seq_kv,
            k: head_dim,
            dst: scores.as_mut_ptr(),
            dst_cs: 1,
            dst_rs: seq_kv as isize,
            read_dst: false,
            lhs: q.as_ptr(),
            lhs_cs: 1,
            lhs_rs: head_dim as isize,
            rhs: k.as_ptr(),
            rhs_cs: head_dim as isize,
            rhs_rs: 1,
            alpha: T::zero(),
            beta: T::one(),
        });
    }

    for qi in 0..seq_q {
        let row = &mut scores[qi * seq_kv..(qi + 1) * seq_kv];

        let mut row_max = neg_inf;
        for (ki, s) in row.iter_mut().enumerate() {
            let mut val = *s * scale;

            if let Some(cap) = softcap {
                val = cap * (val / cap).tanh();
            }

            if let Some(m) = mask
                && m[qi * seq_kv + ki] != 0
            {
                val = neg_inf;
            }

            if let Some(offset) = causal_offset
                && (ki as isize) > (qi as isize) + offset
            {
                val = neg_inf;
            }

            if let Some(b) = bias {
                val += b[qi * seq_kv + ki];
            }

            *s = val;
            if val > row_max {
                row_max = val;
            }
        }

        if row_max == neg_inf {
            row.fill(T::zero());
            continue;
        }

        let mut sum = T::zero();
        for s in row.iter_mut() {
            let e = (*s - row_max).exp();
            *s = e;
            sum += e;
        }

        let inv_sum = T::one() / sum;
        for s in row.iter_mut() {
            *s = *s * inv_sum;
        }
    }

    // output = softmax(scores) @ V
    unsafe {
        T::block_gemm(BlockGemmArgs {
            m: seq_q,
            n: val_dim,
            k: seq_kv,
            dst: output.as_mut_ptr(),
            dst_cs: 1,
            dst_rs: val_dim as isize,
            read_dst: false,
            lhs: scores.as_ptr(),
            lhs_cs: 1,
            lhs_rs: seq_kv as isize,
            rhs: v.as_ptr(),
            rhs_cs: 1,
            rhs_rs: val_dim as isize,
            alpha: T::zero(),
            beta: T::one(),
        });
    }
}
