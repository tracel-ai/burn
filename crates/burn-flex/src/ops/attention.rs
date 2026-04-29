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

// Tests kept here exercise flex-specific internals: direct calls into
// `attention_flash` / `attention_naive` (the public `attention()` dispatcher
// routes small shapes to naive so flash-path coverage requires a direct
// call), the `broadcast_attn_mask_bias` helper's rank/dim validation,
// flex-internal tiling and online-softmax correction paths, and
// dtype-specific kernels (f16/f64). Generic attention semantics (causal,
// custom scale, softcap, cross-attention, bool mask, additive bias,
// multi-batch/multi-head, single-element) live in
// crates/burn-backend-tests/tests/tensor/float/module/attention.rs, which
// exercises every backend. When adding new tests, keep them here only if
// they probe flex internals; otherwise add them to that suite.
#[cfg(test)]
mod tests {
    use alloc::{vec, vec::Vec};
    use burn_backend::DType;
    use burn_backend::ops::AttentionModuleOptions;
    use burn_std::{BoolStore, Bytes, Shape, f16};

    use crate::{FlexTensor, Layout};

    fn flex_f32(data: Vec<f32>, shape: &[usize]) -> FlexTensor {
        FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(shape.to_vec())),
            DType::F32,
        )
    }

    fn flex_f64(data: Vec<f64>, shape: &[usize]) -> FlexTensor {
        FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(shape.to_vec())),
            DType::F64,
        )
    }

    fn flex_f16(data: Vec<f16>, shape: &[usize]) -> FlexTensor {
        FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(shape.to_vec())),
            DType::F16,
        )
    }

    fn flex_bool(data: Vec<u8>, shape: &[usize]) -> FlexTensor {
        FlexTensor::new(
            Bytes::from_elems(data),
            Layout::contiguous(Shape::from(shape.to_vec())),
            DType::Bool(BoolStore::Native),
        )
    }

    /// Assert two attention-output slices are elementwise close. Used by the
    /// flash broadcast tests below.
    fn assert_attention_outputs_close(bcast: &[f32], full: &[f32], label: &str) {
        assert_eq!(bcast.len(), full.len(), "{label}: length mismatch");
        for (i, (&a, &b)) in bcast.iter().zip(full).enumerate() {
            assert!((a - b).abs() < 1e-5, "{label} mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_flash_bias_broadcast_across_batch_and_heads() {
        // Exercises the flash path for a `[1, 1, seq_q, seq_kv]` broadcast
        // bias. The dispatcher in `attention()` routes to naive for
        // `seq_q * seq_kv <= NAIVE_SCORE_BUDGET` (and the backend-tests
        // attention suite uses small shapes), so the flash entry needs a
        // direct call to stay covered. General broadcast semantics for the
        // main `attention()` path live in
        // crates/burn-backend-tests/tests/tensor/float/module/attention.rs.
        let batch = 2;
        let heads = 2;
        let seq_q = 3;
        let seq_kv = 5;
        let head_dim = 4;

        let mk = |shape: &[usize], g: &dyn Fn(usize) -> f32| -> FlexTensor {
            let len: usize = shape.iter().product();
            flex_f32((0..len).map(g).collect(), shape)
        };

        let q = mk(&[batch, heads, seq_q, head_dim], &|i| {
            (i as f32 * 0.1).sin()
        });
        let k = mk(&[batch, heads, seq_kv, head_dim], &|i| {
            (i as f32 * 0.1 + 1.0).sin()
        });
        let v = mk(&[batch, heads, seq_kv, head_dim], &|i| {
            (i as f32 * 0.1 + 2.0).sin()
        });

        let bias_tile: Vec<f32> = (0..seq_q * seq_kv)
            .map(|i| (i as f32 * 0.4).sin())
            .collect();
        let bias_bcast = flex_f32(bias_tile.clone(), &[1, 1, seq_q, seq_kv]);
        let bias_full_vec: Vec<f32> = bias_tile
            .iter()
            .cloned()
            .cycle()
            .take(batch * heads * seq_q * seq_kv)
            .collect();
        let bias_full = flex_f32(bias_full_vec, &[batch, heads, seq_q, seq_kv]);

        let out_bcast = super::attention_flash(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            Some(bias_bcast),
            Default::default(),
        );
        let out_full = super::attention_flash(q, k, v, None, Some(bias_full), Default::default());

        let bcast: &[f32] = out_bcast.storage();
        let full: &[f32] = out_full.storage();
        assert_attention_outputs_close(bcast, full, "flash bias[1,1,sq,skv]");
    }

    #[test]
    fn test_flash_bool_mask_broadcast_across_batch_and_heads() {
        // Flash-path counterpart to the bias broadcast test, but for a
        // `[1, 1, seq_q, seq_kv]` bool mask. The mask (u8) slicing path with
        // stride-0 batch/head steps is a different branch from the bias
        // (f32) path, so both need direct flash coverage.
        let batch = 2;
        let heads = 2;
        let seq_q = 3;
        let seq_kv = 5;
        let head_dim = 4;

        let mk = |shape: &[usize], g: &dyn Fn(usize) -> f32| -> FlexTensor {
            let len: usize = shape.iter().product();
            flex_f32((0..len).map(g).collect(), shape)
        };

        let q = mk(&[batch, heads, seq_q, head_dim], &|i| {
            (i as f32 * 0.1).sin()
        });
        let k = mk(&[batch, heads, seq_kv, head_dim], &|i| {
            (i as f32 * 0.1 + 1.0).sin()
        });
        let v = mk(&[batch, heads, seq_kv, head_dim], &|i| {
            (i as f32 * 0.1 + 2.0).sin()
        });

        // Mask out columns 3 and 4 (1 == masked out).
        let mask_tile: Vec<u8> = (0..seq_q * seq_kv)
            .map(|i| if (i % seq_kv) >= 3 { 1u8 } else { 0u8 })
            .collect();
        let mask_bcast = flex_bool(mask_tile.clone(), &[1, 1, seq_q, seq_kv]);
        let mask_full_vec: Vec<u8> = mask_tile
            .iter()
            .copied()
            .cycle()
            .take(batch * heads * seq_q * seq_kv)
            .collect();
        let mask_full = flex_bool(mask_full_vec, &[batch, heads, seq_q, seq_kv]);

        let out_bcast = super::attention_flash(
            q.clone(),
            k.clone(),
            v.clone(),
            Some(mask_bcast),
            None,
            Default::default(),
        );
        let out_full = super::attention_flash(q, k, v, Some(mask_full), None, Default::default());

        let bcast: &[f32] = out_bcast.storage();
        let full: &[f32] = out_full.storage();
        assert_attention_outputs_close(bcast, full, "flash bool mask[1,1,sq,skv]");
    }

    #[test]
    #[should_panic(expected = "must be 4D")]
    fn test_mask_wrong_rank_panics() {
        // Contract: the broadcast helper rejects non-4D mask/bias up-front
        // with a clearer message than `expand`'s rank prepending would produce.
        let mask = flex_bool(vec![0u8; 6], &[2, 3]);
        super::broadcast_attn_mask_bias(mask, [1, 1, 2, 3], "mask");
    }

    #[test]
    #[should_panic(expected = "bias dim 1 must be 3 or 1, got 2")]
    fn test_bias_incompatible_dim_panics() {
        // Contract: a dim that is neither equal to target nor `1` is a hard
        // error. Here heads=2 in the bias but target heads=3.
        let bias = flex_f32(vec![0.0f32; 2 * 2 * 4 * 5], &[2, 2, 4, 5]);
        super::broadcast_attn_mask_bias(bias, [2, 3, 4, 5], "bias");
    }

    #[test]
    fn test_multi_tile_seq_kv() {
        // seq_kv > TILE_KV (64) exercises tiling with online softmax
        // correction. 128 keys = exactly 2 tiles.
        let seq_q = 2;
        let seq_kv = 128;
        let head_dim = 4;
        let val_dim = 2;

        // Q: first query selects dim 0, second selects dim 1.
        let mut q_data = vec![0.0f32; seq_q * head_dim];
        q_data[0] = 1.0;
        q_data[head_dim + 1] = 1.0;

        // K: all keys identical so all scores are equal.
        let k_data = vec![0.1f32; seq_kv * head_dim];

        // V: linearly increasing values.
        let mut v_data = vec![0.0f32; seq_kv * val_dim];
        for i in 0..seq_kv {
            v_data[i * val_dim] = i as f32;
            v_data[i * val_dim + 1] = (seq_kv - 1 - i) as f32;
        }

        let q = flex_f32(q_data, &[1, 1, seq_q, head_dim]);
        let k = flex_f32(k_data, &[1, 1, seq_kv, head_dim]);
        let v = flex_f32(v_data, &[1, 1, seq_kv, val_dim]);

        let result = super::attention(q, k, v, None, None, Default::default());
        let data: &[f32] = result.storage();

        // All scores equal -> output = mean of 0..127 = 63.5 for both cols.
        assert_eq!(data.len(), seq_q * val_dim);
        assert!(
            (data[0] - 63.5).abs() < 0.1,
            "expected ~63.5, got {}",
            data[0]
        );
        assert!(
            (data[1] - 63.5).abs() < 0.1,
            "expected ~63.5, got {}",
            data[1]
        );
    }

    #[test]
    fn test_multi_tile_causal() {
        // seq_kv=128 with causal mask: each query sees only up to
        // causal_offset + its own index worth of keys.
        let seq_q = 4;
        let seq_kv = 128;
        let head_dim = 2;
        let val_dim = 1;

        // All queries/keys [1, 0] so all visible scores equal.
        let mut q_data = vec![0.0f32; seq_q * head_dim];
        for i in 0..seq_q {
            q_data[i * head_dim] = 1.0;
        }
        let mut k_data = vec![0.0f32; seq_kv * head_dim];
        for i in 0..seq_kv {
            k_data[i * head_dim] = 1.0;
        }
        let v_data: Vec<f32> = (0..seq_kv).map(|i| i as f32).collect();

        let q = flex_f32(q_data, &[1, 1, seq_q, head_dim]);
        let k = flex_f32(k_data, &[1, 1, seq_kv, head_dim]);
        let v = flex_f32(v_data, &[1, 1, seq_kv, val_dim]);

        let opts = AttentionModuleOptions {
            is_causal: true,
            ..Default::default()
        };
        let result = super::attention(q, k, v, None, None, opts);
        let data: &[f32] = result.storage();

        // causal_offset = seq_kv - seq_q = 124
        // q[0] sees 0..=124 -> mean 62.0; q[1] 62.5; q[2] 63.0; q[3] 63.5.
        assert_eq!(data.len(), seq_q);
        assert!((data[0] - 62.0).abs() < 0.1, "q0: got {}", data[0]);
        assert!((data[1] - 62.5).abs() < 0.1, "q1: got {}", data[1]);
        assert!((data[2] - 63.0).abs() < 0.1, "q2: got {}", data[2]);
        assert!((data[3] - 63.5).abs() < 0.1, "q3: got {}", data[3]);
    }

    #[test]
    fn test_tile_boundary_mask() {
        // Mask falls exactly on a tile boundary: first 64 keys masked,
        // next 64 visible.
        let seq_q = 1;
        let seq_kv = 128;
        let head_dim = 2;
        let val_dim = 1;

        let q_data = vec![1.0f32, 0.0];
        let k_data = vec![1.0f32, 0.0].repeat(seq_kv);
        let v_data: Vec<f32> = (0..seq_kv).map(|i| i as f32).collect();
        // true == masked out.
        let mask_data: Vec<u8> = (0..seq_kv).map(|i| (i < 64) as u8).collect();

        let q = flex_f32(q_data, &[1, 1, seq_q, head_dim]);
        let k = flex_f32(k_data, &[1, 1, seq_kv, head_dim]);
        let v = flex_f32(v_data, &[1, 1, seq_kv, val_dim]);
        let mask = flex_bool(mask_data, &[1, 1, seq_q, seq_kv]);

        let result = super::attention(q, k, v, Some(mask), None, Default::default());
        let data: &[f32] = result.storage();

        // Visible = 64..128 -> mean of 64..=127 = 95.5.
        assert!(
            (data[0] - 95.5).abs() < 0.1,
            "expected ~95.5, got {}",
            data[0]
        );
    }

    #[test]
    fn test_non_uniform_scores_across_tiles() {
        // Forces the online softmax correction path: tile 1 has much
        // larger scores than tile 0, so the correction factor
        // exp(old_max - new_max) < 1.
        let seq_q = 1;
        let seq_kv = 128;
        let head_dim = 1;
        let val_dim = 1;

        let q_data = vec![1.0f32];
        let mut k_data = vec![0.0f32; seq_kv];
        for k in k_data.iter_mut().take(64) {
            *k = 0.1;
        }
        for k in k_data.iter_mut().skip(64) {
            *k = 5.0;
        }
        let mut v_data = vec![0.0f32; seq_kv];
        for v in v_data.iter_mut().skip(64) {
            *v = 1.0;
        }

        let q = flex_f32(q_data, &[1, 1, seq_q, head_dim]);
        let k = flex_f32(k_data, &[1, 1, seq_kv, head_dim]);
        let v = flex_f32(v_data, &[1, 1, seq_kv, val_dim]);

        let result = super::attention(q, k, v, None, None, Default::default());
        let data: &[f32] = result.storage();

        // Large-score keys (tile 2) dominate softmax -> output ~ 1.0.
        assert!(data[0] > 0.99, "expected ~1.0, got {}", data[0]);
    }

    #[test]
    fn test_partial_last_tile() {
        // seq_kv=100 is not a multiple of TILE_KV=64, so the last tile has
        // 36 elements -> exercises partial-tile gemm and score buffer
        // indexing.
        let seq_q = 2;
        let seq_kv = 100;
        let head_dim = 2;
        let val_dim = 1;

        let q_data = vec![0.1f32, 0.1].repeat(seq_q);
        let k_data = vec![0.1f32, 0.1].repeat(seq_kv);
        let v_data: Vec<f32> = (0..seq_kv).map(|i| i as f32).collect();

        let q = flex_f32(q_data, &[1, 1, seq_q, head_dim]);
        let k = flex_f32(k_data, &[1, 1, seq_kv, head_dim]);
        let v = flex_f32(v_data, &[1, 1, seq_kv, val_dim]);

        let result = super::attention(q, k, v, None, None, Default::default());
        let data: &[f32] = result.storage();

        // Uniform attention -> mean of 0..99 = 49.5.
        assert_eq!(data.len(), seq_q);
        assert!(
            (data[0] - 49.5).abs() < 0.1,
            "expected ~49.5, got {}",
            data[0]
        );
        assert!(
            (data[1] - 49.5).abs() < 0.1,
            "expected ~49.5, got {}",
            data[1]
        );
    }

    /// Verify naive attention produces the same results as flash attention
    /// across various configurations.
    #[test]
    fn test_naive_matches_flash() {
        /// Deterministic tensor for cross-validation tests.
        ///
        /// Uses `i * 997 % N` as a cheap hash: 997 is prime and coprime to
        /// any power-of-two length, so the sequence visits all residues
        /// before repeating. For f32 this gives values in [-0.5, 0.5]. For
        /// bool masks it gives ~30% density with an irregular pattern that
        /// varies across rows (unlike a regular `i % 3` stride).
        fn make_tensor(shape: &[usize], dtype: DType) -> FlexTensor {
            let len: usize = shape.iter().product();
            match dtype {
                DType::F32 => {
                    let data: Vec<f32> =
                        (0..len).map(|i| ((i % 997) as f32 / 997.0) - 0.5).collect();
                    flex_f32(data, shape)
                }
                DType::Bool(_) => {
                    let data: Vec<u8> = (0..len)
                        .map(|i| (i.wrapping_mul(997) % 100 < 30) as u8)
                        .collect();
                    flex_bool(data, shape)
                }
                _ => unreachable!(),
            }
        }

        fn run_both(
            batch: usize,
            heads: usize,
            seq_q: usize,
            seq_kv: usize,
            head_dim: usize,
            val_dim: usize,
            with_mask: bool,
            with_bias: bool,
            options: AttentionModuleOptions,
            label: &str,
        ) {
            let f32_dt = DType::F32;
            let bool_dt = DType::Bool(BoolStore::Native);
            let q = make_tensor(&[batch, heads, seq_q, head_dim], f32_dt);
            let k = make_tensor(&[batch, heads, seq_kv, head_dim], f32_dt);
            let v = make_tensor(&[batch, heads, seq_kv, val_dim], f32_dt);
            let score_shape = [batch, heads, seq_q, seq_kv];
            let mask = with_mask.then(|| make_tensor(&score_shape, bool_dt));
            let bias = with_bias.then(|| make_tensor(&score_shape, f32_dt));

            let flash = super::attention_flash(
                q.clone(),
                k.clone(),
                v.clone(),
                mask.clone(),
                bias.clone(),
                options,
            );
            let naive = super::attention_naive(q, k, v, mask, bias, options);

            let flash_data: &[f32] = flash.storage();
            let naive_data: &[f32] = naive.storage();
            assert_eq!(
                flash_data.len(),
                naive_data.len(),
                "{label}: length mismatch"
            );

            for (i, (&f, &n)) in flash_data.iter().zip(naive_data.iter()).enumerate() {
                let diff = (f - n).abs();
                let tol = 1e-4 * f.abs().max(n.abs()).max(1.0);
                assert!(
                    diff < tol,
                    "{label}: position {i}: flash={f} vs naive={n}, diff={diff}"
                );
            }
        }

        let default = AttentionModuleOptions::default();
        let causal = AttentionModuleOptions {
            is_causal: true,
            ..Default::default()
        };
        let all_opts = AttentionModuleOptions {
            scale: Some(0.05),
            softcap: Some(30.0),
            is_causal: true,
        };

        run_both(1, 1, 4, 4, 8, 8, false, false, default, "basic_4x4");
        run_both(
            2,
            4,
            8,
            8,
            16,
            16,
            false,
            false,
            default,
            "multi_head_batch",
        );
        run_both(1, 2, 4, 32, 16, 16, false, false, default, "cross_attn");
        run_both(1, 1, 4, 128, 16, 16, false, false, default, "multi_tile");
        run_both(1, 2, 16, 16, 32, 32, false, false, causal, "causal");
        run_both(2, 2, 16, 16, 32, 32, false, false, all_opts, "all_options");
        run_both(1, 1, 32, 256, 64, 64, false, false, causal, "large_causal");
        run_both(1, 2, 8, 8, 16, 16, true, false, default, "with_mask");
        run_both(1, 2, 8, 8, 16, 16, false, true, default, "with_bias");
        run_both(
            2,
            2,
            16,
            128,
            32,
            32,
            true,
            true,
            causal,
            "mask_bias_causal",
        );
        run_both(1, 1, 4, 100, 16, 16, false, false, default, "partial_tile");
        run_both(
            1,
            2,
            8,
            100,
            16,
            16,
            false,
            false,
            causal,
            "partial_tile_causal",
        );
    }

    #[test]
    fn test_f64_flash_attention() {
        let q = flex_f64(vec![1.0f64, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
        let k = q.clone();
        let v = flex_f64(vec![10.0f64, 20.0], &[1, 1, 2, 1]);

        let result = super::attention(q, k, v, None, None, Default::default());
        let data: &[f64] = result.storage();

        assert!((data[0] - 13.30).abs() < 0.1, "got {}", data[0]);
        assert!((data[1] - 16.70).abs() < 0.1, "got {}", data[1]);
    }

    /// Verify f16 cast-to-f32 round-trip produces correct results for both
    /// paths.
    #[test]
    fn test_f16_attention() {
        let q_vec: Vec<f16> = [1.0f32, 0.0, 0.0, 1.0]
            .iter()
            .map(|&v| f16::from_f32(v))
            .collect();
        let v_vec: Vec<f16> = [10.0f32, 20.0].iter().map(|&v| f16::from_f32(v)).collect();
        let q = flex_f16(q_vec, &[1, 1, 2, 2]);
        let k = q.clone();
        let v = flex_f16(v_vec, &[1, 1, 2, 1]);

        let flash = super::attention_flash(
            q.clone(),
            k.clone(),
            v.clone(),
            None,
            None,
            Default::default(),
        );
        let naive = super::attention_naive(q, k, v, None, None, Default::default());

        let flash_data: &[f16] = flash.storage();
        let naive_data: &[f16] = naive.storage();

        // softmax([1/sqrt(2), 0]) = [0.670, 0.330] -> row0: ~13.3, row1: ~16.7
        for (label, data) in [("flash", flash_data), ("naive", naive_data)] {
            let r0 = data[0].to_f32();
            let r1 = data[1].to_f32();
            assert!((r0 - 13.30).abs() < 0.2, "{label} row0: got {r0}");
            assert!((r1 - 16.70).abs() < 0.2, "{label} row1: got {r1}");
        }
    }
}
