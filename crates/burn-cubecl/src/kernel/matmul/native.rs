//! High-throughput native CubeCL/CUDA GEMM and GEMV kernels (`GemmBackend::Native`).
//!
//! Provides a selectable pure-Rust GPU kernel suite alongside `cuBLAS` (`GemmBackend::CuBlas`):
//! - **Small-M (`M <= 16`, Decode / Small Batch)**:
//!   - [`native_gemv_nt_vec4_kernel`]: Vectorized 64/128-bit (`Vector<E, 4>`) warp-per-column
//!     kernel for `NT` layout (`X[M, K] @ W^T[K, N]`), reusing `W` in registers across all `M`
//!     rows and reducing in `f32` via warp shuffle (`plane_sum`) with zero shared memory and
//!     zero warp divergence.
//!   - [`native_gemv_general_kernel`]: General warp-cooperative `plane_sum` kernel for `M <= 16`
//!     supporting arbitrary `(trans_lhs, trans_rhs)` and unaligned `K` (e.g. decode attention).
//! - **Large-M (`M > 16`, Prefill & Training)**:
//!   - [`native_wmma_gemm_kernel`]: Tensor-Core `16x16x16` (`cmma`) double-buffered kernel
//!     (`BM = 64, BN = 64, BK = 16`, `2x2` warps, `32x32` per warp = 4 explicit SSA accumulators
//!     for zero register spill), conflict-free padded shared memory (`stride = 24`), coalesced
//!     global-to-shared loaders for all 4 transpose modes (`NN`, `NT`, `TN`, `TT`), and L2 cache
//!     Group-M (`GROUP_M = 8`) CTA swizzling.
//!   - [`native_simt_gemm_kernel`]: Register-tiled (`4x4` per thread, `BM = 64, BN = 64, BK = 16`)
//!     SIMT kernel for `f32` or targets without `bf16`/`f16` `16x16x16` WMMA.

use crate::{kernel::into_contiguous, tensor::CubeTensor};
use burn_backend::DType;
use core::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use cubecl::{
    calculate_cube_count_elemwise,
    ir::{ElemType, FloatKind, features::MmaConfig},
    prelude::*,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GemmTranspose {
    NonTranspose,
    Transpose,
}

/// Active dense GEMM execution engine in `burn-cubecl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum GemmBackend {
    /// Default CubeK/autotuned matmul strategy (or stream-bound vendor GEMM when enabled).
    CuBlas = 1,
    /// Native CubeCL Tensor-Core WMMA, SIMT, and Warp-GEMV kernel suite.
    Native = 2,
}

static GEMM_BACKEND: AtomicU8 = AtomicU8::new(0);
static CUBLAS_GEMM_CALLS: AtomicU64 = AtomicU64::new(0);
static NATIVE_GEMM_CALLS: AtomicU64 = AtomicU64::new(0);

/// Returns the currently active [`GemmBackend`].
///
/// On first call, inspects `LGPT_GEMM` (`"native"` -> [`GemmBackend::Native`],
/// `"cublas"` -> [`GemmBackend::CuBlas`]), defaulting to [`GemmBackend::CuBlas`].
pub fn gemm_backend() -> GemmBackend {
    match GEMM_BACKEND.load(Ordering::Relaxed) {
        1 => GemmBackend::CuBlas,
        2 => GemmBackend::Native,
        _ => {
            let initial = match std::env::var("LGPT_GEMM") {
                Ok(val) if val.trim().eq_ignore_ascii_case("native") => GemmBackend::Native,
                _ => GemmBackend::CuBlas,
            };
            match GEMM_BACKEND.compare_exchange(
                0,
                initial as u8,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => initial,
                Err(2) => GemmBackend::Native,
                Err(_) => GemmBackend::CuBlas,
            }
        }
    }
}

/// Sets the active [`GemmBackend`] for subsequent `matmul` launches.
pub fn set_gemm_backend(backend: GemmBackend) {
    GEMM_BACKEND.store(backend as u8, Ordering::Relaxed);
}

/// Returns process-wide `(default_or_cublas_calls, native_calls)` launch counts.
pub fn gemm_calls() -> (u64, u64) {
    (
        CUBLAS_GEMM_CALLS.load(Ordering::Relaxed),
        NATIVE_GEMM_CALLS.load(Ordering::Relaxed),
    )
}

/// Resets process-wide `(default_or_cublas_calls, native_calls)` counters to zero.
pub fn reset_gemm_calls() {
    CUBLAS_GEMM_CALLS.store(0, Ordering::Relaxed);
    NATIVE_GEMM_CALLS.store(0, Ordering::Relaxed);
}

pub(crate) fn record_cublas_call() {
    CUBLAS_GEMM_CALLS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_native_call() {
    NATIVE_GEMM_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// Vectorized (`Vector<E, 4>`) Warp-GEMV / Small-M kernel for `NT` layout (`lhs` RowMajor `[M, K]`,
/// `rhs` Transposed `[K, N]` stored row-major as `[N, K]`).
///
/// Grid: 1 warp (32 lanes) per output column `n`, 4 warps per CTA (`CubeDim::new_2d(32, 4)`).
/// Each warp loads `rhs[n, k..k+4]` once and multiplies it against `lhs[0..m_tile, k..k+4]` in
/// registers, then reduces across the 32 lanes via `plane_sum`.
#[cube(launch, address_type = "dynamic")]
fn native_gemv_nt_vec4_kernel<E: Float, F: Float>(
    lhs: &Tensor<Vector<E, Const<4>>>,
    rhs: &Tensor<Vector<E, Const<4>>>,
    out: &mut Tensor<E>,
    m_row: usize,
    n_row: usize,
    k_vec: usize,
    ld_lhs_vec: usize,
    ld_rhs_vec: usize,
    ld_out: usize,
    b0: usize,
    b1: usize,
    stride_lhs_0_vec: usize,
    stride_lhs_1_vec: usize,
    stride_rhs_0_vec: usize,
    stride_rhs_1_vec: usize,
    stride_out_0: usize,
    stride_out_1: usize,
    #[comptime] m_cap: usize,
    #[define(E, F)] _dtypes: [ElemType; 2],
) {
    let cols_per_cta = 4usize;
    let ctas_per_batch = n_row.div_ceil(cols_per_cta);
    let total_batches = b0 * b1;
    let cube_id = CUBE_POS;
    let batch_id = cube_id / ctas_per_batch;
    if batch_id >= total_batches {
        terminate!();
    }

    let n_cta = (cube_id % ctas_per_batch) * cols_per_cta;
    let w = UNIT_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let n = n_cta + w;
    if n >= n_row {
        terminate!();
    }

    let b0_idx = batch_id / b1;
    let b1_idx = batch_id % b1;
    let base_lhs = b0_idx * stride_lhs_0_vec + b1_idx * stride_lhs_1_vec;
    let base_rhs = b0_idx * stride_rhs_0_vec + b1_idx * stride_rhs_1_vec + n * ld_rhs_vec;
    let base_out = b0_idx * stride_out_0 + b1_idx * stride_out_1;

    let mut acc = Array::<F>::new(m_cap);
    #[unroll]
    for mr in 0..m_cap {
        acc[mr] = F::new(0.0);
    }

    let mut kv = lane;
    while kv < k_vec {
        let w_v = Vector::<F, Const<4>>::cast_from(rhs[base_rhs + kv]);
        #[unroll]
        for mr in 0..m_cap {
            if mr < m_row {
                let x_v = Vector::<F, Const<4>>::cast_from(lhs[base_lhs + mr * ld_lhs_vec + kv]);
                let prod = x_v * w_v;
                acc[mr] += prod.extract(0usize)
                    + prod.extract(1usize)
                    + prod.extract(2usize)
                    + prod.extract(3usize);
            }
        }
        kv += 32;
    }

    #[unroll]
    for mr in 0..m_cap {
        if mr < m_row {
            let sum = plane_sum(acc[mr]);
            if lane == 0 {
                out[base_out + mr * ld_out + n] = E::cast_from(sum);
            }
        }
    }
}

/// General Warp-GEMV / Small-M (`M <= 16`) kernel for arbitrary `(trans_lhs, trans_rhs)` and `K`.
#[cube(launch, address_type = "dynamic")]
fn native_gemv_general_kernel<E: Float, F: Float>(
    lhs: &Tensor<E>,
    rhs: &Tensor<E>,
    out: &mut Tensor<E>,
    m_row: usize,
    n_row: usize,
    k_row: usize,
    ld_lhs: usize,
    ld_rhs: usize,
    ld_out: usize,
    b0: usize,
    b1: usize,
    stride_lhs_0: usize,
    stride_lhs_1: usize,
    stride_rhs_0: usize,
    stride_rhs_1: usize,
    stride_out_0: usize,
    stride_out_1: usize,
    #[comptime] trans_lhs: bool,
    #[comptime] trans_rhs: bool,
    #[comptime] m_cap: usize,
    #[define(E, F)] _dtypes: [ElemType; 2],
) {
    let cols_per_cta = 4usize;
    let ctas_per_batch = n_row.div_ceil(cols_per_cta);
    let total_batches = b0 * b1;
    let cube_id = CUBE_POS;
    let batch_id = cube_id / ctas_per_batch;
    if batch_id >= total_batches {
        terminate!();
    }

    let n_cta = (cube_id % ctas_per_batch) * cols_per_cta;
    let w = UNIT_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let n = n_cta + w;
    if n >= n_row {
        terminate!();
    }

    let b0_idx = batch_id / b1;
    let b1_idx = batch_id % b1;
    let base_lhs = b0_idx * stride_lhs_0 + b1_idx * stride_lhs_1;
    let base_rhs = b0_idx * stride_rhs_0 + b1_idx * stride_rhs_1;
    let base_out = b0_idx * stride_out_0 + b1_idx * stride_out_1;

    let mut acc = Array::<F>::new(m_cap);
    #[unroll]
    for mr in 0..m_cap {
        acc[mr] = F::new(0.0);
    }

    let mut k = lane;
    while k < k_row {
        let w_val = if trans_rhs {
            F::cast_from(rhs[base_rhs + n * ld_rhs + k])
        } else {
            F::cast_from(rhs[base_rhs + k * ld_rhs + n])
        };
        #[unroll]
        for mr in 0..m_cap {
            if mr < m_row {
                let x_val = if trans_lhs {
                    F::cast_from(lhs[base_lhs + k * ld_lhs + mr])
                } else {
                    F::cast_from(lhs[base_lhs + mr * ld_lhs + k])
                };
                acc[mr] += x_val * w_val;
            }
        }
        k += 32;
    }

    #[unroll]
    for mr in 0..m_cap {
        if mr < m_row {
            let sum = plane_sum(acc[mr]);
            if lane == 0 {
                out[base_out + mr * ld_out + n] = E::cast_from(sum);
            }
        }
    }
}

/// High-throughput Tensor-Core WMMA (`16x16x16`) double-buffered GEMM kernel (`BM = 64, BN = 64, BK = 16`).
///
/// - **CTA**: 128 threads (`CubeDim::new_2d(32, 4)` = 4 warps in a `2x2` grid).
/// - **Per-warp tile**: `32x32` output held in 4 explicit SSA `cmma::Matrix<F>` accumulators
///   (`acc_00`, `acc_01`, `acc_10`, `acc_11`) -> ~76 registers/thread, zero register spill.
/// - **Shared Memory**: Double-buffered `a_sh` (`[2, 64, 24]` RowMajor) and `b_sh` (`[2, 64, 24]`
///   ColMajor) with padded stride `24` (48 bytes = 12 banks) for bank-conflict-free `16x16` loads.
/// - **Coalesced global loaders**: Specialized via `#[comptime]` for all 4 `(trans_lhs, trans_rhs)`
///   modes (`NN`, `NT`, `TN`, `TT`) with a branch-free fast path for interior tiles.
/// - **L2 Cache Group-M Swizzling**: Groups 8 row-tiles of `A` across column-tiles of `B`.
#[cube(launch, address_type = "dynamic")]
fn native_wmma_gemm_kernel<E: Float, F: Float>(
    lhs: &Tensor<E>,
    rhs: &Tensor<E>,
    out: &mut Tensor<E>,
    m_row: usize,
    n_row: usize,
    k_row: usize,
    ld_lhs: usize,
    ld_rhs: usize,
    ld_out: usize,
    b0: usize,
    b1: usize,
    stride_lhs_0: usize,
    stride_lhs_1: usize,
    stride_rhs_0: usize,
    stride_rhs_1: usize,
    stride_out_0: usize,
    stride_out_1: usize,
    #[comptime] trans_lhs: bool,
    #[comptime] trans_rhs: bool,
    #[define(E, F)] _dtypes: [ElemType; 2],
) {
    let bm = 64usize;
    let bn = 64usize;
    let bk = 16usize;
    let sh_stride = 24usize;
    let stage_elems = 1536usize; // 64 * 24

    let grid_m = m_row.div_ceil(bm);
    let grid_n = n_row.div_ceil(bn);
    let tiles_per_batch = grid_m * grid_n;
    let total_batches = b0 * b1;

    let cube_id = CUBE_POS;
    let batch_id = cube_id / tiles_per_batch;
    if batch_id >= total_batches {
        terminate!();
    }

    // L2 Cache Group-M Swizzling (GROUP_M = 8)
    let tile_id = cube_id % tiles_per_batch;
    let group_m = 8usize;
    let pids_per_group = group_m * grid_n;
    let group_id = tile_id / pids_per_group;
    let first_pid_m = group_id * group_m;
    let rem_m = grid_m - first_pid_m;
    let group_size_m = if rem_m < group_m { rem_m } else { group_m };
    let in_group = tile_id % pids_per_group;
    let pid_m = first_pid_m + (in_group % group_size_m);
    let pid_n = in_group / group_size_m;

    let cta_m = pid_m * bm;
    let cta_n = pid_n * bn;

    let b0_idx = batch_id / b1;
    let b1_idx = batch_id % b1;
    let base_lhs = b0_idx * stride_lhs_0 + b1_idx * stride_lhs_1;
    let base_rhs = b0_idx * stride_rhs_0 + b1_idx * stride_rhs_1;
    let base_out = b0_idx * stride_out_0 + b1_idx * stride_out_1;

    let w = UNIT_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    let tid = w * 32 + lane;
    let warp_m = (w / 2) * 32;
    let warp_n = (w % 2) * 32;

    // Double-buffered shared memory: 2 stages * 64 * 24 = 3072 elements each
    let mut a_sh = Shared::<[E]>::new_aligned_slice(3072usize, 16usize);
    let mut b_sh = Shared::<[E]>::new_aligned_slice(3072usize, 16usize);
    let mut out_sh = Shared::<[F]>::new_aligned_slice(1024usize, 16usize);

    // 4 explicit SSA accumulators per warp (32x32 output tile) -> zero register spill
    let acc_00 = cmma::Matrix::<F>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        F::new(0.0),
    );
    let acc_01 = cmma::Matrix::<F>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        F::new(0.0),
    );
    let acc_10 = cmma::Matrix::<F>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        F::new(0.0),
    );
    let acc_11 = cmma::Matrix::<F>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        F::new(0.0),
    );

    let full_mn = cta_m + bm <= m_row && cta_n + bn <= n_row;
    let num_k_tiles = k_row.div_ceil(bk);

    // Preload stage 0 (k_tile = 0)
    if num_k_tiles > 0 {
        let k_base = 0usize;
        let full_tile = full_mn && (k_base + bk <= k_row);
        if full_tile {
            #[unroll]
            for s in 0..8usize {
                let idx = tid + s * 128;
                if trans_lhs {
                    // Coalesced along M: idx = k_local * 64 + m_local
                    let k_local = idx / 64;
                    let m_local = idx % 64;
                    a_sh[m_local * sh_stride + k_local] =
                        lhs[base_lhs + (k_base + k_local) * ld_lhs + (cta_m + m_local)];
                } else {
                    // Coalesced along K: idx = m_local * 16 + k_local
                    let m_local = idx / 16;
                    let k_local = idx % 16;
                    a_sh[m_local * sh_stride + k_local] =
                        lhs[base_lhs + (cta_m + m_local) * ld_lhs + (k_base + k_local)];
                }

                if trans_rhs {
                    // Coalesced along K: idx = n_local * 16 + k_local
                    let n_local = idx / 16;
                    let k_local = idx % 16;
                    b_sh[n_local * sh_stride + k_local] =
                        rhs[base_rhs + (cta_n + n_local) * ld_rhs + (k_base + k_local)];
                } else {
                    // Coalesced along N: idx = k_local * 64 + n_local
                    let k_local = idx / 64;
                    let n_local = idx % 64;
                    b_sh[n_local * sh_stride + k_local] =
                        rhs[base_rhs + (k_base + k_local) * ld_rhs + (cta_n + n_local)];
                }
            }
        } else {
            #[unroll]
            for s in 0..8usize {
                let idx = tid + s * 128;
                let m_local = idx / 16;
                let k_local = idx % 16;
                let gm = cta_m + m_local;
                let gk = k_base + k_local;
                let a_val = if gm < m_row && gk < k_row {
                    if trans_lhs {
                        lhs[base_lhs + gk * ld_lhs + gm]
                    } else {
                        lhs[base_lhs + gm * ld_lhs + gk]
                    }
                } else {
                    E::cast_from(0.0_f32)
                };
                a_sh[m_local * sh_stride + k_local] = a_val;

                let n_local = m_local;
                let gn = cta_n + n_local;
                let b_val = if gn < n_row && gk < k_row {
                    if trans_rhs {
                        rhs[base_rhs + gn * ld_rhs + gk]
                    } else {
                        rhs[base_rhs + gk * ld_rhs + gn]
                    }
                } else {
                    E::cast_from(0.0_f32)
                };
                b_sh[n_local * sh_stride + k_local] = b_val;
            }
        }
    }
    sync_cube();

    let mut kt = 0usize;
    while kt < num_k_tiles {
        let cur_stage = (kt % 2) * stage_elems;
        let next_kt = kt + 1;

        // Load cur_stage fragments into registers BEFORE writing next_stage or waiting
        let a0_off = cur_stage + warp_m * sh_stride;
        let a1_off = cur_stage + (warp_m + 16) * sh_stride;
        let b0_off = cur_stage + warp_n * sh_stride;
        let b1_off = cur_stage + (warp_n + 16) * sh_stride;

        let a_frag_0 = cmma::Matrix::<E>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &a_sh[a0_off..a0_off + 384],
            24u32,
        );
        let a_frag_1 = cmma::Matrix::<E>::from_slice(
            cmma::MatrixIdent::A,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::RowMajor,
            &a_sh[a1_off..a1_off + 384],
            24u32,
        );
        let b_frag_0 = cmma::Matrix::<E>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
            &b_sh[b0_off..b0_off + 384],
            24u32,
        );
        let b_frag_1 = cmma::Matrix::<E>::from_slice(
            cmma::MatrixIdent::B,
            16usize,
            16usize,
            16usize,
            cmma::MatrixLayout::ColMajor,
            &b_sh[b1_off..b1_off + 384],
            24u32,
        );

        // Execute 4 Tensor-Core 16x16x16 MMA instructions on cur_stage registers
        cmma::execute(&a_frag_0, &b_frag_0, &acc_00, &acc_00);
        cmma::execute(&a_frag_0, &b_frag_1, &acc_01, &acc_01);
        cmma::execute(&a_frag_1, &b_frag_0, &acc_10, &acc_10);
        cmma::execute(&a_frag_1, &b_frag_1, &acc_11, &acc_11);

        // Prefetch next_kt into the alternate shared-memory stage
        if next_kt < num_k_tiles {
            let next_stage = (next_kt % 2) * stage_elems;
            let k_base = next_kt * bk;
            let full_tile = full_mn && (k_base + bk <= k_row);
            if full_tile {
                #[unroll]
                for s in 0..8usize {
                    let idx = tid + s * 128;
                    if trans_lhs {
                        let k_local = idx / 64;
                        let m_local = idx % 64;
                        a_sh[next_stage + m_local * sh_stride + k_local] =
                            lhs[base_lhs + (k_base + k_local) * ld_lhs + (cta_m + m_local)];
                    } else {
                        let m_local = idx / 16;
                        let k_local = idx % 16;
                        a_sh[next_stage + m_local * sh_stride + k_local] =
                            lhs[base_lhs + (cta_m + m_local) * ld_lhs + (k_base + k_local)];
                    }

                    if trans_rhs {
                        let n_local = idx / 16;
                        let k_local = idx % 16;
                        b_sh[next_stage + n_local * sh_stride + k_local] =
                            rhs[base_rhs + (cta_n + n_local) * ld_rhs + (k_base + k_local)];
                    } else {
                        let k_local = idx / 64;
                        let n_local = idx % 64;
                        b_sh[next_stage + n_local * sh_stride + k_local] =
                            rhs[base_rhs + (k_base + k_local) * ld_rhs + (cta_n + n_local)];
                    }
                }
            } else {
                #[unroll]
                for s in 0..8usize {
                    let idx = tid + s * 128;
                    let m_local = idx / 16;
                    let k_local = idx % 16;
                    let gm = cta_m + m_local;
                    let gk = k_base + k_local;
                    let a_val = if gm < m_row && gk < k_row {
                        if trans_lhs {
                            lhs[base_lhs + gk * ld_lhs + gm]
                        } else {
                            lhs[base_lhs + gm * ld_lhs + gk]
                        }
                    } else {
                        E::cast_from(0.0_f32)
                    };
                    a_sh[next_stage + m_local * sh_stride + k_local] = a_val;

                    let n_local = m_local;
                    let gn = cta_n + n_local;
                    let b_val = if gn < n_row && gk < k_row {
                        if trans_rhs {
                            rhs[base_rhs + gn * ld_rhs + gk]
                        } else {
                            rhs[base_rhs + gk * ld_rhs + gn]
                        }
                    } else {
                        E::cast_from(0.0_f32)
                    };
                    b_sh[next_stage + n_local * sh_stride + k_local] = b_val;
                }
            }
        }

        sync_cube();
        kt = next_kt;
    }

    // Drain the 4 per-warp 16x16 accumulators through warp-local shared memory [w * 256 .. (w + 1) * 256]
    let w_sh_off = w * 256;
    let w_sh_end = w_sh_off + 256;

    if full_mn {
        // Branchless fast-path for interior 64x64 output tiles
        cmma::store(
            &mut out_sh[w_sh_off..w_sh_end],
            &acc_00,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for s in 0..8usize {
            let idx = lane + s * 32;
            let r = idx / 16;
            let c = idx % 16;
            out[base_out + (cta_m + warp_m + r) * ld_out + (cta_n + warp_n + c)] =
                E::cast_from(out_sh[w_sh_off + idx]);
        }
        sync_plane();

        cmma::store(
            &mut out_sh[w_sh_off..w_sh_end],
            &acc_01,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for s in 0..8usize {
            let idx = lane + s * 32;
            let r = idx / 16;
            let c = idx % 16;
            out[base_out + (cta_m + warp_m + r) * ld_out + (cta_n + warp_n + 16 + c)] =
                E::cast_from(out_sh[w_sh_off + idx]);
        }
        sync_plane();

        cmma::store(
            &mut out_sh[w_sh_off..w_sh_end],
            &acc_10,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for s in 0..8usize {
            let idx = lane + s * 32;
            let r = idx / 16;
            let c = idx % 16;
            out[base_out + (cta_m + warp_m + 16 + r) * ld_out + (cta_n + warp_n + c)] =
                E::cast_from(out_sh[w_sh_off + idx]);
        }
        sync_plane();

        cmma::store(
            &mut out_sh[w_sh_off..w_sh_end],
            &acc_11,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for s in 0..8usize {
            let idx = lane + s * 32;
            let r = idx / 16;
            let c = idx % 16;
            out[base_out + (cta_m + warp_m + 16 + r) * ld_out + (cta_n + warp_n + 16 + c)] =
                E::cast_from(out_sh[w_sh_off + idx]);
        }
    } else {
        // Boundary tile path
        cmma::store(
            &mut out_sh[w_sh_off..w_sh_end],
            &acc_00,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for s in 0..8usize {
            let idx = lane + s * 32;
            let r = idx / 16;
            let c = idx % 16;
            let gm = cta_m + warp_m + r;
            let gn = cta_n + warp_n + c;
            if gm < m_row && gn < n_row {
                out[base_out + gm * ld_out + gn] = E::cast_from(out_sh[w_sh_off + idx]);
            }
        }
        sync_plane();

        cmma::store(
            &mut out_sh[w_sh_off..w_sh_end],
            &acc_01,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for s in 0..8usize {
            let idx = lane + s * 32;
            let r = idx / 16;
            let c = idx % 16;
            let gm = cta_m + warp_m + r;
            let gn = cta_n + warp_n + 16 + c;
            if gm < m_row && gn < n_row {
                out[base_out + gm * ld_out + gn] = E::cast_from(out_sh[w_sh_off + idx]);
            }
        }
        sync_plane();

        cmma::store(
            &mut out_sh[w_sh_off..w_sh_end],
            &acc_10,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for s in 0..8usize {
            let idx = lane + s * 32;
            let r = idx / 16;
            let c = idx % 16;
            let gm = cta_m + warp_m + 16 + r;
            let gn = cta_n + warp_n + c;
            if gm < m_row && gn < n_row {
                out[base_out + gm * ld_out + gn] = E::cast_from(out_sh[w_sh_off + idx]);
            }
        }
        sync_plane();

        cmma::store(
            &mut out_sh[w_sh_off..w_sh_end],
            &acc_11,
            16u32,
            cmma::MatrixLayout::RowMajor,
        );
        sync_plane();
        #[unroll]
        for s in 0..8usize {
            let idx = lane + s * 32;
            let r = idx / 16;
            let c = idx % 16;
            let gm = cta_m + warp_m + 16 + r;
            let gn = cta_n + warp_n + 16 + c;
            if gm < m_row && gn < n_row {
                out[base_out + gm * ld_out + gn] = E::cast_from(out_sh[w_sh_off + idx]);
            }
        }
    }
}

/// Register-tiled (`4x4` per thread, `BM = 64, BN = 64, BK = 16`, 256 threads/CTA) SIMT GEMM
/// kernel for `f32` or targets without `16x16x16` WMMA.
#[cube(launch, address_type = "dynamic")]
fn native_simt_gemm_kernel<E: Float, F: Float>(
    lhs: &Tensor<E>,
    rhs: &Tensor<E>,
    out: &mut Tensor<E>,
    m_row: usize,
    n_row: usize,
    k_row: usize,
    ld_lhs: usize,
    ld_rhs: usize,
    ld_out: usize,
    b0: usize,
    b1: usize,
    stride_lhs_0: usize,
    stride_lhs_1: usize,
    stride_rhs_0: usize,
    stride_rhs_1: usize,
    stride_out_0: usize,
    stride_out_1: usize,
    #[comptime] trans_lhs: bool,
    #[comptime] trans_rhs: bool,
    #[define(E, F)] _dtypes: [ElemType; 2],
) {
    let bm = 64usize;
    let bn = 64usize;
    let bk = 16usize;
    let a_stride = 17usize; // 16 + 1 padding for zero bank conflicts
    let b_stride = 65usize; // 64 + 1 padding for zero bank conflicts

    let grid_m = m_row.div_ceil(bm);
    let grid_n = n_row.div_ceil(bn);
    let tiles_per_batch = grid_m * grid_n;
    let total_batches = b0 * b1;

    let cube_id = CUBE_POS;
    let batch_id = cube_id / tiles_per_batch;
    if batch_id >= total_batches {
        terminate!();
    }

    let tile_id = cube_id % tiles_per_batch;
    let group_m = 8usize;
    let pids_per_group = group_m * grid_n;
    let group_id = tile_id / pids_per_group;
    let first_pid_m = group_id * group_m;
    let rem_m = grid_m - first_pid_m;
    let group_size_m = if rem_m < group_m { rem_m } else { group_m };
    let in_group = tile_id % pids_per_group;
    let pid_m = first_pid_m + (in_group % group_size_m);
    let pid_n = in_group / group_size_m;

    let cta_m = pid_m * bm;
    let cta_n = pid_n * bn;

    let b0_idx = batch_id / b1;
    let b1_idx = batch_id % b1;
    let base_lhs = b0_idx * stride_lhs_0 + b1_idx * stride_lhs_1;
    let base_rhs = b0_idx * stride_rhs_0 + b1_idx * stride_rhs_1;
    let base_out = b0_idx * stride_out_0 + b1_idx * stride_out_1;

    let tx = UNIT_POS_X as usize; // 0..16
    let ty = UNIT_POS_Y as usize; // 0..16
    let tid = ty * 16 + tx; // 0..256
    let tm = ty * 4;
    let tn = tx * 4;

    let mut a_sh = Shared::<[F]>::new_aligned_slice(1088usize, 16usize); // 64 * 17
    let mut b_sh = Shared::<[F]>::new_aligned_slice(1040usize, 16usize); // 16 * 65

    let mut c00 = F::new(0.0);
    let mut c01 = F::new(0.0);
    let mut c02 = F::new(0.0);
    let mut c03 = F::new(0.0);
    let mut c10 = F::new(0.0);
    let mut c11 = F::new(0.0);
    let mut c12 = F::new(0.0);
    let mut c13 = F::new(0.0);
    let mut c20 = F::new(0.0);
    let mut c21 = F::new(0.0);
    let mut c22 = F::new(0.0);
    let mut c23 = F::new(0.0);
    let mut c30 = F::new(0.0);
    let mut c31 = F::new(0.0);
    let mut c32 = F::new(0.0);
    let mut c33 = F::new(0.0);

    let num_k_tiles = k_row.div_ceil(bk);
    let mut kt = 0usize;
    while kt < num_k_tiles {
        let k_base = kt * bk;
        #[unroll]
        for s in 0..4usize {
            let idx = tid + s * 256;
            let m_local = idx / 16;
            let k_local = idx % 16;
            let gm = cta_m + m_local;
            let gk = k_base + k_local;
            let a_val = if gm < m_row && gk < k_row {
                if trans_lhs {
                    F::cast_from(lhs[base_lhs + gk * ld_lhs + gm])
                } else {
                    F::cast_from(lhs[base_lhs + gm * ld_lhs + gk])
                }
            } else {
                F::new(0.0)
            };
            a_sh[m_local * a_stride + k_local] = a_val;

            let bk_local = idx / 64;
            let bn_local = idx % 64;
            let gbk = k_base + bk_local;
            let gn = cta_n + bn_local;
            let b_val = if gbk < k_row && gn < n_row {
                if trans_rhs {
                    F::cast_from(rhs[base_rhs + gn * ld_rhs + gbk])
                } else {
                    F::cast_from(rhs[base_rhs + gbk * ld_rhs + gn])
                }
            } else {
                F::new(0.0)
            };
            b_sh[bk_local * b_stride + bn_local] = b_val;
        }

        sync_cube();

        #[unroll]
        for kk in 0..16usize {
            let a0 = a_sh[(tm + 0) * a_stride + kk];
            let a1 = a_sh[(tm + 1) * a_stride + kk];
            let a2 = a_sh[(tm + 2) * a_stride + kk];
            let a3 = a_sh[(tm + 3) * a_stride + kk];

            let b_row = kk * b_stride + tn;
            let b0 = b_sh[b_row + 0];
            let b1 = b_sh[b_row + 1];
            let b2 = b_sh[b_row + 2];
            let b3 = b_sh[b_row + 3];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c02 += a0 * b2;
            c03 += a0 * b3;
            c10 += a1 * b0;
            c11 += a1 * b1;
            c12 += a1 * b2;
            c13 += a1 * b3;
            c20 += a2 * b0;
            c21 += a2 * b1;
            c22 += a2 * b2;
            c23 += a2 * b3;
            c30 += a3 * b0;
            c31 += a3 * b1;
            c32 += a3 * b2;
            c33 += a3 * b3;
        }

        sync_cube();
        kt += 1;
    }

    let gm0 = cta_m + tm;
    let gn0 = cta_n + tn;
    if gm0 + 0 < m_row {
        let row_off = base_out + (gm0 + 0) * ld_out;
        if gn0 + 0 < n_row {
            out[row_off + gn0 + 0] = E::cast_from(c00);
        }
        if gn0 + 1 < n_row {
            out[row_off + gn0 + 1] = E::cast_from(c01);
        }
        if gn0 + 2 < n_row {
            out[row_off + gn0 + 2] = E::cast_from(c02);
        }
        if gn0 + 3 < n_row {
            out[row_off + gn0 + 3] = E::cast_from(c03);
        }
    }
    if gm0 + 1 < m_row {
        let row_off = base_out + (gm0 + 1) * ld_out;
        if gn0 + 0 < n_row {
            out[row_off + gn0 + 0] = E::cast_from(c10);
        }
        if gn0 + 1 < n_row {
            out[row_off + gn0 + 1] = E::cast_from(c11);
        }
        if gn0 + 2 < n_row {
            out[row_off + gn0 + 2] = E::cast_from(c12);
        }
        if gn0 + 3 < n_row {
            out[row_off + gn0 + 3] = E::cast_from(c13);
        }
    }
    if gm0 + 2 < m_row {
        let row_off = base_out + (gm0 + 2) * ld_out;
        if gn0 + 0 < n_row {
            out[row_off + gn0 + 0] = E::cast_from(c20);
        }
        if gn0 + 1 < n_row {
            out[row_off + gn0 + 1] = E::cast_from(c21);
        }
        if gn0 + 2 < n_row {
            out[row_off + gn0 + 2] = E::cast_from(c22);
        }
        if gn0 + 3 < n_row {
            out[row_off + gn0 + 3] = E::cast_from(c23);
        }
    }
    if gm0 + 3 < m_row {
        let row_off = base_out + (gm0 + 3) * ld_out;
        if gn0 + 0 < n_row {
            out[row_off + gn0 + 0] = E::cast_from(c30);
        }
        if gn0 + 1 < n_row {
            out[row_off + gn0 + 1] = E::cast_from(c31);
        }
        if gn0 + 2 < n_row {
            out[row_off + gn0 + 2] = E::cast_from(c32);
        }
        if gn0 + 3 < n_row {
            out[row_off + gn0 + 3] = E::cast_from(c33);
        }
    }
}

fn classify_matrix_2d(
    rows: usize,
    cols: usize,
    sr: usize,
    sc: usize,
) -> Option<(GemmTranspose, usize)> {
    if sc == 1 && (rows <= 1 || sr >= cols) {
        let ld = if rows <= 1 {
            cols.max(1)
        } else {
            sr.max(cols).max(1)
        };
        Some((GemmTranspose::NonTranspose, ld))
    } else if sr == 1 && (cols <= 1 || sc >= rows) {
        let ld = if cols <= 1 {
            rows.max(1)
        } else {
            sc.max(rows).max(1)
        };
        Some((GemmTranspose::Transpose, ld))
    } else {
        None
    }
}

/// Validates 2D/3D/4D operands and launches the native GEMM/GEMV kernel suite.
pub(crate) fn try_launch_native_gemm(
    mut lhs: CubeTensor,
    mut rhs: CubeTensor,
    out: &CubeTensor,
) -> Option<()> {
    if lhs.qparams.is_some() || rhs.qparams.is_some() {
        return None;
    }
    if lhs.dtype != rhs.dtype || lhs.dtype != out.dtype {
        return None;
    }
    let elem_ty = match out.dtype {
        DType::BF16 => ElemType::Float(FloatKind::BF16),
        DType::F16 => ElemType::Float(FloatKind::F16),
        DType::F32 | DType::Flex32 => ElemType::Float(FloatKind::F32),
        _ => return None,
    };

    let rank = out.meta.num_dims();
    if !(2..=4).contains(&rank) || lhs.meta.num_dims() != rank || rhs.meta.num_dims() != rank {
        return None;
    }

    let m_row = out.meta.shape()[rank - 2];
    let n_row = out.meta.shape()[rank - 1];
    let k_row = lhs.meta.shape()[rank - 1];
    if rhs.meta.shape()[rank - 2] != k_row
        || lhs.meta.shape()[rank - 2] != m_row
        || rhs.meta.shape()[rank - 1] != n_row
    {
        return None;
    }
    if out.meta.num_elements() == 0 {
        return Some(());
    }
    if k_row == 0 {
        let zero = InputScalar::new(0u32, burn_backend::cubecl::dtype_to_storage_type(out.dtype));
        crate::ops::numeric::fill_device_dtype(out.clone(), zero);
        record_native_call();
        return Some(());
    }

    let out_sr = out.meta.strides()[rank - 2];
    let out_sc = out.meta.strides()[rank - 1];
    if out_sc != 1 || (m_row > 1 && out_sr < n_row) {
        return None;
    }
    let ld_out = if m_row <= 1 {
        n_row.max(1)
    } else {
        out_sr.max(n_row).max(1)
    };

    if classify_matrix_2d(
        m_row,
        k_row,
        lhs.meta.strides()[rank - 2],
        lhs.meta.strides()[rank - 1],
    )
    .is_none()
    {
        lhs = into_contiguous(lhs);
    }
    if classify_matrix_2d(
        k_row,
        n_row,
        rhs.meta.strides()[rank - 2],
        rhs.meta.strides()[rank - 1],
    )
    .is_none()
    {
        rhs = into_contiguous(rhs);
    }

    let (trans_rhs, ld_rhs) = classify_matrix_2d(
        k_row,
        n_row,
        rhs.meta.strides()[rank - 2],
        rhs.meta.strides()[rank - 1],
    )?;
    let (trans_lhs, ld_lhs) = classify_matrix_2d(
        m_row,
        k_row,
        lhs.meta.strides()[rank - 2],
        lhs.meta.strides()[rank - 1],
    )?;

    launch_native_gemm(
        lhs, rhs, out, elem_ty, m_row, n_row, k_row, ld_lhs, ld_rhs, ld_out, trans_lhs, trans_rhs,
    )
}

/// Launches the native GEMM/GEMV kernel suite on validated 2D/3D/4D operands.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_native_gemm(
    lhs: CubeTensor,
    rhs: CubeTensor,
    out: &CubeTensor,
    elem_ty: ElemType,
    m_row: usize,
    n_row: usize,
    k_row: usize,
    ld_lhs: usize,
    ld_rhs: usize,
    ld_out: usize,
    trans_lhs: GemmTranspose,
    trans_rhs: GemmTranspose,
) -> Option<()> {
    let rank = out.meta.num_dims();
    if rank > 4 {
        return None;
    }

    let (
        b0,
        b1,
        stride_lhs_0,
        stride_lhs_1,
        stride_rhs_0,
        stride_rhs_1,
        stride_out_0,
        stride_out_1,
    ) = match rank {
        2 => (
            1usize, 1usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize,
        ),
        3 => {
            let b = out.meta.shape()[0];
            let sl = if lhs.meta.shape()[0] <= 1 {
                0
            } else {
                lhs.meta.strides()[0]
            };
            let sr = if rhs.meta.shape()[0] <= 1 {
                0
            } else {
                rhs.meta.strides()[0]
            };
            let so = if b <= 1 { 0 } else { out.meta.strides()[0] };
            (1usize, b, 0usize, sl, 0usize, sr, 0usize, so)
        }
        4 => {
            let d0 = out.meta.shape()[0];
            let d1 = out.meta.shape()[1];
            let sl0 = if lhs.meta.shape()[0] <= 1 {
                0
            } else {
                lhs.meta.strides()[0]
            };
            let sl1 = if lhs.meta.shape()[1] <= 1 {
                0
            } else {
                lhs.meta.strides()[1]
            };
            let sr0 = if rhs.meta.shape()[0] <= 1 {
                0
            } else {
                rhs.meta.strides()[0]
            };
            let sr1 = if rhs.meta.shape()[1] <= 1 {
                0
            } else {
                rhs.meta.strides()[1]
            };
            let so0 = if d0 <= 1 { 0 } else { out.meta.strides()[0] };
            let so1 = if d1 <= 1 { 0 } else { out.meta.strides()[1] };
            (d0, d1, sl0, sl1, sr0, sr1, so0, so1)
        }
        _ => return None,
    };

    if b0 == 0 || b1 == 0 {
        return Some(());
    }

    let client = &out.client;
    let has_plane_32 = client.properties().hardware.plane_size_max == 32;

    let f32_ty = ElemType::Float(FloatKind::F32);
    let dtypes = [elem_ty, f32_ty];
    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type());

    let t_lhs = matches!(trans_lhs, GemmTranspose::Transpose);
    let t_rhs = matches!(trans_rhs, GemmTranspose::Transpose);
    let total_batches = b0 * b1;

    // 1. Small-M Decode / Small-Batch Path (M <= 16, requires 32-lane warp plane_sum)
    if m_row <= 16 && has_plane_32 {
        let m_cap = if m_row == 1 {
            1usize
        } else if m_row <= 4 {
            4usize
        } else if m_row <= 8 {
            8usize
        } else {
            16usize
        };
        let cube_dim = CubeDim::new_2d(32, 4);
        let ctas_per_batch = n_row.div_ceil(4);
        let num_cubes = total_batches * ctas_per_batch;
        let cube_count = calculate_cube_count_elemwise(
            client,
            num_cubes * (cube_dim.num_elems() as usize),
            cube_dim,
        );

        let vec4_aligned = !t_lhs
            && t_rhs
            && k_row >= 64
            && k_row % 4 == 0
            && ld_lhs % 4 == 0
            && ld_rhs % 4 == 0
            && stride_lhs_0 % 4 == 0
            && stride_lhs_1 % 4 == 0
            && stride_rhs_0 % 4 == 0
            && stride_rhs_1 % 4 == 0
            && lhs.handle.offset_start.unwrap_or(0) % (4 * out.dtype.size() as u64) == 0
            && rhs.handle.offset_start.unwrap_or(0) % (4 * out.dtype.size() as u64) == 0;

        if vec4_aligned {
            native_gemv_nt_vec4_kernel::launch(
                client,
                cube_count,
                cube_dim,
                address_type,
                lhs.into_tensor_arg(),
                rhs.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                m_row,
                n_row,
                k_row / 4,
                ld_lhs / 4,
                ld_rhs / 4,
                ld_out,
                b0,
                b1,
                stride_lhs_0 / 4,
                stride_lhs_1 / 4,
                stride_rhs_0 / 4,
                stride_rhs_1 / 4,
                stride_out_0,
                stride_out_1,
                m_cap,
                dtypes,
            );
        } else {
            native_gemv_general_kernel::launch(
                client,
                cube_count,
                cube_dim,
                address_type,
                lhs.into_tensor_arg(),
                rhs.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                m_row,
                n_row,
                k_row,
                ld_lhs,
                ld_rhs,
                ld_out,
                b0,
                b1,
                stride_lhs_0,
                stride_lhs_1,
                stride_rhs_0,
                stride_rhs_1,
                stride_out_0,
                stride_out_1,
                t_lhs,
                t_rhs,
                m_cap,
                dtypes,
            );
        }

        record_native_call();
        return Some(());
    }

    // 2. Large-M Prefill & Training Path (M > 16, or fallback when plane_size_max != 32)
    let has_wmma_16 = has_plane_32
        && matches!(out.dtype, DType::BF16 | DType::F16)
        && client
            .properties()
            .features
            .matmul
            .cmma
            .contains(&MmaConfig {
                a_type: elem_ty,
                b_type: elem_ty,
                cd_type: f32_ty,
                m: 16,
                k: 16,
                n: 16,
            });

    let grid_m = m_row.div_ceil(64);
    let grid_n = n_row.div_ceil(64);
    let num_cubes = total_batches * grid_m * grid_n;

    if has_wmma_16 {
        let cube_dim = CubeDim::new_2d(32, 4);
        let cube_count = calculate_cube_count_elemwise(
            client,
            num_cubes * (cube_dim.num_elems() as usize),
            cube_dim,
        );
        native_wmma_gemm_kernel::launch(
            client,
            cube_count,
            cube_dim,
            address_type,
            lhs.into_tensor_arg(),
            rhs.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            m_row,
            n_row,
            k_row,
            ld_lhs,
            ld_rhs,
            ld_out,
            b0,
            b1,
            stride_lhs_0,
            stride_lhs_1,
            stride_rhs_0,
            stride_rhs_1,
            stride_out_0,
            stride_out_1,
            t_lhs,
            t_rhs,
            dtypes,
        );
    } else {
        let cube_dim = CubeDim::new_2d(16, 16);
        let cube_count = calculate_cube_count_elemwise(
            client,
            num_cubes * (cube_dim.num_elems() as usize),
            cube_dim,
        );
        native_simt_gemm_kernel::launch(
            client,
            cube_count,
            cube_dim,
            address_type,
            lhs.into_tensor_arg(),
            rhs.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            m_row,
            n_row,
            k_row,
            ld_lhs,
            ld_rhs,
            ld_out,
            b0,
            b1,
            stride_lhs_0,
            stride_lhs_1,
            stride_rhs_0,
            stride_rhs_1,
            stride_out_0,
            stride_out_1,
            t_lhs,
            t_rhs,
            dtypes,
        );
    }

    record_native_call();
    Some(())
}
