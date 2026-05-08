//! Momentum transformation for AdamW 8bit.

use cubecl::prelude::*;

use crate::kernel::blockwise::{self, Scheme};
use crate::kernel::signed;
use crate::kernel::unsigned;
use crate::launch::{PACK_SHIFT, PACKING_AMOUNT, PLANE_SIZE};

#[cube]
pub fn transform(
    block: u32,
    grad: &Array<f32>,

    moment_1_codes: &Array<u32>,
    moment_1_scales: &Array<f32>,
    moment_2_codes: &Array<u32>,
    moment_2_scales: &Array<f32>,
    max_moment_2_codes: &Array<u32>,
    max_moment_2_scales: &Array<f32>,

    moment_1_codes_new: &mut Array<u32>,
    moment_1_scales_new: &mut Array<f32>,
    moment_2_codes_new: &mut Array<u32>,
    moment_2_scales_new: &mut Array<f32>,
    max_moment_2_codes_new: &mut Array<u32>,
    max_moment_2_scales_new: &mut Array<f32>,

    update_delta: &mut Array<f32>,
    m1_dequantized: &mut Array<f32>,

    beta_1: f32,
    beta_2: f32,
    factor_1: f32,
    factor_2: f32,
    correction1: f32,
    correction2_sqrt: f32,
    epsilon: f32,

    #[comptime] block_size: u32,
    #[comptime] amsgrad: bool,
    #[comptime] is_first_step: bool,
) {
    let unit = UNIT_POS_X;
    let block_start = block * block_size;
    let elements_per_thread = comptime!(block_size / PLANE_SIZE);

    // ============ Pass 1: per-element work, accumulate local absmax ============

    let mut m1_local_absmax = 0.0_f32;
    let mut m2_local_absmax = 0.0_f32;
    let mut v_local_absmax = 0.0_f32;

    let step_size = correction2_sqrt / correction1;

    // #[unroll]
    for iter in 0..elements_per_thread {
        let element = unit + iter * PLANE_SIZE;
        let i = block_start + element;

        let m1_old = if comptime!(is_first_step) {
            0.0_f32.into()
        } else {
            blockwise::dequantize_blockwise(
                moment_1_codes,
                moment_1_scales,
                i,
                block_size,
                Scheme::Signed as u32,
            )
        };
        let m2_old = if comptime!(is_first_step) {
            0.0_f32.into()
        } else {
            blockwise::dequantize_blockwise(
                moment_2_codes,
                moment_2_scales,
                i,
                block_size,
                Scheme::Unsigned as u32,
            )
        };

        let g = grad[i as usize];
        let m1_new = m1_old * beta_1 + g * factor_1;
        let m2_new = m2_old * beta_2 + g * g * factor_2;

        let v_to_use = if comptime!(amsgrad) {
            let max_v_old = if comptime!(is_first_step) {
                m2_new
            } else {
                blockwise::dequantize_blockwise(
                    max_moment_2_codes,
                    max_moment_2_scales,
                    i,
                    block_size,
                    Scheme::Unsigned as u32,
                )
            };
            max(max_v_old, m2_new)
        } else {
            m2_new
        };

        let delta = m1_new / (v_to_use.sqrt() + epsilon * correction2_sqrt) * step_size;

        update_delta[i as usize] = delta;
        m1_dequantized[i as usize] = m1_new;

        m1_local_absmax = max(m1_local_absmax, m1_new.abs());
        m2_local_absmax = max(m2_local_absmax, m2_new.abs());
        if comptime!(amsgrad) {
            v_local_absmax = max(v_local_absmax, v_to_use.abs());
        }
    }

    // ============ Reduction across the cube ============

    let m1_block_absmax = plane_max(m1_local_absmax);
    let m2_block_absmax = plane_max(m2_local_absmax);
    let m1_safe_scale = if m1_block_absmax > 0.0_f32 {
        m1_block_absmax
    } else {
        1.0_f32.into()
    };
    let m2_safe_scale = if m2_block_absmax > 0.0_f32 {
        m2_block_absmax
    } else {
        1.0_f32.into()
    };

    if unit == 0 {
        moment_1_scales_new[block as usize] = m1_safe_scale;
        moment_2_scales_new[block as usize] = m2_safe_scale;
    }

    // ============ Pass 2: 4-pack encoding (stride PLANE_SIZE * 4) ============

    let quads_per_thread = comptime!(block_size / (PLANE_SIZE * PACKING_AMOUNT));

    // #[unroll]
    for iter in 0..quads_per_thread {
        let element = unit * PACKING_AMOUNT + iter * PLANE_SIZE * PACKING_AMOUNT;
        let i = block_start + element;

        // Recompute m1_new and m2_new for all 4 elements of the quad.
        let m1_olds_0 = if comptime!(is_first_step) {
            0.0_f32.into()
        } else {
            blockwise::dequantize_blockwise(
                moment_1_codes,
                moment_1_scales,
                i,
                block_size,
                Scheme::Signed as u32,
            )
        };
        let m1_olds_1 = if comptime!(is_first_step) {
            0.0_f32.into()
        } else {
            blockwise::dequantize_blockwise(
                moment_1_codes,
                moment_1_scales,
                i + 1,
                block_size,
                Scheme::Signed as u32,
            )
        };
        let m1_olds_2 = if comptime!(is_first_step) {
            0.0_f32.into()
        } else {
            blockwise::dequantize_blockwise(
                moment_1_codes,
                moment_1_scales,
                i + 2,
                block_size,
                Scheme::Signed as u32,
            )
        };
        let m1_olds_3 = if comptime!(is_first_step) {
            0.0_f32.into()
        } else {
            blockwise::dequantize_blockwise(
                moment_1_codes,
                moment_1_scales,
                i + 3,
                block_size,
                Scheme::Signed as u32,
            )
        };
        let m2_olds_0 = if comptime!(is_first_step) {
            0.0_f32.into()
        } else {
            blockwise::dequantize_blockwise(
                moment_2_codes,
                moment_2_scales,
                i,
                block_size,
                Scheme::Unsigned as u32,
            )
        };
        let m2_olds_1 = if comptime!(is_first_step) {
            0.0_f32.into()
        } else {
            blockwise::dequantize_blockwise(
                moment_2_codes,
                moment_2_scales,
                i + 1,
                block_size,
                Scheme::Unsigned as u32,
            )
        };
        let m2_olds_2 = if comptime!(is_first_step) {
            0.0_f32.into()
        } else {
            blockwise::dequantize_blockwise(
                moment_2_codes,
                moment_2_scales,
                i + 2,
                block_size,
                Scheme::Unsigned as u32,
            )
        };
        let m2_olds_3 = if comptime!(is_first_step) {
            0.0_f32.into()
        } else {
            blockwise::dequantize_blockwise(
                moment_2_codes,
                moment_2_scales,
                i + 3,
                block_size,
                Scheme::Unsigned as u32,
            )
        };

        let g_0 = grad[i as usize];
        let g_1 = grad[i as usize + 1];
        let g_2 = grad[i as usize + 2];
        let g_3 = grad[i as usize + 3];

        let m1_new_0 = m1_olds_0 * beta_1 + g_0 * factor_1;
        let m1_new_1 = m1_olds_1 * beta_1 + g_1 * factor_1;
        let m1_new_2 = m1_olds_2 * beta_1 + g_2 * factor_1;
        let m1_new_3 = m1_olds_3 * beta_1 + g_3 * factor_1;
        let m2_new_0 = m2_olds_0 * beta_2 + g_0 * g_0 * factor_2;
        let m2_new_1 = m2_olds_1 * beta_2 + g_1 * g_1 * factor_2;
        let m2_new_2 = m2_olds_2 * beta_2 + g_2 * g_2 * factor_2;
        let m2_new_3 = m2_olds_3 * beta_2 + g_3 * g_3 * factor_2;

        let m1_c0 = signed::encode(m1_new_0 / m1_safe_scale);
        let m1_c1 = signed::encode(m1_new_1 / m1_safe_scale);
        let m1_c2 = signed::encode(m1_new_2 / m1_safe_scale);
        let m1_c3 = signed::encode(m1_new_3 / m1_safe_scale);
        let m2_c0 = unsigned::encode(m2_new_0 / m2_safe_scale);
        let m2_c1 = unsigned::encode(m2_new_1 / m2_safe_scale);
        let m2_c2 = unsigned::encode(m2_new_2 / m2_safe_scale);
        let m2_c3 = unsigned::encode(m2_new_3 / m2_safe_scale);

        let pack_idx = i / PACKING_AMOUNT;
        moment_1_codes_new[pack_idx as usize] = m1_c0 * PACK_SHIFT * PACK_SHIFT * PACK_SHIFT
            + m1_c1 * PACK_SHIFT * PACK_SHIFT
            + m1_c2 * PACK_SHIFT
            + m1_c3;
        moment_2_codes_new[pack_idx as usize] = m2_c0 * PACK_SHIFT * PACK_SHIFT * PACK_SHIFT
            + m2_c1 * PACK_SHIFT * PACK_SHIFT
            + m2_c2 * PACK_SHIFT
            + m2_c3;
    }

    // ============ AMSGrad max_v requantization ============

    if comptime!(amsgrad) {
        let v_block_absmax = plane_max(v_local_absmax);
        let v_safe_scale = if v_block_absmax > 0.0_f32 {
            v_block_absmax
        } else {
            1.0_f32.into()
        };

        if unit == 0 {
            max_moment_2_scales_new[block as usize] = v_safe_scale;
        }

        // #[unroll]
        for iter in 0..quads_per_thread {
            let element = unit * PACKING_AMOUNT + iter * PLANE_SIZE * PACKING_AMOUNT;
            let i = block_start + element;

            // Recompute v_to_use for all 4 elements.
            let m2_olds_0 = if comptime!(is_first_step) {
                0.0_f32.into()
            } else {
                blockwise::dequantize_blockwise(
                    moment_2_codes,
                    moment_2_scales,
                    i,
                    block_size,
                    Scheme::Unsigned as u32,
                )
            };
            let m2_olds_1 = if comptime!(is_first_step) {
                0.0_f32.into()
            } else {
                blockwise::dequantize_blockwise(
                    moment_2_codes,
                    moment_2_scales,
                    i + 1,
                    block_size,
                    Scheme::Unsigned as u32,
                )
            };
            let m2_olds_2 = if comptime!(is_first_step) {
                0.0_f32.into()
            } else {
                blockwise::dequantize_blockwise(
                    moment_2_codes,
                    moment_2_scales,
                    i + 2,
                    block_size,
                    Scheme::Unsigned as u32,
                )
            };
            let m2_olds_3 = if comptime!(is_first_step) {
                0.0_f32.into()
            } else {
                blockwise::dequantize_blockwise(
                    moment_2_codes,
                    moment_2_scales,
                    i + 3,
                    block_size,
                    Scheme::Unsigned as u32,
                )
            };

            let g_0 = grad[i as usize];
            let g_1 = grad[i as usize + 1];
            let g_2 = grad[i as usize + 2];
            let g_3 = grad[i as usize + 3];

            let m2_new_0 = m2_olds_0 * beta_2 + g_0 * g_0 * factor_2;
            let m2_new_1 = m2_olds_1 * beta_2 + g_1 * g_1 * factor_2;
            let m2_new_2 = m2_olds_2 * beta_2 + g_2 * g_2 * factor_2;
            let m2_new_3 = m2_olds_3 * beta_2 + g_3 * g_3 * factor_2;

            let v_0 = if comptime!(is_first_step) {
                m2_new_0
            } else {
                let v_old = blockwise::dequantize_blockwise(
                    max_moment_2_codes,
                    max_moment_2_scales,
                    i,
                    block_size,
                    Scheme::Unsigned as u32,
                );
                max(v_old, m2_new_0)
            };
            let v_1 = if comptime!(is_first_step) {
                m2_new_1
            } else {
                let v_old = blockwise::dequantize_blockwise(
                    max_moment_2_codes,
                    max_moment_2_scales,
                    i + 1,
                    block_size,
                    Scheme::Unsigned as u32,
                );
                max(v_old, m2_new_1)
            };
            let v_2 = if comptime!(is_first_step) {
                m2_new_2
            } else {
                let v_old = blockwise::dequantize_blockwise(
                    max_moment_2_codes,
                    max_moment_2_scales,
                    i + 2,
                    block_size,
                    Scheme::Unsigned as u32,
                );
                max(v_old, m2_new_2)
            };
            let v_3 = if comptime!(is_first_step) {
                m2_new_3
            } else {
                let v_old = blockwise::dequantize_blockwise(
                    max_moment_2_codes,
                    max_moment_2_scales,
                    i + 3,
                    block_size,
                    Scheme::Unsigned as u32,
                );
                max(v_old, m2_new_3)
            };

            let c0 = unsigned::encode(v_0 / v_safe_scale);
            let c1 = unsigned::encode(v_1 / v_safe_scale);
            let c2 = unsigned::encode(v_2 / v_safe_scale);
            let c3 = unsigned::encode(v_3 / v_safe_scale);

            let pack_idx = i / PACKING_AMOUNT;
            max_moment_2_codes_new[pack_idx as usize] = c0 * PACK_SHIFT * PACK_SHIFT * PACK_SHIFT
                + c1 * PACK_SHIFT * PACK_SHIFT
                + c2 * PACK_SHIFT
                + c3;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::launch::PLANE_SIZE;

    use super::*;
    use cubecl::bytes::Bytes;
    use cubecl::cuda::CudaRuntime;
    use cubecl::prelude::*;

    type TestRuntime = CudaRuntime;
    const BLOCK_SIZE: u32 = 256;
    const N: usize = BLOCK_SIZE as usize;

    /// Test-only kernel that runs `transform` and exposes its outputs.
    #[cube(launch_unchecked)]
    fn transform_test_kernel(
        grad: &Array<f32>,
        moment_1_codes: &Array<u32>,
        moment_1_scales: &Array<f32>,
        moment_2_codes: &Array<u32>,
        moment_2_scales: &Array<f32>,
        max_moment_2_codes: &Array<u32>,
        max_moment_2_scales: &Array<f32>,

        moment_1_codes_new: &mut Array<u32>,
        moment_1_scales_new: &mut Array<f32>,
        moment_2_codes_new: &mut Array<u32>,
        moment_2_scales_new: &mut Array<f32>,
        max_moment_2_codes_new: &mut Array<u32>,
        max_moment_2_scales_new: &mut Array<f32>,

        update_delta: &mut Array<f32>,
        m1_dequantized: &mut Array<f32>,

        beta_1: f32,
        beta_2: f32,
        factor_1: f32,
        factor_2: f32,
        correction1: f32,
        correction2_sqrt: f32,
        epsilon: f32,

        #[comptime] block_size: u32,
        #[comptime] amsgrad: bool,
        #[comptime] is_first_step: bool,
    ) {
        let block = CUBE_POS_X;
        transform(
            block,
            grad,
            moment_1_codes,
            moment_1_scales,
            moment_2_codes,
            moment_2_scales,
            max_moment_2_codes,
            max_moment_2_scales,
            moment_1_codes_new,
            moment_1_scales_new,
            moment_2_codes_new,
            moment_2_scales_new,
            max_moment_2_codes_new,
            max_moment_2_scales_new,
            update_delta,
            m1_dequantized,
            beta_1,
            beta_2,
            factor_1,
            factor_2,
            correction1,
            correction2_sqrt,
            epsilon,
            BLOCK_SIZE,
            amsgrad,
            is_first_step,
        );
    }

    /// Outputs from a transform call.
    struct TransformResult {
        delta: Vec<f32>,
        m1_dequantized: Vec<f32>,
        m1_codes: Vec<u32>,
        m1_scales: Vec<f32>,
        m2_codes: Vec<u32>,
        m2_scales: Vec<f32>,
        max_v_codes: Vec<u32>,
        max_v_scales: Vec<f32>,
    }

    /// Run `transform` once, returning all outputs. Inputs that aren't
    /// applicable (e.g. moment_1 codes when first_step) can be passed as
    /// any data — the kernel won't read them.
    fn run_transform(
        grad: &[f32],
        m1_codes_in: Option<&[u32]>,
        m1_scales_in: Option<&[f32]>,
        m2_codes_in: Option<&[u32]>,
        m2_scales_in: Option<&[f32]>,
        max_v_codes_in: Option<&[u32]>,
        max_v_scales_in: Option<&[f32]>,
        beta_1: f32,
        beta_2: f32,
        time: u32,
        epsilon: f32,
        amsgrad: bool,
        is_first_step: bool,
    ) -> TransformResult {
        let client = TestRuntime::client(&Default::default());
        let n = grad.len();
        assert_eq!(
            n % BLOCK_SIZE as usize,
            0,
            "grad length ({}) must be a multiple of BLOCK_SIZE ({})",
            n,
            BLOCK_SIZE,
        );
        let num_blocks = n as u32 / BLOCK_SIZE;
        let packed_count = n / PACKING_AMOUNT as usize;

        // Host-side scalar precomputation
        let factor_1 = 1.0 - beta_1;
        let factor_2 = 1.0 - beta_2;
        let correction1 = 1.0 - beta_1.powi(time as i32);
        let correction2_sqrt = (1.0 - beta_2.powi(time as i32)).sqrt();

        // Upload all inputs
        let upload_f32 =
            |data: &[f32]| client.create(Bytes::from_bytes_vec(f32::as_bytes(data).to_vec()));
        let upload_u32 =
            |data: &[u32]| client.create(Bytes::from_bytes_vec(u32::as_bytes(data).to_vec()));

        let grad_h = upload_f32(grad);

        // Use 1-element dummies for unused inputs (matches launcher convention)
        let dummy_u32 = client.empty(core::mem::size_of::<u32>());
        let dummy_f32 = client.empty(core::mem::size_of::<f32>());

        let m1_codes_h = m1_codes_in
            .map(upload_u32)
            .unwrap_or_else(|| dummy_u32.clone());
        let m1_scales_h = m1_scales_in
            .map(upload_f32)
            .unwrap_or_else(|| dummy_f32.clone());
        let m2_codes_h = m2_codes_in
            .map(upload_u32)
            .unwrap_or_else(|| dummy_u32.clone());
        let m2_scales_h = m2_scales_in
            .map(upload_f32)
            .unwrap_or_else(|| dummy_f32.clone());
        let max_v_codes_h = max_v_codes_in
            .map(upload_u32)
            .unwrap_or_else(|| dummy_u32.clone());
        let max_v_scales_h = max_v_scales_in
            .map(upload_f32)
            .unwrap_or_else(|| dummy_f32.clone());

        let m1_codes_in_size = if is_first_step { 1 } else { packed_count };
        let m1_scales_in_size = if is_first_step {
            1
        } else {
            num_blocks as usize
        };
        let m2_codes_in_size = if is_first_step { 1 } else { packed_count };
        let m2_scales_in_size = if is_first_step {
            1
        } else {
            num_blocks as usize
        };
        let max_v_codes_in_size = if amsgrad && !is_first_step {
            packed_count
        } else {
            1
        };
        let max_v_scales_in_size = if amsgrad && !is_first_step {
            num_blocks as usize
        } else {
            1
        };

        // Allocate outputs
        let m1_codes_new_h = client.empty(packed_count * core::mem::size_of::<u32>());
        let m1_scales_new_h = client.empty(num_blocks as usize * core::mem::size_of::<f32>());
        let m2_codes_new_h = client.empty(packed_count * core::mem::size_of::<u32>());
        let m2_scales_new_h = client.empty(num_blocks as usize * core::mem::size_of::<f32>());
        let max_v_codes_new_h = if amsgrad {
            client.empty(packed_count * core::mem::size_of::<u32>())
        } else {
            client.empty(core::mem::size_of::<u32>())
        };
        let max_v_scales_new_h = if amsgrad {
            client.empty(num_blocks as usize * core::mem::size_of::<f32>())
        } else {
            client.empty(core::mem::size_of::<f32>())
        };
        let max_v_codes_out_size = if amsgrad { packed_count } else { 1 };
        let max_v_scales_out_size = if amsgrad { num_blocks as usize } else { 1 };

        let delta_h = client.empty(n * core::mem::size_of::<f32>());
        let m1_dequant_h = client.empty(n * core::mem::size_of::<f32>());

        unsafe {
            transform_test_kernel::launch_unchecked::<TestRuntime>(
                &client,
                CubeCount::Static(num_blocks, 1, 1),
                CubeDim::new(&client, PLANE_SIZE as usize),
                ArrayArg::from_raw_parts(grad_h, n),
                ArrayArg::from_raw_parts(m1_codes_h, m1_codes_in_size),
                ArrayArg::from_raw_parts(m1_scales_h, m1_scales_in_size),
                ArrayArg::from_raw_parts(m2_codes_h, m2_codes_in_size),
                ArrayArg::from_raw_parts(m2_scales_h, m2_scales_in_size),
                ArrayArg::from_raw_parts(max_v_codes_h, max_v_codes_in_size),
                ArrayArg::from_raw_parts(max_v_scales_h, max_v_scales_in_size),
                ArrayArg::from_raw_parts(m1_codes_new_h.clone(), packed_count),
                ArrayArg::from_raw_parts(m1_scales_new_h.clone(), num_blocks as usize),
                ArrayArg::from_raw_parts(m2_codes_new_h.clone(), packed_count),
                ArrayArg::from_raw_parts(m2_scales_new_h.clone(), num_blocks as usize),
                ArrayArg::from_raw_parts(max_v_codes_new_h.clone(), max_v_codes_out_size),
                ArrayArg::from_raw_parts(max_v_scales_new_h.clone(), max_v_scales_out_size),
                ArrayArg::from_raw_parts(delta_h.clone(), n),
                ArrayArg::from_raw_parts(m1_dequant_h.clone(), n),
                beta_1,
                beta_2,
                factor_1,
                factor_2,
                correction1,
                correction2_sqrt,
                epsilon,
                BLOCK_SIZE,
                amsgrad,
                is_first_step,
            );
        }

        let read_f32 = |h| f32::from_bytes(&client.read_one_unchecked(h)).to_vec();
        let read_u32 = |h| u32::from_bytes(&client.read_one_unchecked(h)).to_vec();

        TransformResult {
            delta: read_f32(delta_h),
            m1_dequantized: read_f32(m1_dequant_h),
            m1_codes: read_u32(m1_codes_new_h),
            m1_scales: read_f32(m1_scales_new_h),
            m2_codes: read_u32(m2_codes_new_h),
            m2_scales: read_f32(m2_scales_new_h),
            max_v_codes: read_u32(max_v_codes_new_h),
            max_v_scales: read_f32(max_v_scales_new_h),
        }
    }

    // =====================================================================
    //  Tests — all use N (a single block of size BLOCK_SIZE)
    // =====================================================================

    /// First step with constant gradient: delta should be ~1.0 everywhere.
    /// With grad = 0.01, β₁ = 0.9, β₂ = 0.999, ε = 1e-8:
    ///   m1_new = 0.001
    ///   m2_new = 1e-7
    ///   step_size = sqrt(1 - 0.999) / (1 - 0.9) ≈ 0.316
    ///   delta = m1_new / sqrt(m2_new) * step_size ≈ 1.0
    #[test]
    fn first_step_constant_gradient() {
        let grad = vec![0.01_f32; N];

        let result = run_transform(
            &grad, None, None, None, None, None, None, 0.9, 0.999, 1, 1e-8, false, true,
        );

        for (i, &d) in result.delta.iter().enumerate() {
            assert!(d.is_finite(), "delta[{}] is not finite: {}", i, d);
            assert!(
                (d - 1.0).abs() < 0.01,
                "delta[{}] = {}, expected ~1.0",
                i,
                d,
            );
        }

        for (i, &m1) in result.m1_dequantized.iter().enumerate() {
            assert!(
                (m1 - 0.001).abs() < 1e-5,
                "m1_dequantized[{}] = {}, expected 0.001",
                i,
                m1,
            );
        }
    }

    /// First step with zero gradient: deltas and moments should all be 0.
    #[test]
    fn first_step_zero_gradient() {
        let grad = vec![0.0_f32; N];

        let result = run_transform(
            &grad, None, None, None, None, None, None, 0.9, 0.999, 1, 1e-8, false, true,
        );

        for &d in &result.delta {
            assert_eq!(d, 0.0, "expected delta = 0 for zero gradient, got {}", d);
        }

        for &m1 in &result.m1_dequantized {
            assert_eq!(m1, 0.0);
        }
    }

    /// First step with mixed-sign gradient: each delta should match grad's sign.
    #[test]
    fn first_step_sign_preservation() {
        let grad: Vec<f32> = (0..N)
            .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();

        let result = run_transform(
            &grad, None, None, None, None, None, None, 0.9, 0.999, 1, 1e-8, false, true,
        );

        for (i, (&g, &d)) in grad.iter().zip(result.delta.iter()).enumerate() {
            assert!(d.is_finite(), "delta[{}] = {}", i, d);
            assert!(
                (g > 0.0) == (d > 0.0),
                "sign mismatch at {}: grad={}, delta={}",
                i,
                g,
                d,
            );
        }
    }

    /// Two-step parity: step 0 produces state, step 1 reads it back.
    /// Verifies the requantize-then-dequantize roundtrip works.
    #[test]
    fn two_step_state_propagation() {
        let grad0 = vec![0.05_f32; N];
        let grad1 = vec![0.05_f32; N];

        let r0 = run_transform(
            &grad0, None, None, None, None, None, None, 0.9, 0.999, 1, 1e-8, false, true,
        );

        for &d in &r0.delta {
            assert!(d.is_finite(), "step 0 delta has non-finite: {}", d);
        }

        let r1 = run_transform(
            &grad1,
            Some(&r0.m1_codes),
            Some(&r0.m1_scales),
            Some(&r0.m2_codes),
            Some(&r0.m2_scales),
            None,
            None,
            0.9,
            0.999,
            2,
            1e-8,
            false,
            false,
        );

        for &d in &r1.delta {
            assert!(d.is_finite(), "step 1 delta has non-finite: {}", d);
        }

        for (i, &d) in r1.delta.iter().enumerate() {
            assert!(d > 0.0, "step 1 delta[{}] = {}, expected positive", i, d);
        }
    }

    /// Edge case: zero gradient on a non-first step should still produce
    /// finite output (using stored moments from previous step).
    #[test]
    fn second_step_zero_gradient_after_nonzero() {
        let grad0 = vec![0.05_f32; N];
        let grad1 = vec![0.0_f32; N];

        let r0 = run_transform(
            &grad0, None, None, None, None, None, None, 0.9, 0.999, 1, 1e-8, false, true,
        );

        let r1 = run_transform(
            &grad1,
            Some(&r0.m1_codes),
            Some(&r0.m1_scales),
            Some(&r0.m2_codes),
            Some(&r0.m2_scales),
            None,
            None,
            0.9,
            0.999,
            2,
            1e-8,
            false,
            false,
        );

        for &d in &r1.delta {
            assert!(d.is_finite(), "step 1 delta should be finite, got {}", d);
        }
    }
}
