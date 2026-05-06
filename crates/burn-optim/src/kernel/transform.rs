//! Momentum transformation for AdamW 8bit.

use cubecl::prelude::*;

use crate::kernel::blockwise::{self, Scheme};
use crate::kernel::signed;
use crate::kernel::unsigned;
use crate::launch::{PACK_SHIFT, PACKING_AMOUNT};

#[cube]
pub fn transform(
    block: u32,
    grad: &Array<f32>,

    // Incoming state (codes + per-block scales).
    moment_1_codes: &Array<u32>,
    moment_1_scales: &Array<f32>,
    moment_2_codes: &Array<u32>,
    moment_2_scales: &Array<f32>,
    max_moment_2_codes: &Array<u32>,  // unused if !amsgrad
    max_moment_2_scales: &Array<f32>, // unused if !amsgrad

    // Outgoing state.
    moment_1_codes_new: &mut Array<u32>,
    moment_1_scales_new: &mut Array<f32>,
    moment_2_codes_new: &mut Array<u32>,
    moment_2_scales_new: &mut Array<f32>,
    max_moment_2_codes_new: &mut Array<u32>,
    max_moment_2_scales_new: &mut Array<f32>,

    // Outputs consumed by step().
    update_delta: &mut Array<f32>,
    m1_dequantized: &mut Array<f32>,

    // Runtime scalars.
    beta_1: f32,
    beta_2: f32,
    factor_1: f32,         // 1 - beta_1
    factor_2: f32,         // 1 - beta_2
    correction1: f32,      // 1 - beta_1^t
    correction2_sqrt: f32, // sqrt(1 - beta_2^t)
    epsilon: f32,

    #[comptime] block_size: u32,
    #[comptime] amsgrad: bool,
    #[comptime] is_first_step: bool,
) {
    comptime! {
        println!("transform: block_size={} amsgrad={} is_first_step={}",
                 block_size, amsgrad, is_first_step);
    }

    let unit = UNIT_POS_X;
    let i = block * block_size + unit;

    // ---- Step 1: dequantize old moments (or use 0 on first step) ----
    let m1_old = if comptime!(is_first_step) {
        0.0f32.into()
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
        0.0f32.into()
    } else {
        blockwise::dequantize_blockwise(
            moment_2_codes,
            moment_2_scales,
            i,
            block_size,
            Scheme::Unsigned as u32,
        )
    };

    // ---- Step 2: load grad and update moments ----
    let g = grad[i as usize];
    let m1_new = m1_old * beta_1 + g * factor_1;
    let m2_new = m2_old * beta_2 + g * g * factor_2;

    // No branches, all comptime.
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

    // ---- Step 4: compute delta ----
    let step_size = correction2_sqrt / correction1;
    let delta = m1_new / ((v_to_use).sqrt() + epsilon * correction2_sqrt) * step_size;

    // Outputs that don't need the new absmax.
    update_delta[i as usize] = delta;
    m1_dequantized[i as usize] = m1_new;

    // ---- Step 5: per-block absmax for the new moments ----
    let m1_absmax = plane_max(m1_new.abs());
    let m2_absmax = plane_max(m2_new.abs());
    let m1_safe_scale = if m1_absmax > 0.0f32 {
        m1_absmax
    } else {
        1.0f32.into()
    };
    let m2_safe_scale = if m2_absmax > 0.0f32 {
        m2_absmax
    } else {
        1.0f32.into()
    };

    // ---- Step 6: requantize and write codes (paired packing) ----
    // Only even-indexed units write packed entries — they pair with their
    // odd neighbor. The neighbor's m_new value isn't in this thread's
    // registers, so we re-derive it the same way: dequant old → update.
    if unit % PACKING_AMOUNT == 0 {
        // Recompute the neighbor's m1_new / m2_new.
        let neighbor_i = i + 1;

        let neighbor_m1_old = if comptime!(is_first_step) {
            0.0f32.into()
        } else {
            blockwise::dequantize_blockwise(
                moment_1_codes,
                moment_1_scales,
                neighbor_i,
                block_size,
                Scheme::Signed as u32,
            )
        };
        let neighbor_m2_old = if comptime!(is_first_step) {
            0.0f32.into()
        } else {
            blockwise::dequantize_blockwise(
                moment_2_codes,
                moment_2_scales,
                neighbor_i,
                block_size,
                Scheme::Unsigned as u32,
            )
        };
        let neighbor_g = grad[neighbor_i as usize];
        let neighbor_m1_new = neighbor_m1_old * beta_1 + neighbor_g * factor_1;
        let neighbor_m2_new = neighbor_m2_old * beta_2 + neighbor_g * neighbor_g * factor_2;

        let m1_code = signed::encode(m1_new / m1_safe_scale);
        let m1_code_nb = signed::encode(neighbor_m1_new / m1_safe_scale);
        let m2_code = unsigned::encode(m2_new / m2_safe_scale);
        let m2_code_nb = unsigned::encode(neighbor_m2_new / m2_safe_scale);

        let pack_idx = i / PACKING_AMOUNT;
        moment_1_codes_new[pack_idx as usize] = m1_code * PACK_SHIFT + m1_code_nb;
        moment_2_codes_new[pack_idx as usize] = m2_code * PACK_SHIFT + m2_code_nb;
    }

    if unit == 0 {
        moment_1_scales_new[block as usize] = m1_safe_scale;
        moment_2_scales_new[block as usize] = m2_safe_scale;
    }

    // ---- Step 7: AMSGrad max_v requantization (comptime branch) ----
    if comptime!(amsgrad) {
        let max_v_absmax = plane_max(v_to_use.abs());
        let max_v_safe_scale = if max_v_absmax > 0.0f32 {
            max_v_absmax
        } else {
            1.0f32.into()
        };

        if unit % PACKING_AMOUNT == 0 {
            // Recompute neighbor's v_to_use the same way as above.
            let neighbor_i = i + 1;

            let neighbor_m2_old = if comptime!(is_first_step) {
                0.0f32.into()
            } else {
                blockwise::dequantize_blockwise(
                    moment_2_codes,
                    moment_2_scales,
                    neighbor_i,
                    block_size,
                    Scheme::Unsigned as u32,
                )
            };
            let neighbor_g = grad[neighbor_i as usize];
            let neighbor_m2_new = neighbor_m2_old * beta_2 + neighbor_g * neighbor_g * factor_2;

            let neighbor_v_to_use = if comptime!(is_first_step) {
                neighbor_m2_new
            } else {
                let neighbor_max_v_old = blockwise::dequantize_blockwise(
                    max_moment_2_codes,
                    max_moment_2_scales,
                    neighbor_i,
                    block_size,
                    Scheme::Unsigned as u32,
                );
                max(neighbor_max_v_old, neighbor_m2_new)
            };

            let max_v_code = unsigned::encode(v_to_use / max_v_safe_scale);
            let max_v_code_nb = unsigned::encode(neighbor_v_to_use / max_v_safe_scale);

            let pack_idx = i / PACKING_AMOUNT;
            max_moment_2_codes_new[pack_idx as usize] = max_v_code * PACK_SHIFT + max_v_code_nb;
        }

        if unit == 0 {
            max_moment_2_scales_new[block as usize] = max_v_safe_scale;
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
    const BLOCK_SIZE: u32 = 32;

    /// Test-only kernel that runs `transform` and exposes its outputs.
    /// One cube per block, plane-size units per cube.
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
        block_size: u32,
        amsgrad: bool,
        is_first_step: bool,
    ) -> TransformResult {
        let client = TestRuntime::client(&Default::default());
        let n = grad.len();
        assert_eq!(n % BLOCK_SIZE as usize, 0);
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

        // Read back
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
    //  Tests
    // =====================================================================

    /// Sanity check: first step with constant gradient produces expected
    /// moments and delta. With grad = 0.01, m1_old = m2_old = 0,
    /// β₁ = 0.9, β₂ = 0.999, ε = 1e-8:
    ///   m1_new = (1 - 0.9) * 0.01 = 0.001
    ///   m2_new = (1 - 0.999) * 0.0001 = 1e-7
    ///   correction1 = 1 - 0.9 = 0.1
    ///   correction2_sqrt = sqrt(1 - 0.999) ≈ 0.0316
    ///   step_size = 0.0316 / 0.1 = 0.316
    ///   delta = 0.001 / (sqrt(1e-7) + 1e-8 * 0.0316) * 0.316
    ///         ≈ 0.001 / 0.000316 * 0.316
    ///         ≈ 1.0
    #[test]
    fn first_step_constant_gradient() {
        let n = 64;
        let grad = vec![0.01_f32; n];

        let result = run_transform(
            &grad, None, None, None, None, None, None, 0.9, 0.999, 1, 1e-8, BLOCK_SIZE, false, true,
        );

        // Expected delta ≈ 1.0 for every element
        for (i, &d) in result.delta.iter().enumerate() {
            assert!(d.is_finite(), "delta[{}] is not finite: {}", i, d);
            assert!(
                (d - 1.0).abs() < 0.01,
                "delta[{}] = {}, expected ~1.0",
                i,
                d,
            );
        }

        // m1_dequantized should equal m1_new = 0.001 for every element
        for (i, &m1) in result.m1_dequantized.iter().enumerate() {
            assert!(
                (m1 - 0.001).abs() < 1e-5,
                "m1_dequantized[{}] = {}, expected 0.001",
                i,
                m1,
            );
        }
    }

    /// First step with zero gradient produces zero moments and zero delta.
    #[test]
    fn first_step_zero_gradient() {
        let n = 64;
        let grad = vec![0.0_f32; n];

        let result = run_transform(
            &grad, None, None, None, None, None, None, 0.9, 0.999, 1, 1e-8, BLOCK_SIZE, false, true,
        );

        for &d in &result.delta {
            assert!(
                d.is_finite() || d == 0.0,
                "delta should be 0 or finite, got {}",
                d
            );
            assert_eq!(d, 0.0, "expected delta = 0 for zero gradient, got {}", d);
        }

        for &m1 in &result.m1_dequantized {
            assert_eq!(m1, 0.0);
        }
    }

    /// First step with mixed-sign gradient. Each element's delta should
    /// have the same sign as its gradient.
    #[test]
    fn first_step_sign_preservation() {
        let n = 64;
        let grad: Vec<f32> = (0..n)
            .map(|i| if i % 2 == 0 { 0.1 } else { -0.1 })
            .collect();

        let result = run_transform(
            &grad, None, None, None, None, None, None, 0.9, 0.999, 1, 1e-8, BLOCK_SIZE, false, true,
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
        let n = 64;
        let grad0 = vec![0.05_f32; n];
        let grad1 = vec![0.05_f32; n];

        // Step 0: first step
        let r0 = run_transform(
            &grad0, None, None, None, None, None, None, 0.9, 0.999, 1, 1e-8, BLOCK_SIZE, false,
            true,
        );

        // Sanity check step 0 finished cleanly
        for &d in &r0.delta {
            assert!(d.is_finite(), "step 0 delta has non-finite: {}", d);
        }

        // Step 1: feed step 0's state back in
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
            BLOCK_SIZE,
            false,
            false,
        );

        for &d in &r1.delta {
            assert!(d.is_finite(), "step 1 delta has non-finite: {}", d);
        }

        // After two steps with identical gradients, deltas should be
        // close to but not identical to step 0.
        // (Specifically, the bias correction changes: c1 goes from 0.1 to 0.19,
        // c2_sqrt goes from 0.0316 to 0.0447, step_size goes from 0.316 to 0.235.)
        for (i, &d) in r1.delta.iter().enumerate() {
            // We expect delta to be smaller in step 1 than step 0 (bias correction effect).
            assert!(d > 0.0, "step 1 delta[{}] = {}, expected positive", i, d);
        }
    }

    /// Edge case: zero gradient on a non-first step should still produce
    /// finite output (using stored moments from previous step).
    #[test]
    fn second_step_zero_gradient_after_nonzero() {
        let n = 64;
        let grad0 = vec![0.05_f32; n];
        let grad1 = vec![0.0_f32; n];

        let r0 = run_transform(
            &grad0, None, None, None, None, None, None, 0.9, 0.999, 1, 1e-8, BLOCK_SIZE, false,
            true,
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
            BLOCK_SIZE,
            false,
            false,
        );

        for &d in &r1.delta {
            assert!(d.is_finite(), "step 1 delta should be finite, got {}", d);
        }
    }
}
