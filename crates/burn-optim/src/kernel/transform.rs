//! Momentum transformation for AdamW 8bit.

use cubecl::prelude::*;

use crate::kernel::blockwise::{self, Scheme};
use crate::launch::PACKING_AMOUNT;

#[cube]
pub fn transform(
    block: u32,
    grad: &Array<f32>,

    moment_1_codes: &mut Array<u32>,
    moment_1_scales: &mut Array<f32>,
    moment_2_codes: &mut Array<u32>,
    moment_2_scales: &mut Array<f32>,
    max_moment_2_codes: &mut Array<u32>,
    max_moment_2_scales: &mut Array<f32>,

    local_delta: &mut Array<f32>,
    local_m1: &mut Array<f32>,

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
    #[comptime] plane_size: u32,
) {
    let unit = UNIT_POS_X;
    let block_start = block * block_size;
    let elements_per_thread = comptime!(block_size / plane_size);

    // Per-lane register storage for new moment values.
    // Sized to elements_per_thread; each lane owns its elements across iters.
    let mut local_m1_new = Array::<f32>::new(elements_per_thread as usize);
    let mut local_m2_new = Array::<f32>::new(elements_per_thread as usize);
    let mut local_v_to_use = Array::<f32>::new(elements_per_thread as usize);

    let mut m1_local_absmax = 0.0_f32;
    let mut m2_local_absmax = 0.0_f32;
    let mut v_local_absmax = 0.0_f32;

    let step_size = correction2_sqrt / correction1;

    // No global scratch writes. All reads of old state happen here.
    #[unroll]
    for iter in 0..elements_per_thread {
        let element = unit + iter * plane_size;
        let i = block_start + element;

        let iter = iter as usize;

        let (m1_old, m2_old) = if comptime!(is_first_step) {
            (0.0_f32.into(), 0.0_f32.into())
        } else {
            (
                blockwise::dequantize_blockwise(
                    moment_1_codes,
                    moment_1_scales,
                    i,
                    block_size,
                    Scheme::Signed as u32,
                ),
                blockwise::dequantize_blockwise(
                    moment_2_codes,
                    moment_2_scales,
                    i,
                    block_size,
                    Scheme::Unsigned as u32,
                ),
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

        // Outputs to outer kernel (registers)
        local_delta[iter] = delta;
        local_m1[iter] = m1_new;

        // Stash for Pass 2 (registers)
        local_m1_new[iter] = m1_new;
        local_m2_new[iter] = m2_new;
        if comptime!(amsgrad) {
            local_v_to_use[iter] = v_to_use;
        }

        m1_local_absmax = max(m1_local_absmax, m1_new.abs());
        m2_local_absmax = max(m2_local_absmax, m2_new.abs());
        if comptime!(amsgrad) {
            v_local_absmax = max(v_local_absmax, v_to_use.abs());
        }
    }

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

    // Write scales in-place. Safe now because Pass 2 won't re-read them.
    if unit == 0 {
        moment_1_scales[block as usize] = m1_safe_scale;
        moment_2_scales[block as usize] = m2_safe_scale;
    }

    let mut shared_m1 = SharedMemory::<f32>::new(block_size as usize);
    let mut shared_m2 = SharedMemory::<f32>::new(block_size as usize);
    #[unroll]
    for iter in 0..elements_per_thread {
        let element = unit + iter * plane_size;
        shared_m1[element as usize] = local_m1_new[iter as usize];
        shared_m2[element as usize] = local_m2_new[iter as usize];
    }

    sync_cube();

    let quads_per_thread = comptime!(block_size / (plane_size * PACKING_AMOUNT));
    #[unroll]
    for iter in 0..quads_per_thread {
        let element = unit * PACKING_AMOUNT + iter * plane_size * PACKING_AMOUNT;
        let i = block_start + element;

        blockwise::quantize_blockwise(
            &shared_m1,
            moment_1_codes,
            moment_1_scales,
            i,
            element,
            block_size,
            Scheme::Signed as u32,
        );
        blockwise::quantize_blockwise(
            &shared_m2,
            moment_2_codes,
            moment_2_scales,
            i,
            element,
            block_size,
            Scheme::Unsigned as u32,
        );
    }

    if comptime!(amsgrad) {
        let v_block_absmax = plane_max(v_local_absmax);
        let v_safe_scale = if v_block_absmax > 0.0_f32 {
            v_block_absmax
        } else {
            1.0_f32.into()
        };

        if unit == 0 {
            max_moment_2_scales[block as usize] = v_safe_scale;
        }

        sync_cube();
        #[unroll]
        for iter in 0..elements_per_thread {
            let element = unit + iter * plane_size;
            shared_m1[element as usize] = local_v_to_use[iter as usize];
        }
        sync_cube();

        #[unroll]
        for iter in 0..quads_per_thread {
            let element = unit * PACKING_AMOUNT + iter * plane_size * PACKING_AMOUNT;
            let i = block_start + element;

            blockwise::quantize_blockwise(
                &shared_m1,
                max_moment_2_codes,
                max_moment_2_scales,
                i,
                element,
                block_size,
                Scheme::Unsigned as u32,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::bytes::Bytes;
    use cubecl::cuda::CudaRuntime;
    use cubecl::hip::HipRuntime;
    use cubecl::prelude::*;
    use cubecl::wgpu::WgpuRuntime;

    // type TestRuntime = CudaRuntime;
    type TestRuntime = WgpuRuntime;
    // type TestRuntime = HipRuntime;
    const BLOCK_SIZE: u32 = 256;
    const N: usize = BLOCK_SIZE as usize;

    /// Test-only kernel that runs `transform` and spills its per-lane
    /// register outputs (local_delta, local_m1) to global memory so the
    /// host can read them.
    #[cube(launch_unchecked)]
    fn transform_test_kernel(
        grad: &Array<f32>,
        moment_1_codes: &mut Array<u32>,
        moment_1_scales: &mut Array<f32>,
        moment_2_codes: &mut Array<u32>,
        moment_2_scales: &mut Array<f32>,
        max_moment_2_codes: &mut Array<u32>,
        max_moment_2_scales: &mut Array<f32>,

        // Test-only spill buffers for the in-register outputs
        delta_spill: &mut Array<f32>,
        m1_spill: &mut Array<f32>,

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
        #[comptime] plane_size: u32,
    ) {
        let block = CUBE_POS_X;
        let unit = UNIT_POS_X;
        let elements_per_thread = comptime!(block_size / plane_size);

        let mut local_delta = Array::<f32>::new(elements_per_thread as usize);
        let mut local_m1 = Array::<f32>::new(elements_per_thread as usize);

        transform(
            block,
            grad,
            moment_1_codes,
            moment_1_scales,
            moment_2_codes,
            moment_2_scales,
            max_moment_2_codes,
            max_moment_2_scales,
            &mut local_delta,
            &mut local_m1,
            beta_1,
            beta_2,
            factor_1,
            factor_2,
            correction1,
            correction2_sqrt,
            epsilon,
            block_size,
            amsgrad,
            is_first_step,
            plane_size,
        );

        // Spill registers to global memory for host inspection.
        // Pass-1 distribution: lane `unit` owns elements at offsets
        // {unit, unit + plane_size, unit + 2*plane_size, ...}
        let block_start = block * block_size;
        #[unroll]
        for iter in 0..elements_per_thread {
            let element = unit + iter * plane_size;
            let i = block_start + element;
            delta_spill[i as usize] = local_delta[iter as usize];
            m1_spill[i as usize] = local_m1[iter as usize];
        }
    }

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

        let factor_1 = 1.0 - beta_1;
        let factor_2 = 1.0 - beta_2;
        let correction1 = 1.0 - beta_1.powi(time as i32);
        let correction2_sqrt = (1.0 - beta_2.powi(time as i32)).sqrt();

        let upload_f32 =
            |data: &[f32]| client.create(Bytes::from_bytes_vec(f32::as_bytes(data).to_vec()));
        let upload_u32 =
            |data: &[u32]| client.create(Bytes::from_bytes_vec(u32::as_bytes(data).to_vec()));

        let grad_h = upload_f32(grad);

        // In-place state: always allocate full-size buffers. On first step,
        // the kernel ignores their contents. On subsequent steps, upload prior state.
        let m1_codes_h = match m1_codes_in {
            Some(d) => upload_u32(d),
            None => client.empty(packed_count * core::mem::size_of::<u32>()),
        };
        let m1_scales_h = match m1_scales_in {
            Some(d) => upload_f32(d),
            None => client.empty(num_blocks as usize * core::mem::size_of::<f32>()),
        };
        let m2_codes_h = match m2_codes_in {
            Some(d) => upload_u32(d),
            None => client.empty(packed_count * core::mem::size_of::<u32>()),
        };
        let m2_scales_h = match m2_scales_in {
            Some(d) => upload_f32(d),
            None => client.empty(num_blocks as usize * core::mem::size_of::<f32>()),
        };

        let max_v_codes_h = if amsgrad {
            match max_v_codes_in {
                Some(d) => upload_u32(d),
                None => client.empty(packed_count * core::mem::size_of::<u32>()),
            }
        } else {
            client.empty(core::mem::size_of::<u32>())
        };
        let max_v_scales_h = if amsgrad {
            match max_v_scales_in {
                Some(d) => upload_f32(d),
                None => client.empty(num_blocks as usize * core::mem::size_of::<f32>()),
            }
        } else {
            client.empty(core::mem::size_of::<f32>())
        };

        let m1_codes_size = packed_count;
        let m1_scales_size = num_blocks as usize;
        let m2_codes_size = packed_count;
        let m2_scales_size = num_blocks as usize;
        let max_v_codes_size = if amsgrad { packed_count } else { 1 };
        let max_v_scales_size = if amsgrad { num_blocks as usize } else { 1 };

        let delta_h = client.empty(n * core::mem::size_of::<f32>());
        let m1_dequant_h = client.empty(n * core::mem::size_of::<f32>());

        let plane_size = client.properties().hardware.plane_size_max;

        unsafe {
            transform_test_kernel::launch_unchecked::<TestRuntime>(
                &client,
                CubeCount::Static(num_blocks, 1, 1),
                CubeDim::new(&client, plane_size as usize),
                ArrayArg::from_raw_parts(grad_h, n),
                ArrayArg::from_raw_parts(m1_codes_h.clone(), m1_codes_size),
                ArrayArg::from_raw_parts(m1_scales_h.clone(), m1_scales_size),
                ArrayArg::from_raw_parts(m2_codes_h.clone(), m2_codes_size),
                ArrayArg::from_raw_parts(m2_scales_h.clone(), m2_scales_size),
                ArrayArg::from_raw_parts(max_v_codes_h.clone(), max_v_codes_size),
                ArrayArg::from_raw_parts(max_v_scales_h.clone(), max_v_scales_size),
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
                plane_size,
            );
        }

        let read_f32 = |h| f32::from_bytes(&client.read_one_unchecked(h)).to_vec();
        let read_u32 = |h| u32::from_bytes(&client.read_one_unchecked(h)).to_vec();

        TransformResult {
            delta: read_f32(delta_h),
            m1_dequantized: read_f32(m1_dequant_h),
            // State buffers updated in place — read back from same handles
            m1_codes: read_u32(m1_codes_h),
            m1_scales: read_f32(m1_scales_h),
            m2_codes: read_u32(m2_codes_h),
            m2_scales: read_f32(m2_scales_h),
            max_v_codes: read_u32(max_v_codes_h),
            max_v_scales: read_f32(max_v_scales_h),
        }
    }

    // =====================================================================
    //  Tests
    // =====================================================================

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
