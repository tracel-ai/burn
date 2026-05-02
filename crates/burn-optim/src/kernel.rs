mod blockwise;
mod launch;
mod signed;
mod unsigned;

use burn_core as burn;

use burn::Tensor;
use burn::prelude::*;
use cubecl::prelude::*;

// use signed;
// use unsigned;

// #[cube]
// pub trait QuantScheme: 'static + Send + Sync {
//     /// Decode one packed code to a normalized value.
//     fn decode(code: u32) -> f32;

//     /// Encode one normalized value to its nearest code.
//     fn encode(x: f32) -> u32;
// }

// /// Signed dynamic quantization (used for `moment_1`).
// pub struct SignedDynamic;

// #[cube]
// impl QuantScheme for SignedDynamic {
//     fn decode(code: u32) -> f32 {
//         signed::decode(code)
//     }
//     fn encode(x: f32) -> u32 {
//         signed::encode(x)
//     }
// }

// /// Unsigned dynamic quantization (used for `moment_2` and `max_moment_2`).
// pub struct UnsignedDynamic;

// #[cube]
// impl QuantScheme for UnsignedDynamic {
//     fn decode(code: u32) -> f32 {
//         unsigned::decode(code)
//     }
//     fn encode(x: f32) -> u32 {
//         unsigned::encode(x)
//     }
// }

// // -------------------------------------------------------------------------
// //  The fused kernel
// // -------------------------------------------------------------------------

// /// Fused 8-bit AdamW transform.
// ///
// /// One cube per quantization block. Each cube cooperatively dequantizes,
// /// updates moments, computes the delta, finds the new block absmax, and
// /// requantizes — all in registers, with `m1_new` / `m2_new` never round-
// /// tripping through HBM.
// #[cube(launch_unchecked)]
// pub fn adamw_8bit_transform_kernel<F: Float, Q1: QuantScheme, Q2: QuantScheme>(
//     // ---- Gradient ----
//     grad: &Tensor<Line<F>>,

//     // ---- Incoming state ----
//     moment_1_codes: &Tensor<Line<u32>>,
//     moment_1_absmax: &Tensor<F>,
//     moment_2_codes: &Tensor<Line<u32>>,
//     moment_2_absmax: &Tensor<F>,
//     max_moment_2_codes: &Tensor<Line<u32>>, // unused if !amsgrad
//     max_moment_2_absmax: &Tensor<F>,        // unused if !amsgrad

//     // ---- Outgoing state ----
//     moment_1_codes_new: &mut Tensor<Line<u32>>,
//     moment_1_absmax_new: &mut Tensor<F>,
//     moment_2_codes_new: &mut Tensor<Line<u32>>,
//     moment_2_absmax_new: &mut Tensor<F>,
//     max_moment_2_codes_new: &mut Tensor<Line<u32>>,
//     max_moment_2_absmax_new: &mut Tensor<F>,

//     // ---- Outputs consumed by step() ----
//     update_delta: &mut Tensor<Line<F>>,
//     m1_dequantized: &mut Tensor<Line<F>>,

//     // ---- Runtime scalars ----
//     beta_1: F,
//     beta_2: F,
//     factor_1: F,
//     factor_2: F,
//     correction1: F,
//     correction2: F,
//     epsilon: F,

//     // ---- Compile-time specialization ----
//     #[comptime] block_size: u32,
//     #[comptime] amsgrad: bool,
//     #[comptime] is_first_step: bool,
// ) {
//     let block = CUBE_POS_X;
//     let unit = UNIT_POS_X;

//     // ---- Load per-block absmax scales (broadcast within the cube) ----
//     // On first step there's no prior state to dequantize.
//     let m1_scale = if is_first_step {
//         F::new(0.0)
//     } else {
//         moment_1_absmax[block]
//     };
//     let m2_scale = if is_first_step {
//         F::new(0.0)
//     } else {
//         moment_2_absmax[block]
//     };
//     let max_scale = if amsgrad && !is_first_step {
//         max_moment_2_absmax[block]
//     } else {
//         F::new(0.0)
//     };

//     // ---- Per-element work: dequant, update, delta, stash for requant ----
//     // For now we assume CUBE_DIM_X == block_size (one element per thread).
//     // For block_size > CUBE_DIM_X you'd loop here; for block_size larger
//     // than plane_size you also need a hierarchical reduction below.
//     let i = block * block_size + unit;

//     let g = grad[i];

//     let m1_old = if is_first_step {
//         Line::empty(grad.line_size()).fill(F::new(0.0))
//     } else {
//         // TODO: line-wise decode — Q1::decode operates on a scalar.
//         // The simple version does this per-lane; vectorized decode is a
//         // perf optimization we can add once correctness is established.
//         decode::<F, Q1>(moment_1_codes[i], m1_scale)
//     };
//     let m2_old = if is_first_step {
//         Line::empty(grad.line_size()).fill(F::new(0.0))
//     } else {
//         decode::<F, Q2>(moment_2_codes[i], m2_scale)
//     };

//     // Update moments (still in registers).
//     let m1_new = m1_old * beta_1 + g * factor_1;
//     let m2_new = m2_old * beta_2 + g * g * factor_2;

//     // AMSGrad branch — comptime, fully eliminated when amsgrad=false.
//     let v_to_use = if amsgrad {
//         let max_v_old = if is_first_step {
//             m2_new
//         } else {
//             decode::<F, Q2>(max_moment_2_codes[i], max_scale)
//         };
//         line_max(max_v_old, m2_new)
//     } else {
//         m2_new
//     };

//     // Compute delta.
//     let step_size = correction2 / correction1;
//     let delta = m1_new / (line_sqrt(v_to_use) + Line::cast_from(epsilon * correction2))
//         * Line::cast_from(step_size);

//     // Write outputs that don't need the new absmax.
//     update_delta[i] = delta;
//     m1_dequantized[i] = m1_new;

//     // ---- Find new block absmaxes via cube-wide reduction ----
//     let new_m1_absmax = cube_max_abs(m1_new);
//     let new_m2_absmax = cube_max_abs(m2_new);

//     // One thread per cube writes the new absmax.
//     if unit == 0 {
//         moment_1_absmax_new[block] = new_m1_absmax;
//         moment_2_absmax_new[block] = new_m2_absmax;
//     }

//     // ---- Requantize and write codes ----
//     moment_1_codes_new[i] = encode::<F, Q1>(m1_new, new_m1_absmax);
//     moment_2_codes_new[i] = encode::<F, Q2>(m2_new, new_m2_absmax);

//     if amsgrad {
//         let new_max_absmax = cube_max_abs(v_to_use);
//         if unit == 0 {
//             max_moment_2_absmax_new[block] = new_max_absmax;
//         }
//         max_moment_2_codes_new[i] = encode::<F, Q2>(v_to_use, new_max_absmax);
//     }
// }

// /// Hyperparameters for one transform invocation. Mirrors the relevant
// /// fields of `AdaptiveMomentumW8BitFused` plus host-precomputed bias
// /// corrections.
// pub struct TransformParams {
//     pub beta_1: f32,
//     pub beta_2: f32,
//     pub factor_1: f32,
//     pub factor_2: f32,
//     pub correction1: f32,
//     pub correction2: f32,
//     pub epsilon: f32,
//     pub block_size: u32,
//     pub amsgrad: bool,
//     pub is_first_step: bool,
// }

// /// Outputs returned by the launch wrapper, already wrapped as Burn types
// /// so that `transform` can plug them straight into `AdamWState8BitFused`.
// pub struct TransformOutputs<B: Backend, const D: usize> {
//     pub update_delta: Tensor<B, D>,
//     pub m1_dequantized: Tensor<B, D>,
//     pub moment_1_new: QuantizeBlockwise<B, D>,
//     pub moment_2_new: QuantizeBlockwise<B, D>,
//     pub max_moment_2_new: Option<QuantizeBlockwise<B, D>>,
// }

// /// Allocate output buffers, compute launch geometry, and invoke the
// /// fused kernel. Returns Burn-tensor-wrapped outputs.
// pub fn launch_adamw_8bit_transform<B, const D: usize>(
//     grad: &Tensor<B, D>,
//     moment_1_in: Option<&QuantizeBlockwise<B, D>>,
//     moment_2_in: Option<&QuantizeBlockwise<B, D>>,
//     max_moment_2_in: Option<&QuantizeBlockwise<B, D>>,
//     params: TransformParams,
// ) -> TransformOutputs<B, D>
// where
//     B: CubeBackend,
// {
//     // ---- 1. Extract CubeCL handles from Burn tensors ----
//     // TODO: the exact path from `Tensor<B, D>` to `CubeTensor<R>` depends
//     // on Burn 0.20's primitive API. Sketch:
//     //
//     //   let grad_prim  = grad.clone().into_primitive().tensor();
//     //   let client     = &grad_prim.client;
//     //   let device     = &grad_prim.device;
//     //
//     // Verify against your existing kernels (e.g. dequantize_blockwise)
//     // to see how they bridge Tensor <-> CubeTensor.

//     // ---- 2. Compute launch geometry ----
//     let n = grad.shape().num_elements() as u32;
//     let block_size = params.block_size;
//     debug_assert!(
//         n % block_size == 0,
//         "tensor numel ({n}) must be divisible by block_size ({block_size})"
//     );
//     let num_blocks = n / block_size;

//     let line_size: u8 = pick_line_size(n as usize);
//     let cube_count = CubeCount::Static(num_blocks, 1, 1);
//     let cube_dim = CubeDim::new(block_size / line_size as u32, 1, 1);

//     // ---- 3. Allocate output buffers ----
//     // TODO: allocate via client.empty(...) for codes/absmax and
//     // empty_device::<R, F>(...) for the f32 outputs. Wrap as Burn tensors
//     // before returning.

//     // ---- 4. Launch ----
//     // TODO: assemble the TensorArgs and call:
//     //
//     // unsafe {
//     //     adamw_8bit_transform_kernel::launch_unchecked
//     //         ::<f32, SignedDynamic, UnsignedDynamic, R>(
//     //             client, cube_count, cube_dim,
//     //             grad.as_tensor_arg(line_size),
//     //             /* ... all the other tensor args ... */,
//     //             ScalarArg::new(params.beta_1),
//     //             /* ... other scalars ... */,
//     //             params.block_size,
//     //             params.amsgrad,
//     //             params.is_first_step,
//     //         );
//     // }

//     // ---- 5. Wrap outputs and return ----
//     unimplemented!("see numbered TODOs above")

//     // /// Choose the SIMD line size for f32 tensors of the given total element
//     // /// count. Returns the largest power-of-two divisor of `numel` that is
//     // /// <= 4 (the CubeCL upper bound for typical f32 vectorization).
// }
