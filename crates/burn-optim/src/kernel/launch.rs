// pub struct TransformOutputs<R: CubeRuntime> {
//     pub delta: CubeTensor<R>,
//     pub m1_dequant: CubeTensor<R>,
//     pub m1_codes: CubeTensor<R>,
//     pub m1_absmax: CubeTensor<R>,
//     pub m2_codes: CubeTensor<R>,
//     pub m2_absmax: CubeTensor<R>,
//     pub max_v_codes: Option<CubeTensor<R>>, // None if !amsgrad
//     pub max_v_absmax: Option<CubeTensor<R>>,
// }

// /// Inputs to the launch wrapper. Grouped to keep the call site readable.
// pub struct TransformInputs<'a, R: CubeRuntime> {
//     pub grad: &'a CubeTensor<R>,
//     pub m1_codes: Option<&'a CubeTensor<R>>, // None on first step
//     pub m1_absmax: Option<&'a CubeTensor<R>>,
//     pub m2_codes: Option<&'a CubeTensor<R>>,
//     pub m2_absmax: Option<&'a CubeTensor<R>>,
//     pub max_v_codes: Option<&'a CubeTensor<R>>, // None if !amsgrad or first step
//     pub max_v_absmax: Option<&'a CubeTensor<R>>,
// }

// /// Hyperparameters for one transform invocation.
// pub struct TransformParams {
//     pub beta_1: f32,
//     pub beta_2: f32,
//     pub epsilon: f32,
//     pub bias_correction_1: f32,
//     pub bias_correction_2_sqrt: f32,
//     pub block_size: u32,
//     pub amsgrad: bool,
// }

// /// Allocate output buffers, compute launch geometry, and invoke the kernel.
// /// Returns raw CubeCL handles; Burn-tensor wrapping happens in Layer 3.
// pub fn launch_adamw_8bit_transform<R: CubeRuntime>(
//     client: &ComputeClient<R::Server, R::Channel>,
//     device: &R::Device,
//     inputs: TransformInputs<'_, R>,
//     params: TransformParams,
// ) -> TransformOutputs<R>;

// /// Pick the SIMD line size for f32 tensors of the given total element count.
// /// Returns 4, 2, or 1 depending on alignment.
// fn pick_line_size(numel: usize) -> u8;
