//! Fused 8-bit AdamW optimizer.

use burn_core as burn;
use burn_core::tensor::TensorData;

use burn::config::Config;
use burn::module::AutodiffModule;
use burn::record::Record;
use burn::tensor::{
    DType, Int, Shape, Tensor, TensorPrimitive,
    backend::{AutodiffBackend, Backend},
    ops::Device,
};
use burn_cubecl::{
    BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement, tensor::CubeTensor,
};
use cubecl::CubeElement;
use cubecl::client::ComputeClient;
use cubecl::prelude::ArrayArg;
use cubecl::server::Handle;

use crate::kernel::fused_adamw8bit_step::fused_adamw8bit_step_kernel;
use crate::optim::{SimpleOptimizer, adaptor::OptimizerAdaptor};
use crate::{LearningRate, grad_clipping::GradientClippingConfig};

const BLOCK_SIZE: usize = 256;
const PACK_FACTOR: usize = 4;
const PACKED_PER_BLOCK: usize = BLOCK_SIZE / PACK_FACTOR;

#[derive(Config, Debug)]
pub struct AdamWConfig8BitFused {
    #[config(default = 0.9)]
    beta_1: f32,
    #[config(default = 0.999)]
    beta_2: f32,
    #[config(default = 1e-8)]
    epsilon: f32,
    #[config(default = 1e-4)]
    weight_decay: f32,
    grad_clipping: Option<GradientClippingConfig>,
}

#[derive(Clone)]
pub struct AdamW8BitFused {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    weight_decay: f32,
}

#[derive(Record, Clone)]
pub struct AdamW8BitFusedState<B: Backend, const D: usize> {
    pub time: usize,
    pub original_shape: [usize; D],
    pub moment_1_codes: Tensor<B, 1, Int>,
    pub moment_1_scales: Tensor<B, 1>,
    pub moment_2_codes: Tensor<B, 1, Int>,
    pub moment_2_scales: Tensor<B, 1>,
}

impl<R, F, I, BT> SimpleOptimizer<CubeBackend<R, F, I, BT>> for AdamW8BitFused
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    type State<const D: usize> = AdamW8BitFusedState<CubeBackend<R, F, I, BT>, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<CubeBackend<R, F, I, BT>, D>,
        grad: Tensor<CubeBackend<R, F, I, BT>, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<CubeBackend<R, F, I, BT>, D>, Option<Self::State<D>>) {
        let original_shape = tensor.shape().dims::<D>();
        let total_elements: usize = original_shape.iter().product();
        let padding = (BLOCK_SIZE - (total_elements % BLOCK_SIZE)) % BLOCK_SIZE;
        let padded_len = total_elements + padding;
        let num_blocks = padded_len / BLOCK_SIZE;

        // let grad_primitive = grad.into_primitive().tensor();
        let grad_data: TensorData = grad.to_data();
        let grad_vec: Vec<f32> = grad_data.to_vec().expect("grad to_vec failed");

        // Now consume grad to get its primitive (we no longer need it).
        let grad_primitive = grad.into_primitive().tensor();
        let client = grad_primitive.client.clone();
        let device = grad_primitive.device.clone();

        let mut padded = grad_vec;
        padded.resize(padded_len, 0.0f32);
        let grad_handle = client.create(cubecl::bytes::Bytes::from_bytes_vec(
            f32::as_bytes(&padded).to_vec(),
        ));

        let (m_codes_handle, m_scales_handle, v_codes_handle, v_scales_handle, time) = match &state
        {
            Some(s) => {
                let m_codes = s.moment_1_codes.clone().into_primitive();
                let m_scales = s.moment_1_scales.clone().into_primitive().tensor();
                let v_codes = s.moment_2_codes.clone().into_primitive();
                let v_scales = s.moment_2_scales.clone().into_primitive().tensor();
                (
                    m_codes.handle.clone(),
                    m_scales.handle.clone(),
                    v_codes.handle.clone(),
                    v_scales.handle.clone(),
                    s.time + 1,
                )
            }
            None => {
                let zero_codes_vec = vec![0u32; num_blocks * PACKED_PER_BLOCK];
                let one_scales_vec = vec![1.0f32; num_blocks];

                let m_codes = client.create(cubecl::bytes::Bytes::from_bytes_vec(
                    u32::as_bytes(&zero_codes_vec).to_vec(),
                ));
                let m_scales = client.create(cubecl::bytes::Bytes::from_bytes_vec(
                    f32::as_bytes(&one_scales_vec).to_vec(),
                ));
                let v_codes = client.create(cubecl::bytes::Bytes::from_bytes_vec(
                    u32::as_bytes(&zero_codes_vec).to_vec(),
                ));
                let v_scales = client.create(cubecl::bytes::Bytes::from_bytes_vec(
                    f32::as_bytes(&one_scales_vec).to_vec(),
                ));
                (m_codes, m_scales, v_codes, v_scales, 1)
            }
        };

        let delta_handle = client.empty(padded_len * core::mem::size_of::<f32>());
        let m_codes_out_handle =
            client.empty(num_blocks * PACKED_PER_BLOCK * core::mem::size_of::<u32>());
        let m_scales_out_handle = client.empty(num_blocks * core::mem::size_of::<f32>());
        let v_codes_out_handle =
            client.empty(num_blocks * PACKED_PER_BLOCK * core::mem::size_of::<u32>());
        let v_scales_out_handle = client.empty(num_blocks * core::mem::size_of::<f32>());

        let t = time as i32;
        let correction1 = 1.0f32 - self.beta_1.powi(t);
        let correction2_sqrt = (1.0f32 - self.beta_2.powi(t)).sqrt();
        let step_size = correction2_sqrt / correction1;
        let epsilon_eff = self.epsilon * correction2_sqrt;

        let delta_for_launch = delta_handle.clone();
        let m_codes_for_launch = m_codes_out_handle.clone();
        let m_scales_for_launch = m_scales_out_handle.clone();
        let v_codes_for_launch = v_codes_out_handle.clone();
        let v_scales_for_launch = v_scales_out_handle.clone();

        let cube_count = cubecl::CubeCount::Static(num_blocks as u32, 1, 1);
        let cube_dim = cubecl::CubeDim::new_1d(BLOCK_SIZE as u32);

        unsafe {
            fused_adamw8bit_step_kernel::launch_unchecked::<R>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(grad_handle, padded_len),
                ArrayArg::from_raw_parts(m_codes_handle, num_blocks * PACKED_PER_BLOCK),
                ArrayArg::from_raw_parts(m_scales_handle, num_blocks),
                ArrayArg::from_raw_parts(v_codes_handle, num_blocks * PACKED_PER_BLOCK),
                ArrayArg::from_raw_parts(v_scales_handle, num_blocks),
                ArrayArg::from_raw_parts(delta_for_launch, padded_len),
                ArrayArg::from_raw_parts(m_codes_for_launch, num_blocks * PACKED_PER_BLOCK),
                ArrayArg::from_raw_parts(m_scales_for_launch, num_blocks),
                ArrayArg::from_raw_parts(v_codes_for_launch, num_blocks * PACKED_PER_BLOCK),
                ArrayArg::from_raw_parts(v_scales_for_launch, num_blocks),
                self.beta_1,
                self.beta_2,
                epsilon_eff,
                step_size,
            );
        }

        let delta_tensor_padded = wrap_handle_as_tensor_1d_float::<R, F, I, BT>(
            &client,
            &device,
            delta_handle,
            padded_len,
        );
        let delta = delta_tensor_padded
            .slice([0..total_elements])
            .reshape(Shape::from(original_shape));

        let decay_rate = lr * (self.weight_decay as f64);
        let decayed_tensor = if decay_rate == 0.0 {
            tensor
        } else {
            tensor.mul_scalar(1.0 - decay_rate)
        };
        let tensor_updated = decayed_tensor - delta.mul_scalar(lr);

        let m_codes_out_tensor = wrap_handle_as_tensor_1d_int::<R, F, I, BT>(
            &client,
            &device,
            m_codes_out_handle,
            num_blocks * PACKED_PER_BLOCK,
        );
        let m_scales_out_tensor = wrap_handle_as_tensor_1d_float::<R, F, I, BT>(
            &client,
            &device,
            m_scales_out_handle,
            num_blocks,
        );
        let v_codes_out_tensor = wrap_handle_as_tensor_1d_int::<R, F, I, BT>(
            &client,
            &device,
            v_codes_out_handle,
            num_blocks * PACKED_PER_BLOCK,
        );
        let v_scales_out_tensor = wrap_handle_as_tensor_1d_float::<R, F, I, BT>(
            &client,
            &device,
            v_scales_out_handle,
            num_blocks,
        );

        let new_state = AdamW8BitFusedState {
            time,
            original_shape,
            moment_1_codes: m_codes_out_tensor,
            moment_1_scales: m_scales_out_tensor,
            moment_2_codes: v_codes_out_tensor,
            moment_2_scales: v_scales_out_tensor,
        };

        (tensor_updated, Some(new_state))
    }

    fn to_device<const D: usize>(
        mut state: Self::State<D>,
        device: &Device<CubeBackend<R, F, I, BT>>,
    ) -> Self::State<D> {
        state.moment_1_codes = state.moment_1_codes.to_device(device);
        state.moment_1_scales = state.moment_1_scales.to_device(device);
        state.moment_2_codes = state.moment_2_codes.to_device(device);
        state.moment_2_scales = state.moment_2_scales.to_device(device);
        state
    }
}

impl AdamWConfig8BitFused {
    pub fn init<R, F, I, BT, B, M>(&self) -> OptimizerAdaptor<AdamW8BitFused, M, B>
    where
        R: CubeRuntime,
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
        B: AutodiffBackend<InnerBackend = CubeBackend<R, F, I, BT>>,
        M: AutodiffModule<B>,
    {
        let optim = AdamW8BitFused {
            beta_1: self.beta_1,
            beta_2: self.beta_2,
            epsilon: self.epsilon,
            weight_decay: self.weight_decay,
        };
        let mut optim = OptimizerAdaptor::from(optim);
        if let Some(config) = &self.grad_clipping {
            optim = optim.with_grad_clipping(config.init());
        }
        optim
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// fn pad_to_block_size<R: CubeRuntime>(
//     client: &ComputeClient<R>,
//     grad: &CubeTensor<R>,
//     padded_len: usize,
// ) -> Handle {
//     let bytes = client.read_one_unchecked(grad.handle.clone());
//     let raw: &[u8] = &bytes;
//     let mut padded: Vec<f32> = f32::from_bytes(raw).to_vec();
//     padded.resize(padded_len, 0.0f32);
//     client.create(cubecl::bytes::Bytes::from_bytes_vec(
//         f32::as_bytes(&padded).to_vec(),
//     ))
// }

fn pad_to_block_size<R: CubeRuntime>(
    client: &ComputeClient<R>,
    grad: &CubeTensor<R>,
    padded_len: usize,
) -> Handle {
    let n_elements: usize = grad.meta.shape().num_elements();
    let n_bytes = n_elements * core::mem::size_of::<f32>();

    let bytes = client.read_one_unchecked(grad.handle.clone());
    let raw: &[u8] = &bytes;

    // Take exactly the bytes belonging to the logical tensor data, in
    // case the underlying handle's buffer is larger (alignment padding).
    let take = n_bytes.min(raw.len());
    let mut padded: Vec<f32> = f32::from_bytes(&raw[..take]).to_vec();

    // Defensive truncate before resize, in case from_bytes gave us extras.
    padded.truncate(n_elements);
    padded.resize(padded_len, 0.0f32);

    client.create(cubecl::bytes::Bytes::from_bytes_vec(
        f32::as_bytes(&padded).to_vec(),
    ))
}

/// Wrap a handle holding `n` fp32 elements into a Burn 1D float Tensor.
fn wrap_handle_as_tensor_1d_float<R, F, I, BT>(
    client: &ComputeClient<R>,
    device: &R::Device,
    handle: Handle,
    n: usize,
) -> Tensor<CubeBackend<R, F, I, BT>, 1>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let cube_tensor = CubeTensor::<R>::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([n]),
        handle,
        DType::F32,
    );
    Tensor::from_primitive(TensorPrimitive::Float(cube_tensor))
}

/// Wrap a handle holding `n` u32 elements into a Burn 1D int Tensor.
fn wrap_handle_as_tensor_1d_int<R, F, I, BT>(
    client: &ComputeClient<R>,
    device: &R::Device,
    handle: Handle,
    n: usize,
) -> Tensor<CubeBackend<R, F, I, BT>, 1, Int>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let cube_tensor = CubeTensor::<R>::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([n]),
        handle,
        DType::U32,
    );
    Tensor::from_primitive(cube_tensor)
}

#[cfg(test)]
#[cfg(feature = "test-cuda")]
mod tests {
    use super::*;
    use crate::{GradientsParams, Optimizer};
    use burn::module::{Module, Param};
    use burn::tensor::{Distribution, Shape, Tensor, TensorData, Tolerance, ops::FloatElem};
    use burn_autodiff::Autodiff;
    use burn_cubecl::CubeBackend;
    use burn_nn::{Linear, LinearConfig, LinearRecord};
    use cubecl::cuda::CudaRuntime;

    /// CubeCL-CUDA autodiff backend used for fused optimizer tests.
    /// The four CubeBackend generics are pinned to the standard combo:
    ///   F = f32, I = i32, BT = u32 (the common cuda-friendly choice).
    type TestCubeBackend = CubeBackend<CudaRuntime, f32, i32, u32>;
    type TestAutodiffBackend = Autodiff<TestCubeBackend>;
    type FT = FloatElem<TestAutodiffBackend>;

    const LEARNING_RATE: LearningRate = 0.01;

    fn given_linear_layer(weight: TensorData, bias: TensorData) -> Linear<TestAutodiffBackend> {
        let device = Default::default();
        let record = LinearRecord {
            weight: Param::from_data(weight, &device),
            bias: Some(Param::from_data(bias, &device)),
        };
        LinearConfig::new(6, 6).init(&device).load_record(record)
    }

    fn create_adamw_fused()
    -> OptimizerAdaptor<AdamW8BitFused, Linear<TestAutodiffBackend>, TestAutodiffBackend> {
        AdamWConfig8BitFused::new().init()
    }

    #[test]
    fn test_adamw_8bit_fused_optimizer_with_numbers_one_step() {
        let linear = given_linear_layer(
            TensorData::from([
                [-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
                [0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
                [-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
                [-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
                [0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
                [-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
            ]),
            TensorData::from([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130]),
        );
        let device = Default::default();
        let x_1 = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &device,
        )
        .require_grad();

        let mut optimizer = AdamWConfig8BitFused::new()
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_weight_decay(0.5)
            .init();

        let grads = linear.forward(x_1).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_updated = linear.into_record();
        let weight_after_step1 = state_updated.weight.to_data();
        let bias_after_step1 = state_updated.bias.unwrap().to_data();

        // Print actual values so we can manually compare against running the
        // tensor-op AdamW8Bit on the same input. If step 1 produces sensible
        // values close to what we'd expect, the bug is in step 2's state handoff.
        println!("Weight after step 1: {:?}", weight_after_step1);
        println!("Bias after step 1: {:?}", bias_after_step1);

        // Sanity: weights should have moved measurably from their initial values.
        // Original [0,0] = -0.3206. After one step with LR=0.01 and WD=0.5, with
        // any reasonable gradient, we expect non-trivial movement (>1e-4 change).
        let updated_slice = weight_after_step1.as_slice::<f32>().unwrap();
        let original_slice: &[f32] = &[
            -0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671, 0.0777, -0.0185, -0.3667, 0.2550,
            0.1955, -0.2922, -0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130, -0.2980, -0.2214,
            -0.3715, -0.2981, -0.0761, 0.1626, 0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304,
            -0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833,
        ];

        let mut moved_count = 0;
        for (i, (u, o)) in updated_slice.iter().zip(original_slice.iter()).enumerate() {
            let diff = (*u - *o).abs();
            if diff > 1e-4 {
                moved_count += 1;
            }
            println!("[{}] orig={:.6} updated={:.6} diff={:.6}", i, o, u, diff);
        }

        // We expect most positions to move noticeably after one optimizer step.
        assert!(
            moved_count >= 30,
            "Only {moved_count}/36 weights moved >1e-4 — kernel may be producing near-zero deltas on step 1"
        );

        // Also assert no NaN.
        for val in updated_slice {
            assert!(val.is_finite(), "non-finite weight after step 1: {val}");
        }
    }

    #[test]
    fn test_adamw_8bit_fused_save_load_state() {
        let device = Default::default();
        let linear = LinearConfig::new(6, 6).init(&device);
        let x = Tensor::<TestAutodiffBackend, 2>::random([2, 6], Distribution::Default, &device);
        let mut optimizer = create_adamw_fused();
        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let _linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_optim_before = optimizer.to_record();
        let state_optim_before_copy = optimizer.to_record();
        let optimizer = create_adamw_fused();
        let optimizer = optimizer.load_record(state_optim_before_copy);
        let state_optim_after = optimizer.to_record();

        assert_eq!(state_optim_before.len(), state_optim_after.len());
    }

    #[test]
    fn test_adamw_8bit_fused_optimizer_with_numbers() {
        let linear = given_linear_layer(
            TensorData::from([
                [-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
                [0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
                [-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
                [-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
                [0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
                [-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
            ]),
            TensorData::from([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130]),
        );
        let device = Default::default();
        let x_1 = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [
                [0.6294, 0.0940, 0.8176, 0.8824, 0.5228, 0.4310],
                [0.7152, 0.9559, 0.7893, 0.5684, 0.5939, 0.8883],
            ],
            &device,
        )
        .require_grad();
        let x_2 = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [
                [0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
                [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, 0.9085],
            ],
            &device,
        )
        .require_grad();

        let mut optimizer = AdamWConfig8BitFused::new()
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_weight_decay(0.5)
            .init();

        let grads = linear.forward(x_1).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let grads = linear.forward(x_2).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_updated = linear.into_record();
        let weights_expected = TensorData::from([
            [-0.337295, 0.117827, 0.380358, 0.296868, 0.065232, 0.046534],
            [
                0.057032, -0.036518, -0.382951, 0.232516, 0.173738, -0.309182,
            ],
            [
                -0.038703, 0.016052, -0.313155, 0.225982, -0.295039, 0.289981,
            ],
            [
                -0.314920, -0.237394, -0.387704, -0.315067, -0.095153, 0.141081,
            ],
            [
                0.306815, -0.234226, 0.348083, -0.191115, 0.356002, -0.049993,
            ],
            [-0.035634, -0.030083, 0.104636, 0.170244, 0.009196, 0.359580],
        ]);
        let bias_expected = TensorData::from([
            -0.406555, 0.067568, -0.115982, 0.096477, 0.115287, -0.007080,
        ]);

        let (weight_updated, bias_updated) = (
            state_updated.weight.to_data(),
            state_updated.bias.unwrap().to_data(),
        );

        let tolerance = Tolerance::absolute(1e-2);
        bias_updated.assert_approx_eq::<FT>(&bias_expected, tolerance);
        weight_updated.assert_approx_eq::<FT>(&weights_expected, tolerance);
    }

    #[test]
    fn test_adamw_8bit_fused_optimizer_no_nan() {
        let linear = given_linear_layer(
            TensorData::from([
                [-0.3206, 0.1374, 0.4043, 0.3200, 0.0859, 0.0671],
                [0.0777, -0.0185, -0.3667, 0.2550, 0.1955, -0.2922],
                [-0.0190, 0.0346, -0.2962, 0.2484, -0.2780, 0.3130],
                [-0.2980, -0.2214, -0.3715, -0.2981, -0.0761, 0.1626],
                [0.3300, -0.2182, 0.3717, -0.1729, 0.3796, -0.0304],
                [-0.0159, -0.0120, 0.1258, 0.1921, 0.0293, 0.3833],
            ]),
            TensorData::from([-0.3905, 0.0884, -0.0970, 0.1176, 0.1366, 0.0130]),
        );

        let x = Tensor::<TestAutodiffBackend, 2>::from_floats(
            [
                [0.8491, 0.2108, 0.8939, 0.4433, 0.5527, 0.2528],
                [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, 0.9085],
            ],
            &Default::default(),
        )
        .require_grad();

        let mut optimizer = AdamWConfig8BitFused::new()
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_weight_decay(0.5)
            .init();

        let grads = linear.forward(x.clone()).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_updated = linear.into_record();
        assert!(!state_updated.weight.to_data().as_slice::<f32>().unwrap()[0].is_nan());
    }

    #[test]
    fn test_adamw_8bit_fused_distribution_shift_nan_check() {
        let device = Default::default();

        let weight_data = TensorData::ones::<f32, _>([12, 12]);
        let bias_data = TensorData::ones::<f32, _>([12]);

        let mut linear = given_linear_layer(weight_data, bias_data);

        let mut optimizer = AdamWConfig8BitFused::new().with_epsilon(1e-8).init();

        // Phase 1: high-energy gradient regime
        for _ in 1..=20 {
            let x =
                Tensor::<TestAutodiffBackend, 2>::random([4, 12], Distribution::Default, &device)
                    .mul_scalar(50.0);
            let grads = linear.forward(x).backward();
            let grads = GradientsParams::from_grads(grads, &linear);
            linear = optimizer.step(LEARNING_RATE, linear, grads);
        }

        // Phase 2: tiny-gradient regime
        for step in 1..=100 {
            let x =
                Tensor::<TestAutodiffBackend, 2>::random([4, 12], Distribution::Default, &device)
                    .mul_scalar(0.001);
            let grads = linear.forward(x).backward();
            let grads = GradientsParams::from_grads(grads, &linear);
            linear = optimizer.step(LEARNING_RATE, linear, grads);

            let weights = linear.weight.val().to_data();
            for (i, val) in weights.as_slice::<f32>().unwrap().iter().enumerate() {
                if !val.is_finite() {
                    panic!(
                        "NaN/Inf detected at step {} (index {}). Value: {}",
                        step, i, val
                    );
                }
            }
        }
    }

    use crate::kernel::quantize_blockwise_signed::quantize_blockwise_signed_via_kernel;
    use cubecl::Runtime;

    #[cfg(feature = "test-cuda")]
    #[test]
    fn diagnostic_36_elements_roundtrip() {
        use crate::kernel::dequantize_blockwise_signed::dequantize_blockwise_signed_via_kernel;
        use cubecl::cuda::CudaRuntime;

        let device = <CudaRuntime as Runtime>::Device::default();
        let client = <CudaRuntime as Runtime>::client(&device);

        // 36 distinct nonzero values, mid-range magnitudes (depth 0-1 in encoding).
        let data: Vec<f32> = (0..36).map(|i| 0.1 + 0.01 * (i as f32)).collect();

        let q = quantize_blockwise_signed_via_kernel::<CudaRuntime>(&client, &data);
        let recovered =
            dequantize_blockwise_signed_via_kernel::<CudaRuntime>(&client, &q.codes, &q.scales, 36);

        println!("Standalone blockwise roundtrip on 36 elements:");
        println!("idx | original | recovered | rel_err | flag");
        for i in 0..36 {
            let rel = ((recovered[i] - data[i]) / data[i]).abs();
            let flag = if rel > 0.10 { " *** BROKEN ***" } else { "" };
            println!(
                "[{:>2}] {:>9.6} | {:>9.6} | {:.4}{}",
                i, data[i], recovered[i], rel, flag
            );
        }
    }

    #[test]
    fn diagnostic_fused_uniform_inputs() {
        use crate::kernel::fused_adamw8bit_step::{
            FusedStepInput, fused_adamw8bit_step_via_kernel,
        };

        let device = <CudaRuntime as cubecl::Runtime>::Device::default();
        let client = <CudaRuntime as cubecl::Runtime>::client(&device);

        // 36-element problem. Pad will make num_blocks=1, block=256 elements.
        let n = 36;
        let grad = vec![1.0f32; n]; // uniform gradient

        // Initial state: zero codes, scale 1.0. 36 padded to 256 → 1 block.
        let num_blocks = 1;
        let m_codes = vec![0u32; num_blocks * 64];
        let m_scales = vec![1.0f32; num_blocks];
        let v_codes = vec![0u32; num_blocks * 64];
        let v_scales = vec![1.0f32; num_blocks];

        let result = fused_adamw8bit_step_via_kernel::<CudaRuntime>(
            &client,
            FusedStepInput {
                grad: &grad,
                m_codes: &m_codes,
                m_scales: &m_scales,
                v_codes: &v_codes,
                v_scales: &v_scales,
                original_len: n,
                beta_1: 0.9,
                beta_2: 0.999,
                epsilon: 1e-8,
                time_step: 1,
            },
        );

        println!("Fused step with uniform grad=1.0, zero state:");
        println!("idx | delta | flag");
        let first_delta = result.delta[0];
        for (i, d) in result.delta.iter().enumerate() {
            let rel_to_first = (d - first_delta).abs() / first_delta.abs().max(1e-12);
            let flag = if rel_to_first > 1e-3 {
                " *** DIFFERS ***"
            } else {
                ""
            };
            println!("[{:>2}] {:>10.6}{}", i, d, flag);
        }
    }
}
