//! An 8-bit optimizer of AdamW.

use burn_core as burn;
use burn_core::tensor::DType;

use burn::config::Config;
use burn::tensor::TensorPrimitive;
use burn::tensor::{
    Tensor,
    backend::{AutodiffBackend, Backend},
    ops::Device,
};
use burn::{module::AutodiffModule, record::Record};
use burn_autodiff::Autodiff;
use burn_core::prelude::Shape;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use cubecl::CubeElement;
use cubecl::Runtime;
use cubecl::bytes::Bytes;
use cubecl::client::ComputeClient;
use cubecl::cuda::CudaRuntime;
use cubecl::server::Handle;
use cubecl::zspace::metadata::Metadata;

use super::{SimpleOptimizer, adaptor::OptimizerAdaptor};
use crate::launch::{AdamWStepInputs, AdamWStepParams, launch_adamw_8bit_step};
use crate::quantization::{
    QuantizeBlockwise, dequantize_blockwise, quantize_blockwise, signed_dynamic, unsigned_dynamic,
};
use crate::{LearningRate, grad_clipping::GradientClippingConfig};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// [`AdamW8BitFused`] Configuration.
#[derive(Config, Debug)]
pub struct AdamWConfig8BitFused {
    /// Parameter for AdamW.
    #[config(default = 0.9)]
    beta_1: f32,
    /// Parameter for AdamW.
    #[config(default = 0.999)]
    beta_2: f32,
    /// The amount of quantization applied to the optimizer. Always use a power of 2, or have
    /// highly degraded performance. Default value for the bitsandbytes library.
    #[config(default = 256)]
    block_size: usize,
    /// A value required for numerical stability.
    #[config(default = 1e-5)]
    epsilon: f32,
    /// Weight decay config.
    #[config(default = 1e-4)]
    weight_decay: f32,

    /// Cautious weight decay config.
    ///
    /// See: <https://arxiv.org/abs/2510.12402>
    #[config(default = false)]
    cautious_weight_decay: bool,

    /// Whether to use AMSGrad algorithm
    #[config(default = false)]
    amsgrad: bool,
    /// [Gradient Clipping](GradientClippingConfig) config.
    grad_clipping: Option<GradientClippingConfig>,
}

/// AdamW 8 bit optimizer.
///
/// See:
/// - [`crate::optim::AdamW`] module
/// - [8-Bit Optimizers via Block-Wise Quantization](https://arxiv.org/pdf/2110.02861)
///
/// Configured by [`AdamWConfig`].
#[derive(Clone)]
pub struct AdamW8BitFused {
    momentum: AdaptiveMomentumW8Bit,
    weight_decay: f32,
    cautious_weight_decay: bool,
}

/// [`AdamW8Bit`] state.
#[derive(Record, Clone)]
pub struct AdamWState8BitFused<B: Backend, const D: usize> {
    time: usize,
    moment_1: QuantizeBlockwise<B, D>,
    moment_2: QuantizeBlockwise<B, D>,
    max_moment_2: Option<QuantizeBlockwise<B, D>>,
}

// impl SimpleOptimizer<MyBackend> for AdamW8BitFused {
//     type State<const D: usize> = AdamWState8BitFused<MyBackend, D>;

//     fn step<const D: usize>(
//         &self,
//         lr: LearningRate,
//         tensor: Tensor<MyBackend, D>,
//         grad: Tensor<MyBackend, D>,
//         state: Option<Self::State<D>>,
//     ) -> (Tensor<MyBackend, D>, Option<Self::State<D>>) {
//     }

//     fn to_device<const D: usize>(
//         mut state: Self::State<D>,
//         device: &Device<MyBackend>,
//     ) -> Self::State<D> {
//         state.moment_1 = state.moment_1.to_device(device);
//         state.moment_2 = state.moment_2.to_device(device);
//         state.max_moment_2 = state.max_moment_2.map(|m| m.to_device(device));
//         state
//     }
// }

impl<R, F, I, BT> SimpleOptimizer<CubeBackend<R, F, I, BT>> for AdamW8BitFused
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    type State<const D: usize> = AdamWState8BitFused<CubeBackend<R, F, I, BT>, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<CubeBackend<R, F, I, BT>, D>,
        grad: Tensor<CubeBackend<R, F, I, BT>, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<CubeBackend<R, F, I, BT>, D>, Option<Self::State<D>>) {
        let device = tensor.device();
        let shape = tensor.shape();
        let numel = shape.num_elements();
        let dims: [_; D] = shape.dims(); // ← capture dims while shape is still alive

        // 1. Compute time and bias corrections on the host
        let time = state.as_ref().map(|s| s.time + 1).unwrap_or(1);
        let is_first_step = state.is_none();

        let theta_prim = tensor.clone().into_primitive();
        let theta_cube = match theta_prim {
            TensorPrimitive::Float(t) => t,
            TensorPrimitive::QFloat(_) => panic!("quantized inputs not supported"),
        };
        let theta_handle = theta_cube.handle.clone();
        let client = theta_cube.client.clone(); // ← only one client, from the tensor

        // Now you have access to its handle and client.
        let theta_handle = theta_cube.handle.clone();
        let client = theta_cube.client.clone();

        // Same for grad.
        let grad_prim = grad.into_primitive();
        let grad_cube = match grad_prim {
            TensorPrimitive::Float(t) => t,
            TensorPrimitive::QFloat(_) => panic!("quantized inputs not supported"),
        };
        let grad_handle = grad_cube.handle.clone();
        let (m1_codes, m1_scales, m2_codes, m2_scales, max_v_codes, max_v_scales) = match state {
            Some(s) => {
                // moment_1: QuantizeBlockwise<B, D> { quantized: Tensor<B, 1, Int>, scales: Tensor<B, 1>, ... }
                let m1_codes_prim = s.moment_1.quantized.into_primitive();
                let m1_codes_handle = m1_codes_prim.handle.clone();

                let m1_scales_prim = s.moment_1.scales.into_primitive();
                let m1_scales_cube = match m1_scales_prim {
                    TensorPrimitive::Float(t) => t,
                    TensorPrimitive::QFloat(_) => panic!("unexpected"),
                };
                let m1_scales_handle = m1_scales_cube.handle.clone();

                let m2_codes_handle = s.moment_2.quantized.into_primitive().handle.clone();
                let m2_scales_handle = match s.moment_2.scales.into_primitive() {
                    TensorPrimitive::Float(t) => t.handle.clone(),
                    TensorPrimitive::QFloat(_) => panic!("unexpected"),
                };

                let (max_v_codes_handle, max_v_scales_handle) = match s.max_moment_2 {
                    Some(m) => {
                        let codes_handle = m.quantized.into_primitive().handle.clone();
                        let scales_handle = match m.scales.into_primitive() {
                            TensorPrimitive::Float(t) => t.handle.clone(),
                            TensorPrimitive::QFloat(_) => panic!("unexpected"),
                        };
                        (Some(codes_handle), Some(scales_handle))
                    }
                    None => (None, None),
                };

                (
                    Some(m1_codes_handle),
                    Some(m1_scales_handle),
                    Some(m2_codes_handle),
                    Some(m2_scales_handle),
                    max_v_codes_handle,
                    max_v_scales_handle,
                )
            }
            None => (None, None, None, None, None, None),
        };

        // 3. Build inputs for the launcher
        let inputs = AdamWStepInputs {
            theta: theta_handle,
            grad: grad_handle,
            moment_1_codes: m1_codes,
            moment_1_scales: m1_scales,
            moment_2_codes: m2_codes,
            moment_2_scales: m2_scales,
            max_moment_2_codes: max_v_codes,
            max_moment_2_scales: max_v_scales,
            numel: tensor.shape().num_elements(),
            _phantom: core::marker::PhantomData,
        };

        let params = AdamWStepParams {
            beta_1: self.momentum.beta_1,
            beta_2: self.momentum.beta_2,
            epsilon: self.momentum.epsilon,
            lr: lr as f32,
            weight_decay: self.weight_decay,
            time: time as u32,
            block_size: self.momentum.block_size as u32,
            amsgrad: self.momentum.amsgrad,
            cautious_weight_decay: self.cautious_weight_decay,
        };

        // 4. Call the launcher (returns raw handles)
        let output = launch_adamw_8bit_step(&client, inputs, params);

        // 5. Wrap output handles back into Burn tensors
        let theta_new_cube = CubeTensor {
            client: client.clone(),
            handle: output.theta_new,
            device: device.clone(),
            dtype: DType::F32,
            qparams: None,
            meta: Box::new(Metadata::new(
                shape.clone(),
                contiguous_strides::<D>(&shape.dims()),
            )),
        };

        let theta_new: Tensor<CubeBackend<R, F, I, BT>, D> =
            Tensor::from_primitive(TensorPrimitive::Float(theta_new_cube));

        // Build the new QuantizeBlockwise structs the same way.
        let moment_1 = build_quantize_blockwise::<R, F, I, BT, D>(
            &client,
            &device,
            output.moment_1_codes,
            output.moment_1_scales,
            shape.dims(),
            self.momentum.block_size,
        );
        let moment_2 = build_quantize_blockwise::<R, F, I, BT, D>(
            &client,
            &device,
            output.moment_2_codes,
            output.moment_2_scales,
            shape.dims(),
            self.momentum.block_size,
        );
        let max_moment_2 = if self.momentum.amsgrad {
            Some(build_quantize_blockwise::<R, F, I, BT, D>(
                &client,
                &device,
                output.max_moment_2_codes.unwrap(),
                output.max_moment_2_scales.unwrap(),
                shape.dims(),
                self.momentum.block_size,
            ))
        } else {
            None
        };

        let new_state = AdamWState8BitFused {
            time,
            moment_1,
            moment_2,
            max_moment_2,
        };

        (theta_new, Some(new_state))
    }

    fn to_device<const D: usize>(
        mut state: Self::State<D>,
        device: &<CubeBackend<R, F, I, BT> as Backend>::Device,
    ) -> Self::State<D> {
        state.moment_1 = state.moment_1.to_device(device);
        state.moment_2 = state.moment_2.to_device(device);
        state.max_moment_2 = state.max_moment_2.map(|m| m.to_device(device));
        state
    }
}

impl AdamWConfig8BitFused {
    pub fn init<R, F, I, BT, M>(
        &self,
    ) -> OptimizerAdaptor<AdamW8BitFused, M, Autodiff<CubeBackend<R, F, I, BT>>>
    where
        R: CubeRuntime,
        F: FloatElement,
        I: IntElement,
        BT: BoolElement,
        M: AutodiffModule<Autodiff<CubeBackend<R, F, I, BT>>>,
    {
        let optim = AdamW8BitFused {
            momentum: AdaptiveMomentumW8Bit {
                beta_1: self.beta_1,
                beta_2: self.beta_2,
                epsilon: self.epsilon,
                amsgrad: self.amsgrad,
                block_size: self.block_size,
            },
            weight_decay: self.weight_decay,
            cautious_weight_decay: self.cautious_weight_decay,
        };

        let mut optim = OptimizerAdaptor::from(optim);
        if let Some(config) = &self.grad_clipping {
            optim = optim.with_grad_clipping(config.init());
        }
        optim
    }
}

#[derive(Clone)]
struct AdaptiveMomentumW8Bit {
    beta_1: f32,
    beta_2: f32,
    epsilon: f32,
    amsgrad: bool,
    block_size: usize,
}

/// Compute contiguous row-major strides for a shape.
// fn contiguous_strides<const D: usize>(shape: &Shape) -> Vec<usize> {
//     let dims: [_; D] = shape.dims();
//     let n = dims.len();
//     let mut strides = vec![1; n];
//     for i in (0..n - 1).rev() {
//         strides[i] = strides[i + 1] * dims[i + 1];
//     }
//     strides
// }

fn contiguous_strides<const D: usize>(dims: &[usize; D]) -> Vec<usize> {
    let n = D;
    let mut strides = vec![1; n];
    for i in (0..n.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// Build a QuantizeBlockwise<B, D> from raw codes and scales handles.
fn build_quantize_blockwise<R, F, I, BT, const D: usize>(
    client: &ComputeClient<R>,
    device: &R::Device,
    codes_handle: Handle,
    scales_handle: Handle,
    shape: [usize; D],
    block_size: usize,
) -> QuantizeBlockwise<CubeBackend<R, F, I, BT>, D>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    let total_elements: usize = shape.iter().product();
    let packed_count = total_elements / 2; // PACKING_AMOUNT = 2
    let num_blocks = total_elements / block_size;

    // Codes tensor: 1D, length = packed_count, dtype = the int element type.
    let codes_cube = CubeTensor {
        client: client.clone(),
        handle: codes_handle,
        device: device.clone(),
        dtype: I::dtype(), // ← whatever produces the right DType for the int type
        qparams: None,
        meta: Box::new(Metadata::new(
            shape.clone(),
            contiguous_strides::<D>(&shape),
        )),
    };

    // Scales tensor: 1D, length = num_blocks, dtype = f32.
    let scales_cube = CubeTensor {
        client: client.clone(),
        handle: scales_handle,
        device: device.clone(),
        dtype: DType::F32,
        qparams: None,
        meta: Box::new(Metadata::new(
            shape.clone(),
            contiguous_strides::<D>(&shape),
        )),
    };

    QuantizeBlockwise {
        quantized: Tensor::from_primitive(codes_cube),
        scales: Tensor::from_primitive(TensorPrimitive::Float(scales_cube)),
        shape,
    }
}
