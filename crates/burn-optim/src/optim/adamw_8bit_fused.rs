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
use crate::launch::{AdamWStepInputs, AdamWStepParams, PACKING_AMOUNT, launch_adamw_8bit_step};
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

        assert_eq!(
            numel % self.momentum.block_size,
            0,
            "param numel must be divisible by block_size"
        );
        let num_blocks = numel / self.momentum.block_size;
        let packed_count = numel / PACKING_AMOUNT as usize;

        let time = state.as_ref().map(|s| s.time + 1).unwrap_or(1);

        // Extract param + grad handles
        let theta_prim = tensor.into_primitive();
        let theta_cube = match theta_prim {
            TensorPrimitive::Float(t) => t,
            TensorPrimitive::QFloat(_) => panic!("quantized inputs not supported"),
        };
        let client = theta_cube.client.clone();
        let theta_handle = theta_cube.handle.clone();

        let grad_prim = grad.into_primitive();
        let grad_cube = match grad_prim {
            TensorPrimitive::Float(t) => t,
            TensorPrimitive::QFloat(_) => panic!("quantized inputs not supported"),
        };
        let grad_handle = grad_cube.handle;

        // First step: allocate state buffers once. Subsequent: reuse existing.
        let (m1_codes_h, m1_scales_h, m2_codes_h, m2_scales_h, max_v_codes_h, max_v_scales_h) =
            match state {
                Some(s) => {
                    let m1c = s.moment_1.quantized.into_primitive().handle.clone();
                    let m1s = match s.moment_1.scales.into_primitive() {
                        TensorPrimitive::Float(t) => t.handle.clone(),
                        _ => unreachable!(),
                    };
                    let m2c = s.moment_2.quantized.into_primitive().handle.clone();
                    let m2s = match s.moment_2.scales.into_primitive() {
                        TensorPrimitive::Float(t) => t.handle.clone(),
                        _ => unreachable!(),
                    };
                    let (mvc, mvs) = match s.max_moment_2 {
                        Some(m) => {
                            let c = m.quantized.into_primitive().handle.clone();
                            let s = match m.scales.into_primitive() {
                                TensorPrimitive::Float(t) => t.handle.clone(),
                                _ => unreachable!(),
                            };
                            (Some(c), Some(s))
                        }
                        None => (None, None),
                    };
                    (m1c, m1s, m2c, m2s, mvc, mvs)
                }
                None => {
                    // Allocate state ONCE — kernel initializes contents from zero.
                    let m1c = client.empty(packed_count * core::mem::size_of::<u32>());
                    let m1s = client.empty(num_blocks * core::mem::size_of::<f32>());
                    let m2c = client.empty(packed_count * core::mem::size_of::<u32>());
                    let m2s = client.empty(num_blocks * core::mem::size_of::<f32>());
                    let (mvc, mvs) = if self.momentum.amsgrad {
                        (
                            Some(client.empty(packed_count * core::mem::size_of::<u32>())),
                            Some(client.empty(num_blocks * core::mem::size_of::<f32>())),
                        )
                    } else {
                        (None, None)
                    };
                    (m1c, m1s, m2c, m2s, mvc, mvs)
                }
            };

        let inputs = AdamWStepInputs {
            theta: theta_handle.clone(),
            grad: grad_handle,
            moment_1_codes: Some(m1_codes_h.clone()),
            moment_1_scales: Some(m1_scales_h.clone()),
            moment_2_codes: Some(m2_codes_h.clone()),
            moment_2_scales: Some(m2_scales_h.clone()),
            max_moment_2_codes: max_v_codes_h.clone(),
            max_moment_2_scales: max_v_scales_h.clone(),
            numel,
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

        let _ = launch_adamw_8bit_step(&client, inputs, params);

        // Rewrap the SAME handles back into Burn tensors.
        let theta_new_cube = CubeTensor {
            client: client.clone(),
            handle: theta_handle,
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

        let moment_1 = build_quantize_blockwise::<R, F, I, BT, D>(
            &client,
            &device,
            m1_codes_h,
            m1_scales_h,
            shape.dims(),
            self.momentum.block_size,
        );
        let moment_2 = build_quantize_blockwise::<R, F, I, BT, D>(
            &client,
            &device,
            m2_codes_h,
            m2_scales_h,
            shape.dims(),
            self.momentum.block_size,
        );
        let max_moment_2 = if self.momentum.amsgrad {
            Some(build_quantize_blockwise::<R, F, I, BT, D>(
                &client,
                &device,
                max_v_codes_h.unwrap(),
                max_v_scales_h.unwrap(),
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
    let packed_count = total_elements / PACKING_AMOUNT as usize;
    let num_blocks = total_elements / block_size;

    // Codes tensor: 1D buffer of `packed_count` u32s.
    let codes_cube = CubeTensor {
        client: client.clone(),
        handle: codes_handle,
        device: device.clone(),
        dtype: I::dtype(),
        qparams: None,
        meta: Box::new(Metadata::new(
            Shape::new([packed_count]),
            vec![1], // 1D contiguous
        )),
    };

    // Scales tensor: 1D buffer of `num_blocks` f32s.
    let scales_cube = CubeTensor {
        client: client.clone(),
        handle: scales_handle,
        device: device.clone(),
        dtype: DType::F32,
        qparams: None,
        meta: Box::new(Metadata::new(
            Shape::new([num_blocks]),
            vec![1], // 1D contiguous
        )),
    };

    QuantizeBlockwise {
        quantized: Tensor::from_primitive(codes_cube),
        scales: Tensor::from_primitive(TensorPrimitive::Float(scales_cube)),
        shape, // The original parameter shape — stored alongside, not on the buffers
    }
}
