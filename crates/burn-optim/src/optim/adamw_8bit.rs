//! An 8-bit optimizer of AdamW.

use burn_core as burn;

use burn::config::Config;
use burn::tensor::{
    Tensor,
    backend::{AutodiffBackend, Backend},
    ops::Device,
};
use burn::{module::AutodiffModule, record::Record};

use super::{SimpleOptimizer, adaptor::OptimizerAdaptor};
use crate::quantization::{
    QuantizeBlockwise, dequantize_blockwise, quantize_blockwise, signed_dynamic, unsigned_dynamic,
};
use crate::{LearningRate, grad_clipping::GradientClippingConfig};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float as _;

/// [`AdamW8Bit`] Configuration.
#[derive(Config, Debug)]
pub struct AdamWConfig8Bit {
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
pub struct AdamW8Bit {
    momentum: AdaptiveMomentumW8Bit,
    weight_decay: f32,
    cautious_weight_decay: bool,
}

/// [`AdamW8Bit`] state.
#[derive(Record, Clone)]
pub struct AdamWState8Bit<B: Backend, const D: usize> {
    time: usize,
    moment_1: QuantizeBlockwise<B, D>,
    moment_2: QuantizeBlockwise<B, D>,
    max_moment_2: Option<QuantizeBlockwise<B, D>>,
}

impl<B: Backend> SimpleOptimizer<B> for AdamW8Bit {
    type State<const D: usize> = AdamWState8Bit<B, D>;

    /// A single optimization step for any tensor that represents the parameters of a model.
    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let (raw_delta, new_state, m1) = self.momentum.transform(grad, state);

        let decay_rate = lr * (self.weight_decay as f64);
        let decayed_tensor = if decay_rate == 0.0 {
            tensor.clone()
        } else if self.cautious_weight_decay {
            let tensor_pos = tensor.clone().greater_equal_elem(0.0);

            let grad_pos = m1.greater_equal_elem(0.0);
            let differ = tensor_pos.not_equal(grad_pos);

            let decay = tensor.clone().mul_scalar(decay_rate).mask_fill(differ, 0.0);
            tensor.clone() - decay
        } else {
            tensor.clone().mul_scalar(1.0 - decay_rate)
        };

        let tensor_updated = decayed_tensor - raw_delta.mul_scalar(lr);

        (tensor_updated, Some(new_state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.moment_1 = state.moment_1.to_device(device);
        state.moment_2 = state.moment_2.to_device(device);
        state.max_moment_2 = state.max_moment_2.map(|m| m.to_device(device));
        state
    }
}

impl AdamWConfig8Bit {
    /// Initialize [`AdamW8Bit`] optimizer.
    ///
    /// # Returns
    ///
    /// Returns an optimizer that can be used to optimize a module.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<AdamW8Bit, M, B> {
        let optim = AdamW8Bit {
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

impl AdaptiveMomentumW8Bit {
    pub fn transform<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        state: Option<AdamWState8Bit<B, D>>, // Using 8-bit state
    ) -> (Tensor<B, D>, AdamWState8Bit<B, D>, Tensor<B, D>) {
        let factor_1 = 1.0 - self.beta_1;
        let factor_2 = 1.0 - self.beta_2;

        let (mut m1, mut m2, mut max_v, time) = if let Some(s) = state {
            (
                dequantize_blockwise::<B, D, _>(
                    s.moment_1,
                    self.block_size,
                    signed_dynamic::decode,
                ),
                dequantize_blockwise::<B, D, _>(
                    s.moment_2,
                    self.block_size,
                    unsigned_dynamic::decode,
                ),
                s.max_moment_2.map(|m| {
                    dequantize_blockwise::<B, D, _>(m, self.block_size, unsigned_dynamic::decode)
                }),
                s.time + 1,
            )
        } else {
            (
                Tensor::zeros(grad.shape(), &grad.device()),
                Tensor::zeros(grad.shape(), &grad.device()),
                None,
                1,
            )
        };

        // Full precision.
        m1 = m1
            .mul_scalar(self.beta_1)
            .add(grad.clone().mul_scalar(factor_1));
        m2 = m2
            .mul_scalar(self.beta_2)
            .add(grad.square().mul_scalar(factor_2));

        let v_to_use = if self.amsgrad {
            let current_max = max_v.unwrap_or_else(|| m2.clone());
            let new_max = current_max.max_pair(m2.clone());
            max_v = Some(new_max.clone());
            new_max
        } else {
            m2.clone()
        };

        // Compute delta.
        let correction1 = 1.0 - self.beta_1.powi(time as i32);
        let correction2 = (1.0 - self.beta_2.powi(time as i32)).sqrt();
        let step_size = correction2 / correction1; // absorb lr into caller
        let update_delta = m1
            .clone()
            .div(v_to_use.sqrt().add_scalar(self.epsilon * correction2))
            .mul_scalar(step_size);

        // Requantize for storage.
        let state_8bit = AdamWState8Bit {
            time,
            moment_1: quantize_blockwise(m1.clone(), self.block_size, signed_dynamic::encode),
            moment_2: quantize_blockwise(m2, self.block_size, unsigned_dynamic::encode),
            max_moment_2: max_v
                .map(|m| quantize_blockwise(m, self.block_size, unsigned_dynamic::encode)),
        };

        (update_delta, state_8bit, m1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestAutodiffBackend;
    use crate::{GradientsParams, Optimizer};
    use burn::module::{Module, Param};
    use burn::tensor::{Distribution, Tensor, TensorData};
    use burn::tensor::{Tolerance, ops::FloatElem};
    use burn_nn::{Linear, LinearConfig, LinearRecord};

    type FT = FloatElem<TestAutodiffBackend>;

    const LEARNING_RATE: LearningRate = 0.01;

    #[test]
    fn test_adamw_8bit_optimizer_save_load_state() {
        let device = Default::default();
        let linear = LinearConfig::new(6, 6).init(&device);
        let x = Tensor::<TestAutodiffBackend, 2>::random([2, 6], Distribution::Default, &device);
        let mut optimizer = create_adamw_8bit();
        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let _linear = optimizer.step(LEARNING_RATE, linear, grads);

        #[cfg(feature = "std")]
        {
            use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

            BinFileRecorder::<FullPrecisionSettings>::default()
                .record(
                    optimizer.to_record(),
                    std::env::temp_dir().as_path().join("test_optim_adamw"),
                )
                .unwrap();
        }
        #[cfg(not(feature = "std"))]
        {
            use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};

            let result = BinBytesRecorder::<FullPrecisionSettings>::default()
                .record(optimizer.to_record(), ())
                .unwrap();
            assert!(!result.is_empty());
        }

        let state_optim_before = optimizer.to_record();
        let state_optim_before_copy = optimizer.to_record();
        let optimizer = create_adamw_8bit();
        let optimizer = optimizer.load_record(state_optim_before_copy);
        let state_optim_after = optimizer.to_record();

        assert_eq!(state_optim_before.len(), state_optim_after.len());
    }
    #[test]
    fn test_adamw_8bit_optimizer_with_amsgrad_50_steps() {
        let device = Default::default();
        let mut linear = given_linear_layer(
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

        let mut optimizer = AdamWConfig8Bit::new()
            .with_epsilon(1e-8)
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_amsgrad(true)
            .with_weight_decay(0.5)
            .init();

        for i in 1..=50 {
            let x = Tensor::<TestAutodiffBackend, 2>::ones([2, 6], &device)
                .mul_scalar(i as f32 * 0.1)
                .require_grad();

            let grads = linear.forward(x).backward();
            let grads = GradientsParams::from_grads(grads, &linear);
            linear = optimizer.step(LEARNING_RATE, linear, grads);
        }

        let state_updated = linear.into_record();
        let weight_updated = state_updated.weight.to_data();
        let bias_updated = state_updated.bias.unwrap().to_data();

        let weights_expected = TensorData::from([
            [
                -0.7822558283805847,
                -0.42578864097595215,
                -0.21805696189403534,
                -0.28366872668266296,
                -0.46587175130844116,
                -0.4805040955543518,
            ],
            [
                -0.4722539782524109,
                -0.5471276640892029,
                -0.8181359767913818,
                -0.33425918221473694,
                -0.3805687427520752,
                -0.7601516842842102,
            ],
            [
                -0.5475167632102966,
                -0.5057991743087769,
                -0.763265073299408,
                -0.3393959403038025,
                -0.7490996718406677,
                -0.28911691904067993,
            ],
            [
                -0.7646660208702087,
                -0.7050473093986511,
                -0.8218720555305481,
                -0.7647438049316406,
                -0.5919585227966309,
                -0.40617525577545166,
            ],
            [
                -0.27588561177253723,
                -0.7025567889213562,
                -0.24343004822731018,
                -0.6672990918159485,
                -0.23728127777576447,
                -0.556389570236206,
            ],
            [
                -0.5451040267944336,
                -0.5420684814453125,
                -0.4348171353340149,
                -0.3832150399684906,
                -0.5099242925643921,
                -0.23440153896808624,
            ],
        ]);
        let bias_expected = TensorData::from([
            -0.7473056316375732,
            -0.3745720386505127,
            -0.5188710689544678,
            -0.35184532403945923,
            -0.33705732226371765,
            -0.4332566559314728,
        ]);

        type FT = FloatElem<TestAutodiffBackend>;
        let tolerance = Tolerance::absolute(1e-5);
        weight_updated.assert_approx_eq::<FT>(&weights_expected, tolerance);
        bias_updated.assert_approx_eq::<FT>(&bias_expected, tolerance);
    }
    #[test]
    fn test_adamw_8bit_optimizer_with_numbers() {
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

        let mut optimizer = AdamWConfig8Bit::new()
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
    fn test_adamw_8bit_optimizer_with_numbers_cautious() {
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
                [0.3270, 0.0412, 0.5538, 0.9605, 0.3195, -0.9085],
            ],
            &device,
        )
        .require_grad();

        let mut optimizer = AdamWConfig8Bit::new()
            .with_cautious_weight_decay(true)
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
            [
                -0.035634, -0.030083, 0.104636, 0.170244, 0.009196, 0.37061332,
            ],
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
    fn test_adamw_8bit_optimizer_no_nan() {
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

        let mut optimizer = AdamWConfig8Bit::new()
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

    fn given_linear_layer(weight: TensorData, bias: TensorData) -> Linear<TestAutodiffBackend> {
        let device = Default::default();
        let record = LinearRecord {
            weight: Param::from_data(weight, &device),
            bias: Some(Param::from_data(bias, &device)),
        };

        LinearConfig::new(6, 6).init(&device).load_record(record)
    }

    fn create_adamw_8bit()
    -> OptimizerAdaptor<AdamW8Bit, Linear<TestAutodiffBackend>, TestAutodiffBackend> {
        let config = AdamWConfig8Bit::new();
        AdamW8Bit {
            momentum: AdaptiveMomentumW8Bit {
                beta_1: config.beta_1,
                beta_2: config.beta_2,
                epsilon: config.epsilon,
                amsgrad: config.amsgrad,
                block_size: config.block_size,
            },
            weight_decay: config.weight_decay,
            cautious_weight_decay: false,
        }
        .into()
    }

    #[test]
    fn test_adamw_8bit_distribution_shift_nan_check() {
        let device = Default::default();

        // Corrected TensorData initialization
        let weight_data = TensorData::ones::<f32, _>([12, 12]);
        let bias_data = TensorData::ones::<f32, _>([12]);

        let mut linear = given_linear_layer(weight_data, bias_data);

        let mut optimizer = AdamWConfig8Bit::new()
            .with_epsilon(1e-8)
            .with_block_size(32)
            .init();

        // --- PHASE 1: The "Hot" Start (High Energy) ---
        for _ in 1..=20 {
            let x =
                Tensor::<TestAutodiffBackend, 2>::random([4, 12], Distribution::Default, &device)
                    .mul_scalar(50.0);
            let grads = linear.forward(x).backward();
            let grads = GradientsParams::from_grads(grads, &linear);
            linear = optimizer.step(LEARNING_RATE, linear, grads);
        }

        // --- PHASE 2: The "Cold" Convergence (Tiny Gradients) ---
        for step in 1..=100 {
            let x =
                Tensor::<TestAutodiffBackend, 2>::random([4, 12], Distribution::Default, &device)
                    .mul_scalar(0.001);
            let grads = linear.forward(x).backward();
            let grads = GradientsParams::from_grads(grads, &linear);
            linear = optimizer.step(LEARNING_RATE, linear, grads);

            // Check finite
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

        println!("Passed stability test without NaNs.");
    }

    #[test]
    fn test_adamw_8bit_zipfian_update_frequency() {
        use burn::tensor::{Distribution, Shape};

        let device = Default::default();
        let rows = 1024;
        let cols = 8;
        let shape = [rows, cols];

        let momentum = AdaptiveMomentumW8Bit {
            beta_1: 0.9,
            beta_2: 0.999,
            epsilon: 1e-8,
            amsgrad: false,
            block_size: 256,
        };

        let mut param = Tensor::<TestAutodiffBackend, 2>::random(
            shape,
            Distribution::Normal(0.0, 0.02),
            &device,
        );
        let mut state: Option<AdamWState8Bit<TestAutodiffBackend, 2>> = None;

        let lr = 0.01_f32;
        let mut prev_max_abs = 0.0f32;

        for step in 1..=200 {
            // Build Zipfian gradient as flat data, then reshape to 2D
            let mut grad_data = vec![0.0f32; rows * cols];

            if step % 17 == 0 {
                let rare_row = (rows * 3 / 4) + (step % (rows / 4));
                for c in 0..cols {
                    grad_data[rare_row * cols + c] = 3.0;
                }
            } else {
                let frequent_count = rows / 20;
                for r in 0..frequent_count {
                    for c in 0..cols {
                        grad_data[r * cols + c] = 0.3;
                    }
                }
            }

            // Build as 1D then reshape to 2D — avoids the from_floats rank inference issue
            let grad = Tensor::<TestAutodiffBackend, 1>::from_floats(grad_data.as_slice(), &device)
                .reshape(Shape::from(shape));

            let (delta, new_state, _m1) = momentum.transform(grad, state);
            param = param - delta.mul_scalar(lr);
            state = Some(new_state);

            let data = param.clone().to_data();
            let slice = data.as_slice::<f32>().unwrap();

            for (i, val) in slice.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "Non-finite param at step {}, index {}: {}",
                    step,
                    i,
                    val,
                );
            }

            let current_max_abs = slice.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

            if step > 20 {
                assert!(
                    current_max_abs < 10.0,
                    "Param magnitude diverged at step {}: max |p| = {} (prev: {})",
                    step,
                    current_max_abs,
                    prev_max_abs,
                );
            }

            prev_max_abs = current_max_abs;
        }
    }
}
