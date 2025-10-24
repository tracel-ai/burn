use burn_core as burn;

use burn::{module::AutodiffModule, record::Record};

use burn::config::Config;
use burn::tensor::{Tensor, backend::AutodiffBackend};
use burn::tensor::{backend::Backend, ops::Device};

use super::{
    SimpleOptimizer,
    adaptor::OptimizerAdaptor,
    decay::{WeightDecay, WeightDecayConfig},
};
use crate::{LearningRate, grad_clipping::GradientClippingConfig};

/// AdaGrad configuration.
#[derive(Config, Debug)]
pub struct AdaGradConfig {
    #[config(default = 0.)]
    lr_decay: f64,
    #[config(default = 1e-5)]
    epsilon: f32,
    /// [Weight decay](WeightDecayConfig) config.
    weight_decay: Option<WeightDecayConfig>,
    /// [Gradient Clipping](GradientClippingConfig) config.
    grad_clipping: Option<GradientClippingConfig>,
}

/// AdaGrad optimizer
#[derive(Clone)]
pub struct AdaGrad {
    lr_decay: LrDecay,
    weight_decay: Option<WeightDecay>,
}

/// AdaGrad state.
#[derive(Record, Clone, new)]
pub struct AdaGradState<B: Backend, const D: usize> {
    lr_decay: LrDecayState<B, D>,
}

impl<B: Backend> SimpleOptimizer<B> for AdaGrad {
    type State<const D: usize> = AdaGradState<B, D>;

    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<B, D>,
        mut grad: Tensor<B, D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<B, D>, Option<Self::State<D>>) {
        let mut state_lr_decay = None;

        if let Some(state) = state {
            state_lr_decay = Some(state.lr_decay);
        }

        if let Some(weight_decay) = &self.weight_decay {
            grad = weight_decay.transform(grad, tensor.clone());
        }

        let (grad, state_lr_decay) = self.lr_decay.transform(grad, lr, state_lr_decay);

        let state = AdaGradState::new(state_lr_decay);

        (tensor - grad, Some(state))
    }

    fn to_device<const D: usize>(mut state: Self::State<D>, device: &Device<B>) -> Self::State<D> {
        state.lr_decay = state.lr_decay.to_device(device);
        state
    }
}

impl AdaGradConfig {
    /// Initialize AdaGrad optimizer.
    ///
    /// # Returns
    ///
    /// Returns an optimizer that can be used to optimize a module.
    pub fn init<B: AutodiffBackend, M: AutodiffModule<B>>(
        &self,
    ) -> OptimizerAdaptor<AdaGrad, M, B> {
        let optim = AdaGrad {
            lr_decay: LrDecay {
                lr_decay: self.lr_decay,
                epsilon: self.epsilon,
            },
            weight_decay: self.weight_decay.as_ref().map(WeightDecay::new),
        };

        let mut optim = OptimizerAdaptor::from(optim);
        if let Some(config) = &self.grad_clipping {
            optim = optim.with_grad_clipping(config.init());
        }
        optim
    }
}

/// Learning rate decay state (also includes sum state).
#[derive(Record, new, Clone)]
pub struct LrDecayState<B: Backend, const D: usize> {
    time: usize,
    sum: Tensor<B, D>,
}

#[derive(Clone)]
struct LrDecay {
    lr_decay: f64,
    epsilon: f32,
}

impl LrDecay {
    pub fn transform<B: Backend, const D: usize>(
        &self,
        grad: Tensor<B, D>,
        lr: LearningRate,
        lr_decay_state: Option<LrDecayState<B, D>>,
    ) -> (Tensor<B, D>, LrDecayState<B, D>) {
        let state = if let Some(mut state) = lr_decay_state {
            state.sum = state.sum.add(grad.clone().square());
            state.time += 1;
            state
        } else {
            LrDecayState::new(1, grad.clone().square())
        };

        let new_lr = lr / (1. + (state.time as f64 - 1.) * self.lr_decay);

        let grad = grad
            .div(state.sum.clone().sqrt().add_scalar(self.epsilon))
            .mul_scalar(new_lr);

        (grad, state)
    }
}

impl<B: Backend, const D: usize> LrDecayState<B, D> {
    /// Move state to device.
    ///
    /// # Arguments
    ///
    /// * `device` - Device to move state to.
    ///
    /// # Returns
    ///
    /// Returns state moved to device.
    pub fn to_device(mut self, device: &B::Device) -> Self {
        self.sum = self.sum.to_device(device);
        self
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tolerance;
    use burn::tensor::ops::FloatElem;

    use super::*;
    use crate::TestAutodiffBackend;
    use crate::{GradientsParams, Optimizer};
    use burn::module::{Module, Param};
    use burn::tensor::{Distribution, Tensor, TensorData};
    use burn_nn::{Linear, LinearConfig, LinearRecord};

    const LEARNING_RATE: LearningRate = 0.01;

    #[test]
    fn test_adagrad_optimizer_save_load_state() {
        let device = Default::default();
        let linear = LinearConfig::new(6, 6).init(&device);
        let x = Tensor::<TestAutodiffBackend, 2>::random([2, 6], Distribution::Default, &device);
        let mut optimizer = create_adagrad();
        let grads = linear.forward(x).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let _linear = optimizer.step(LEARNING_RATE, linear, grads);

        #[cfg(feature = "std")]
        {
            use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};

            BinFileRecorder::<FullPrecisionSettings>::default()
                .record(
                    optimizer.to_record(),
                    std::env::temp_dir().as_path().join("test_optim_adagrad"),
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
        let optimizer = create_adagrad();
        let optimizer = optimizer.load_record(state_optim_before_copy);
        let state_optim_after = optimizer.to_record();

        assert_eq!(state_optim_before.len(), state_optim_after.len());
    }

    #[test]
    fn test_adagrad_optimizer_with_numbers() {
        let device = Default::default();
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

        let mut optimizer = AdaGradConfig::new()
            .with_epsilon(1e-8)
            .with_lr_decay(0.5)
            .init();

        let grads = linear.forward(x_1).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let grads = linear.forward(x_2).backward();
        let grads = GradientsParams::from_grads(grads, &linear);
        let linear = optimizer.step(LEARNING_RATE, linear, grads);

        let state_updated = linear.into_record();
        let weights_expected = TensorData::from([
            [-0.334989, 0.123011, 0.389911, 0.305611, 0.071511, 0.052711],
            [
                0.066144, -0.030056, -0.378256, 0.243444, 0.183944, -0.303756,
            ],
            [
                -0.033462, 0.020138, -0.310662, 0.233938, -0.292462, 0.298538,
            ],
            [
                -0.312636, -0.236036, -0.386136, -0.312736, -0.090736, 0.147964,
            ],
            [
                0.315896, -0.232304, 0.357596, -0.187004, 0.365496, -0.044504,
            ],
            [-0.030305, -0.026405, 0.111395, 0.177695, 0.014895, 0.368895],
        ]);
        let bias_expected = TensorData::from([
            -0.405214, 0.073686, -0.111714, 0.102886, 0.121886, -0.001714,
        ]);

        let (weight_updated, bias_updated) = (
            state_updated.weight.val().into_data(),
            state_updated.bias.unwrap().val().into_data(),
        );

        type FT = FloatElem<TestAutodiffBackend>;
        let tolerance = Tolerance::absolute(1e-6);
        bias_updated.assert_approx_eq::<FT>(&bias_expected, tolerance);
        weight_updated.assert_approx_eq::<FT>(&weights_expected, tolerance);
    }

    fn given_linear_layer(weight: TensorData, bias: TensorData) -> Linear<TestAutodiffBackend> {
        let device = Default::default();
        let record = LinearRecord {
            weight: Param::from_data(weight, &device),
            bias: Some(Param::from_data(bias, &device)),
        };

        LinearConfig::new(6, 6).init(&device).load_record(record)
    }

    fn create_adagrad()
    -> OptimizerAdaptor<AdaGrad, Linear<TestAutodiffBackend>, TestAutodiffBackend> {
        let config = AdaGradConfig::new();
        AdaGrad {
            lr_decay: LrDecay {
                lr_decay: config.lr_decay,
                epsilon: config.epsilon,
            },
            weight_decay: config.weight_decay.as_ref().map(WeightDecay::new),
        }
        .into()
    }
}
