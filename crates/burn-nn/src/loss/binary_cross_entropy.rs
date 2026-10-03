use burn_core as burn;

use alloc::vec::Vec;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::activation::log_sigmoid;
use burn::tensor::{Device, Int, Tensor};
use burn::{config::Config, module::Module};

/// Configuration to create a [Binary Cross-entropy loss](BinaryCrossEntropyLoss) using the [init function](BinaryCrossEntropyLossConfig::init).
#[derive(Config, Debug)]
pub struct BinaryCrossEntropyLossConfig {
    /// Create weighted binary cross-entropy with a weight for each class.
    ///
    /// The loss of a specific sample will simply be multiplied by its label weight.
    pub weights: Option<Vec<f32>>,

    /// Create binary cross-entropy with label smoothing according to [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629).
    ///
    /// Hard labels {0, 1} will be changed to `y_smoothed = y(1 - a) + a / num_classes`.
    /// Alpha = 0 would be the same as default.
    pub smoothing: Option<f32>,

    /// Treat the inputs as logits, applying a sigmoid activation when computing the loss.
    #[config(default = false)]
    pub logits: bool,
}

impl BinaryCrossEntropyLossConfig {
    /// Initialize [Binary Cross-entropy loss](BinaryCrossEntropyLoss).
    pub fn init(&self, device: &Device) -> BinaryCrossEntropyLoss {
        self.assertions();
        BinaryCrossEntropyLoss {
            weights: self
                .weights
                .as_ref()
                .map(|e| Tensor::<1>::from_floats(e.as_slice(), device)),
            smoothing: self.smoothing,
            logits: self.logits,
        }
    }

    fn assertions(&self) {
        if let Some(alpha) = self.smoothing {
            assert!(
                (0.0..=1.).contains(&alpha),
                "Alpha of Cross-entropy loss with smoothed labels should be in interval [0, 1]. Got {alpha}"
            );
        };
        if let Some(weights) = self.weights.as_ref() {
            assert!(
                weights.iter().all(|e| e > &0.),
                "Weights of cross-entropy have to be positive."
            );
        }
    }
}

/// Calculate the binary cross entropy loss from input probabilities or logits and binary targets.
///
/// Should be created using [BinaryCrossEntropyLossConfig]
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct BinaryCrossEntropyLoss {
    /// Weights for cross-entropy.
    pub weights: Option<Tensor<1>>,
    /// Label smoothing alpha.
    pub smoothing: Option<f32>,
    /// Treat the inputs as logits
    pub logits: bool,
}

impl ModuleDisplay for BinaryCrossEntropyLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("weights", &self.weights)
            .add("smoothing", &self.smoothing)
            .add("logits", &self.logits)
            .optional()
    }
}

impl BinaryCrossEntropyLoss {
    /// Compute the criterion on the input tensor.
    ///
    /// Targets must be binary labels (`0` or `1`), before applying label smoothing.
    /// When [logits](Self::logits) is `false`, inputs must be finite probabilities in `[0, 1]`.
    /// When it is `true`, inputs are logits and are not restricted to `[0, 1]`.
    ///
    /// # Shapes
    ///
    /// Binary:
    /// - logits: `[batch_size]`
    /// - targets: `[batch_size]`
    ///
    /// Multi-label:
    /// - logits: `[batch_size, num_classes]`
    /// - targets: `[batch_size, num_classes]`
    ///
    /// # Panics
    ///
    /// - If input and target shapes do not match, or multi-label weights do not match the number of classes.
    /// - If any target is not `0` or `1`.
    /// - If `logits` is `false` and any input is non-finite or outside `[0, 1]`.
    pub fn forward<const D: usize>(&self, logits: Tensor<D>, targets: Tensor<D, Int>) -> Tensor<1> {
        self.assertions(&logits, &targets);

        let mut targets_float = targets.clone().float();
        let shape = targets.dims();

        if let Some(alpha) = self.smoothing {
            let num_classes = if D > 1 { shape[D - 1] } else { 2 };
            targets_float = targets_float * (1. - alpha) + alpha / num_classes as f32;
        }

        let mut loss = if self.logits {
            // Numerically stable by combining `log(sigmoid(x))` with `log_sigmoid(x)`
            (targets_float.neg() + 1.) * logits.clone() - log_sigmoid(logits)
        } else {
            // - (target * log(input) + (1 - target) * log(1 - input))
            // https://github.com/tracel-ai/burn/issues/2739: clamp at -100.0 to avoid undefined values
            (targets_float.clone() - 1) * logits.clone().neg().log1p().clamp_min(-100.0)
                - targets_float * logits.log().clamp_min(-100.0)
        };

        if let Some(weights) = &self.weights {
            let weights = if D > 1 {
                weights.clone().expand(shape)
            } else {
                // Flatten targets and expand resulting weights to make it compatible with
                // Tensor<D> for binary 1-D case
                weights
                    .clone()
                    .gather(0, targets.flatten(0, 0))
                    .expand(shape)
            };
            loss = loss * weights;
        }

        loss.mean()
    }

    fn assertions<const D: usize>(&self, logits: &Tensor<D>, targets: &Tensor<D, Int>) {
        let logits_dims = logits.dims();
        let targets_dims = targets.dims();
        assert!(
            logits_dims == targets_dims,
            "Shape of targets ({targets_dims:?}) should correspond to outer shape of logits ({logits_dims:?})."
        );

        if let Some(weights) = &self.weights
            && D > 1
        {
            let targets_classes = targets_dims[D - 1];
            let weights_classes = weights.dims()[0];
            assert!(
                weights_classes == targets_classes,
                "The number of classes ({weights_classes}) does not match the weights provided ({targets_classes})."
            );
        }

        assert!(
            targets
                .clone()
                .greater_equal_scalar(0)
                .bool_and(targets.clone().lower_equal_scalar(1))
                .all()
                .into_scalar::<bool>(),
            "Targets must be in the interval [0, 1]."
        );

        if !self.logits {
            // Both comparisons must hold, which also rejects NaN and infinities.
            assert!(
                logits
                    .clone()
                    .greater_equal_scalar(0.0)
                    .bool_and(logits.clone().lower_equal_scalar(1.0))
                    .all()
                    .into_scalar::<bool>(),
                "Probability inputs must be finite and in the interval [0, 1]."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tolerance;
    use burn::tensor::{TensorData, activation::sigmoid};
    use rstest::rstest;
    type FT = f32;

    #[rstest]
    #[case::below_zero(-0.1)]
    #[case::above_one(1.2)]
    #[case::nan(f32::NAN)]
    #[case::positive_infinity(f32::INFINITY)]
    #[case::negative_infinity(f32::NEG_INFINITY)]
    #[should_panic(expected = "Probability inputs must be finite and in the interval [0, 1].")]
    fn invalid_probabilities_should_panic(
        #[case] invalid: f32,
        #[values(false, true)] multilabel: bool,
    ) {
        let device = Default::default();
        let loss = BinaryCrossEntropyLossConfig::new().init(&device);
        let inputs = Tensor::<1>::from_floats([0.5, invalid], &device);
        let targets = Tensor::<1, Int>::from_data([0, 1], &device);

        if multilabel {
            loss.forward(inputs.reshape([1, 2]), targets.reshape([1, 2]));
        } else {
            loss.forward(inputs, targets);
        }
    }

    #[rstest]
    #[case::below_zero(-1)]
    #[case::above_one(2)]
    #[should_panic(expected = "Targets must be in the interval [0, 1].")]
    fn invalid_targets_should_panic(
        #[case] invalid: i32,
        #[values(false, true)] logits: bool,
        #[values(false, true)] multilabel: bool,
        #[values(None, Some(1.0))] smoothing: Option<f32>,
        #[values(false, true)] weighted: bool,
    ) {
        let device = Default::default();
        let weights = weighted.then(|| alloc::vec![3.0, 7.0]);
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(logits)
            .with_smoothing(smoothing)
            .with_weights(weights)
            .init(&device);
        let inputs = Tensor::<1>::from_floats([0.5, 0.5], &device);
        let targets = Tensor::<1, Int>::from_data([0, invalid], &device);

        if multilabel {
            loss.forward(inputs.reshape([1, 2]), targets.reshape([1, 2]));
        } else {
            loss.forward(inputs, targets);
        }
    }

    #[test]
    fn logits_outside_probability_range_should_be_valid() {
        let device = Default::default();
        let inputs = Tensor::<1>::from_floats([-100.0, 100.0], &device);
        let targets = Tensor::<1, Int>::from_data([0, 1], &device);

        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&device)
            .forward(inputs, targets)
            .into_data();

        loss.assert_approx_eq::<FT>(&TensorData::from([0.0]), Tolerance::default());
    }

    #[test]
    fn probability_boundaries_with_smoothing_should_be_finite() {
        let device = Default::default();
        let inputs = Tensor::<1>::from_floats([0.0, 1.0], &device);
        let targets = Tensor::<1, Int>::from_data([0, 1], &device);

        let loss = BinaryCrossEntropyLossConfig::new()
            .with_smoothing(Some(0.1))
            .init(&device)
            .forward(inputs, targets)
            .into_data();

        loss.assert_approx_eq::<FT>(&TensorData::from([5.0]), Tolerance::default());
    }

    #[test]
    fn test_binary_cross_entropy_preds_all_correct() {
        let device = Default::default();
        let preds = Tensor::<1>::from_floats([1.0, 0.0, 1.0, 0.0], &device);
        let targets = Tensor::<1, Int>::from_data(TensorData::from([1, 0, 1, 0]), &device);

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .init(&device)
            .forward(preds, targets)
            .into_data();

        let loss_expected = TensorData::from([0.000]);
        loss_actual.assert_approx_eq::<FT>(&loss_expected, Tolerance::default());
    }

    #[test]
    fn test_binary_cross_entropy_preds_all_incorrect() {
        let device = Default::default();
        let preds = Tensor::<1>::from_floats([0.0, 1.0, 0.0, 1.0], &device);
        let targets = Tensor::<1, Int>::from_data(TensorData::from([1, 0, 1, 0]), &device);

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .init(&device)
            .forward(preds, targets)
            .into_data();

        let loss_expected = TensorData::from([100.000]); // clamped value
        loss_actual.assert_approx_eq::<FT>(&loss_expected, Tolerance::default());
    }

    #[test]
    fn test_binary_cross_entropy() {
        // import torch
        // from torch import nn
        // input = torch.tensor([0.8271, 0.9626, 0.3796, 0.2355])
        // target = torch.tensor([0., 1., 0., 1.])
        // loss = nn.BCELoss()
        // sigmoid = nn.Sigmoid()
        // out = loss(sigmoid(input), target) # tensor(0.7491)

        let device = Default::default();
        let logits = Tensor::<1>::from_floats([0.8271, 0.9626, 0.3796, 0.2355], &device);
        let targets = Tensor::<1, Int>::from_data(TensorData::from([0, 1, 0, 1]), &device);

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .init(&device)
            .forward(sigmoid(logits), targets)
            .into_data();

        let loss_expected = TensorData::from([0.7491]);
        loss_actual.assert_approx_eq::<FT>(&loss_expected, Tolerance::relative(1e-4));
    }

    #[test]
    fn test_binary_cross_entropy_with_logits() {
        let device = Default::default();
        let logits = Tensor::<1>::from_floats([0.8271, 0.9626, 0.3796, 0.2355], &device);
        let targets = Tensor::<1, Int>::from_data(TensorData::from([0, 1, 0, 1]), &device);

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&device)
            .forward(logits, targets)
            .into_data();

        let loss_expected = TensorData::from([0.7491]);
        loss_actual.assert_approx_eq::<FT>(&loss_expected, Tolerance::relative(1e-4));
    }

    #[test]
    fn test_binary_cross_entropy_with_weights() {
        // import torch
        // from torch import nn
        // input = torch.tensor([0.8271, 0.9626, 0.3796, 0.2355])
        // target = torch.tensor([0, 1, 0, 1])
        // weights = torch.tensor([3., 7.]).gather(0, target)
        // loss = nn.BCELoss(weights)
        // sigmoid = nn.Sigmoid()
        // out = loss(sigmoid(input), target.float()) # tensor(3.1531)

        let device = Default::default();
        let logits = Tensor::<1>::from_floats([0.8271, 0.9626, 0.3796, 0.2355], &device);
        let targets = Tensor::<1, Int>::from_data(TensorData::from([0, 1, 0, 1]), &device);
        let weights = [3., 7.];

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .with_weights(Some(weights.to_vec()))
            .init(&device)
            .forward(sigmoid(logits), targets)
            .into_data();

        let loss_expected = TensorData::from([3.1531]);
        loss_actual.assert_approx_eq::<FT>(&loss_expected, Tolerance::relative(1e-4));
    }

    #[test]
    fn test_binary_cross_entropy_with_smoothing() {
        // import torch
        // from torch import nn
        // input = torch.tensor([0.8271, 0.9626, 0.3796, 0.2355])
        // target = torch.tensor([0., 1., 0., 1.])
        // target_smooth = target * (1 - 0.1) + (0.1 / 2)
        // loss = nn.BCELoss()
        // sigmoid = nn.Sigmoid()
        // out = loss(sigmoid(input), target_smooth) # tensor(0.7490)

        let device = Default::default();
        let logits = Tensor::<1>::from_floats([0.8271, 0.9626, 0.3796, 0.2355], &device);
        let targets = Tensor::<1, Int>::from_data(TensorData::from([0, 1, 0, 1]), &device);

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .with_smoothing(Some(0.1))
            .init(&device)
            .forward(sigmoid(logits), targets)
            .into_data();

        let loss_expected = TensorData::from([0.7490]);
        loss_actual.assert_approx_eq::<FT>(&loss_expected, Tolerance::relative(1e-4));
    }

    #[test]
    fn test_binary_cross_entropy_multilabel() {
        // import torch
        // from torch import nn
        // input = torch.tensor([[0.5150, 0.3097, 0.7556], [0.4974, 0.9879, 0.1564]])
        // target = torch.tensor([[1., 0., 1.], [1., 0., 0.]])
        // weights = torch.tensor([3., 7., 0.9])
        // loss = nn.BCEWithLogitsLoss()
        // out = loss(input, target) # tensor(0.7112)

        let device = Default::default();
        let logits = Tensor::<2>::from_floats(
            [[0.5150, 0.3097, 0.7556], [0.4974, 0.9879, 0.1564]],
            &device,
        );
        let targets =
            Tensor::<2, Int>::from_data(TensorData::from([[1, 0, 1], [1, 0, 0]]), &device);

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&device)
            .forward(logits, targets)
            .into_data();

        let loss_expected = TensorData::from([0.7112]);
        loss_actual.assert_approx_eq::<FT>(&loss_expected, Tolerance::relative(1e-4));
    }

    #[test]
    fn test_binary_cross_entropy_multilabel_with_weights() {
        // import torch
        // from torch import nn
        // input = torch.tensor([[0.5150, 0.3097, 0.7556], [0.4974, 0.9879, 0.1564]])
        // target = torch.tensor([[1., 0., 1.], [1., 0., 0.]])
        // loss = nn.BCEWithLogitsLoss()
        // out = loss(input, target) # tensor(3.1708)

        let device = Default::default();
        let logits = Tensor::<2>::from_floats(
            [[0.5150, 0.3097, 0.7556], [0.4974, 0.9879, 0.1564]],
            &device,
        );
        let targets =
            Tensor::<2, Int>::from_data(TensorData::from([[1, 0, 1], [1, 0, 0]]), &device);
        let weights = [3., 7., 0.9];

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .with_weights(Some(weights.to_vec()))
            .init(&device)
            .forward(logits, targets)
            .into_data();

        let loss_expected = TensorData::from([3.1708]);
        loss_actual.assert_approx_eq::<FT>(&loss_expected, Tolerance::default());
    }

    #[test]
    fn test_binary_cross_entropy_multilabel_with_smoothing() {
        // import torch
        // from torch import nn
        // input = torch.tensor([[0.5150, 0.3097, 0.7556], [0.4974, 0.9879, 0.1564]])
        // target = torch.tensor([[1., 0., 1.], [1., 0., 0.]])
        // target_smooth = target * (1 - 0.1) + (0.1 / 3)
        // loss = nn.BCELoss()
        // sigmoid = nn.Sigmoid()
        // out = loss(sigmoid(input), target_smooth) # tensor(0.7228)

        let device = Default::default();
        let logits = Tensor::<2>::from_floats(
            [[0.5150, 0.3097, 0.7556], [0.4974, 0.9879, 0.1564]],
            &device,
        );
        let targets =
            Tensor::<2, Int>::from_data(TensorData::from([[1, 0, 1], [1, 0, 0]]), &device);

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .with_smoothing(Some(0.1))
            .init(&device)
            .forward(sigmoid(logits), targets)
            .into_data();

        let loss_expected = TensorData::from([0.7228]);
        loss_actual.assert_approx_eq::<FT>(&loss_expected, Tolerance::default());
    }

    #[test]
    #[should_panic = "The number of classes"]
    fn multilabel_weights_should_match_target() {
        // import torch
        // from torch import nn
        // input = torch.tensor([[0.5150, 0.3097, 0.7556], [0.4974, 0.9879, 0.1564]])
        // target = torch.tensor([[1., 0., 1.], [1., 0., 0.]])
        // loss = nn.BCEWithLogitsLoss()
        // out = loss(input, target) # tensor(3.1708)

        let device = Default::default();
        let logits = Tensor::<2>::from_floats(
            [[0.5150, 0.3097, 0.7556], [0.4974, 0.9879, 0.1564]],
            &device,
        );
        let targets =
            Tensor::<2, Int>::from_data(TensorData::from([[1, 0, 1], [1, 0, 0]]), &device);
        let weights = [3., 7.];

        let _loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .with_weights(Some(weights.to_vec()))
            .init(&device)
            .forward(logits, targets);
    }

    #[test]
    fn display() {
        let config =
            BinaryCrossEntropyLossConfig::new().with_weights(Some(alloc::vec![3., 7., 0.9]));
        let loss = config.init(&Default::default());

        assert_eq!(
            alloc::format!("{loss}"),
            "BinaryCrossEntropyLoss {weights: Tensor {rank: 1, shape: [3]}, smoothing: None, logits: false}"
        );
    }
}
