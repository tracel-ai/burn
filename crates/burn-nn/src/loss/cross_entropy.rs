use burn_core as burn;
use burn_core::tensor::IndexingUpdateOp;

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::activation::log_softmax;
use burn::tensor::{Bool, Device, Int, Tensor, assert_shape};
use burn::{config::Config, module::Module};

#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::Float;

/// Configuration to create a [Cross-entropy loss](CrossEntropyLoss) using the [init function](CrossEntropyLossConfig::init).
#[derive(Config, Debug)]
pub struct CrossEntropyLossConfig {
    /// Create padded cross entropy.
    ///
    /// Prevents pad tokens from impacting loss calculation.
    ///
    /// An entirely padded batch returns NaN, since the mean has no valid samples.
    pub pad_tokens: Option<Vec<usize>>,
    /// Target value that marks a position to ignore, like PyTorch's `ignore_index`.
    ///
    /// Positions whose target equals this value are excluded from both the loss and the
    /// normalizer, together with any `pad_tokens`. Unlike `pad_tokens` it may be negative, so the
    /// `-100` sentinel used by PyTorch and ONNX can be passed as is.
    ///
    /// Nothing is ignored by default: the sentinel must be set here for its positions to be
    /// excluded. Without it, a `-100` target is treated like any other out-of-range target, which
    /// is not validated.
    ///
    /// An entirely ignored batch returns NaN, since the mean has no valid samples.
    pub ignore_index: Option<i64>,
    /// Create weighted cross-entropy.
    ///
    /// The loss of a specific sample will simply be given by: weight * log(p(x)) * 1,
    ///
    /// # Pre-conditions
    ///   - The order of the weight vector should correspond to the label integer assignment.
    ///   - Targets assigned negative Int's will not be allowed, except for `ignore_index`.
    pub weights: Option<Vec<f32>>,

    /// Create cross-entropy with label smoothing.
    ///
    /// Hard labels {0, 1} will be changed to y_smoothed = y(1 - a) + a / nr_classes.
    /// Alpha = 0 would be the same as default.
    pub smoothing: Option<f32>,

    /// Create cross-entropy with probabilities as input instead of logits.
    ///
    #[config(default = true)]
    pub logits: bool,
}

impl CrossEntropyLossConfig {
    /// Initialize [Cross-entropy loss](CrossEntropyLoss).
    pub fn init(&self, device: &Device) -> CrossEntropyLoss {
        self.assertions();
        CrossEntropyLoss {
            pad_tokens: self.pad_tokens.clone(),
            ignore_index: self.ignore_index,
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

/// Calculate the cross entropy loss from the input logits and the targets.
///
/// Should be created using [CrossEntropyLossConfig]
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct CrossEntropyLoss {
    /// Pad tokens to ignore in the loss calculation.
    pub pad_tokens: Option<Vec<usize>>,
    /// Target value to ignore in the loss calculation.
    pub ignore_index: Option<i64>,
    /// Weights for cross-entropy.
    pub weights: Option<Tensor<1>>,
    /// Label smoothing factor.
    pub smoothing: Option<f32>,
    /// Use logits as input.
    pub logits: bool,
}

impl ModuleDisplay for CrossEntropyLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        let pad_tokens = if let Some(pad_tokens) = &self.pad_tokens {
            alloc::format!("Vec<0..{}>", pad_tokens.len())
        } else {
            "None".to_string()
        };

        content
            .add("pad_tokens", &pad_tokens)
            .add("ignore_index", &self.ignore_index)
            .add("weights", &self.weights)
            .add("smoothing", &self.smoothing)
            .add("logits", &self.logits)
            .optional()
    }
}

impl CrossEntropyLoss {
    /// For backward compatibility.
    pub fn new(pad_index: Option<usize>, device: &Device) -> Self {
        CrossEntropyLossConfig::new()
            .with_pad_tokens(pad_index.map(|e| vec![e]))
            .init(device)
    }

    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// - logits: `[batch_size, num_targets]`
    /// - targets: `[batch_size]`
    pub fn forward(&self, logits: Tensor<2>, targets: Tensor<1, Int>) -> Tensor<1> {
        Self::assertions(logits.clone(), targets.clone());
        let mask = self.padding_mask(&targets);
        let targets = self.remap_ignored(targets);
        match self.smoothing {
            Some(alpha) => self.forward_smoothed(logits, targets, mask, alpha),
            _ => self.forward_default(logits, targets, mask),
        }
    }

    fn forward_smoothed(
        &self,
        logits: Tensor<2>,
        targets: Tensor<1, Int>,
        mask: Option<Tensor<1, Bool>>,
        alpha: f32,
    ) -> Tensor<1> {
        let tensor = if self.logits {
            log_softmax(logits, 1)
        } else {
            Self::clamp_probs(logits).log()
        };
        let [batch_size, nr_classes] = tensor.dims();
        let tensor = tensor
            * Self::compute_smoothed_targets([batch_size, nr_classes], targets.clone(), alpha);

        let (tensor, weights) = match &self.weights {
            Some(weights) => {
                let tensor = tensor
                    * weights
                        .clone()
                        .reshape([1, nr_classes])
                        .repeat_dim(0, batch_size);
                let weights = weights.clone().gather(0, targets);
                (tensor, Some(weights))
            }
            None => (tensor, None),
        };

        let tensor = tensor.sum_dim(1).squeeze_dim::<1>(1);
        Self::reduce_mean(tensor, weights, mask)
    }

    fn forward_default(
        &self,
        logits: Tensor<2>,
        targets: Tensor<1, Int>,
        mask: Option<Tensor<1, Bool>>,
    ) -> Tensor<1> {
        let [batch_size] = targets.dims();

        let target_indices = targets.clone().reshape([batch_size, 1]);
        let tensor = if self.logits {
            log_softmax(logits, 1).gather(1, target_indices)
        } else {
            Self::clamp_probs(logits).gather(1, target_indices).log()
        };

        let tensor = tensor.reshape([batch_size]);
        let (tensor, weights) = match &self.weights {
            Some(weights) => {
                let weights = weights.clone().gather(0, targets);
                (tensor * weights.clone(), Some(weights))
            }
            None => (tensor, None),
        };

        Self::reduce_mean(tensor, weights, mask)
    }

    /// Clamps probabilities away from zero so their log stays finite.
    fn clamp_probs(probs: Tensor<2>) -> Tensor<2> {
        // TODO: finfo stable eps
        let finfo = probs.dtype().finfo().unwrap();
        let eps = finfo.min_positive.sqrt();
        probs.clamp_min(eps)
    }

    fn compute_smoothed_targets(
        shape: [usize; 2],
        targets: Tensor<1, Int>,
        alpha: f32,
    ) -> Tensor<2> {
        let [batch_size, nr_classes] = shape;
        let device = &targets.device();
        let targets_matrix = Tensor::<2>::zeros(shape, device).scatter(
            1,
            targets.reshape([batch_size, 1]),
            Tensor::ones([batch_size, 1], device),
            IndexingUpdateOp::Add,
        );
        targets_matrix * (1. - alpha) + alpha / nr_classes as f32
    }

    fn padding_mask(&self, targets: &Tensor<1, Int>) -> Option<Tensor<1, Bool>> {
        self.pad_tokens
            .iter()
            .flatten()
            .map(|token| *token as i64)
            .chain(self.ignore_index)
            .map(|value| targets.clone().equal_scalar(value))
            .reduce(|mask, value_mask| mask.bool_or(value_mask))
    }

    /// Replaces ignored targets with class 0 so they stay in bounds for `gather` and `scatter`.
    ///
    /// Their contribution is removed afterwards by the padding mask.
    fn remap_ignored(&self, targets: Tensor<1, Int>) -> Tensor<1, Int> {
        match self.ignore_index {
            Some(ignore_index) => {
                let ignored = targets.clone().equal_scalar(ignore_index);
                targets.mask_fill(ignored, 0)
            }
            None => targets,
        }
    }

    fn reduce_mean(
        tensor: Tensor<1>,
        weights: Option<Tensor<1>>,
        mask: Option<Tensor<1, Bool>>,
    ) -> Tensor<1> {
        if weights.is_none() && mask.is_none() {
            return tensor.mean().neg();
        }

        let normalizer = weights.unwrap_or_else(|| tensor.ones_like());
        let tensor = Self::apply_mask(tensor, mask.clone());
        let normalizer = Self::apply_mask(normalizer, mask);

        tensor.sum().neg() / normalizer.sum()
    }

    fn apply_mask(mut tensor: Tensor<1>, mask: Option<Tensor<1, Bool>>) -> Tensor<1> {
        if let Some(mask) = mask {
            tensor = tensor.mask_fill(mask, 0);
        }

        tensor
    }

    fn assertions(logits: Tensor<2>, targets: Tensor<1, Int>) {
        let [batch_size, _] = logits.dims();
        assert_shape!(targets, [batch_size]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tolerance;
    use burn::tensor::{Distribution, TensorData, loss::cross_entropy_with_logits};
    type FT = f32;

    macro_rules! setup {
        () => {{
            let [batch_size, num_targets] = [4, 5];
            let device = Default::default();
            let logits = Tensor::<2>::random(
                [batch_size, num_targets],
                Distribution::Normal(0., 1.0),
                &device,
            );
            let targets = Tensor::<1, Int>::from_data(TensorData::from([2, 0, 4, 1]), &device);
            let targets_logits = Tensor::<2>::from_data(
                TensorData::from([
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                ]),
                &device,
            );
            (logits, targets, targets_logits)
        }};
    }

    macro_rules! setup_padded {
        () => {{
            let [batch_size, num_targets, pad_index] = [4, 5, 1];
            let device = Default::default();
            let logits = Tensor::<2>::random(
                [batch_size, num_targets],
                Distribution::Normal(0., 1.0),
                &device,
            );
            let targets =
                Tensor::<1, Int>::from_data(TensorData::from([2, 0, 4, pad_index as i64]), &device);
            let targets_logits = Tensor::<2>::from_data(
                TensorData::from([
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ]),
                &device,
            );
            (logits, targets, targets_logits)
        }};
    }

    #[test]
    #[should_panic(expected = "assert_shape!(targets, [batch_size]): axis 0 expected 4, got 3")]
    fn test_cross_entropy_loss_targets_must_match_batch_size() {
        let (logits, _, _) = setup!();
        let device = logits.device();
        let targets = Tensor::<1, Int>::from_data(TensorData::from([2, 0, 4]), &device);
        let _ = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits, targets);
    }

    #[test]
    fn test_cross_entropy_loss_with_weights() {
        let (logits, targets, targets_logits) = setup!();
        let weights = vec![1.0, 2., 3., 4., 5.];
        let device = Default::default();
        let loss_1 = CrossEntropyLossConfig::new()
            .with_weights(Some(weights.clone()))
            .init(&device)
            .forward(logits.clone(), targets);
        let tensor = log_softmax(logits, 1);
        let loss_2 = tensor
            * targets_logits
            * Tensor::<1>::from_floats(weights.as_slice(), &device)
                .unsqueeze()
                .repeat_dim(0, 4);
        let loss_2 = loss_2.sum().neg() / (1. + 2. + 3. + 5.);
        loss_1
            .into_data()
            .assert_approx_eq::<FT>(&loss_2.into_data(), Tolerance::default());
    }

    #[test]
    fn test_label_smoothing_with_weights_and_alpha_zero() {
        let (logits, targets, _) = setup!();
        let device = Default::default();
        let weights = vec![1.0, 2., 3., 4., 5.];
        let loss_1 = CrossEntropyLossConfig::new()
            .with_weights(Some(weights.clone()))
            .init(&device)
            .forward(logits.clone(), targets.clone());
        let loss_2 = CrossEntropyLossConfig::new()
            .with_weights(Some(weights.clone()))
            .with_smoothing(Some(0.))
            .init(&device)
            .forward(logits.clone(), targets);
        loss_1
            .into_data()
            .assert_approx_eq::<FT>(&loss_2.into_data(), Tolerance::default());
    }

    #[test]
    fn test_cross_entropy_loss() {
        let (logits, targets, targets_logits) = setup!();
        let device = Default::default();
        let loss_1 = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits.clone(), targets);
        let loss_2 = cross_entropy_with_logits(logits, targets_logits);

        loss_1
            .into_data()
            .assert_approx_eq::<FT>(&loss_2.into_data(), Tolerance::default());
    }

    #[test]
    fn test_label_smoothing_alpha_equal_zero() {
        let (logits, targets, _) = setup!();
        let device = Default::default();
        let loss_1 = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits.clone(), targets.clone());
        let loss_2 = CrossEntropyLossConfig::new()
            .with_smoothing(Some(0.))
            .init(&device)
            .forward(logits, targets);

        loss_1
            .into_data()
            .assert_approx_eq::<FT>(&loss_2.into_data(), Tolerance::default());
    }

    #[test]
    fn test_cross_entropy_loss_with_pad_token() {
        let (logits, targets, targets_logits) = setup_padded!();
        let device = logits.device();
        let pad_index = 1;

        let loss_1 = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![pad_index, 2]))
            .init(&device)
            .forward(logits.clone(), targets);
        let valid_indices = Tensor::<1, Int>::from_data(TensorData::from([1i64, 2]), &device);
        let loss_2 = cross_entropy_with_logits(
            logits.select(0, valid_indices.clone()),
            targets_logits.select(0, valid_indices),
        );

        loss_1
            .into_data()
            .assert_approx_eq::<FT>(&loss_2.into_data(), Tolerance::default());
    }

    #[test]
    fn test_label_smoothing_with_zero_alpha_and_pad_token() {
        let (logits, targets, _) = setup_padded!();
        let pad_index = 1;

        let loss_1 = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![pad_index, 2]))
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());
        let loss_2 = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![pad_index, 2]))
            .with_smoothing(Some(0.))
            .init(&logits.device())
            .forward(logits.clone(), targets);

        loss_1
            .into_data()
            .assert_approx_eq::<FT>(&loss_2.into_data(), Tolerance::default());
    }

    #[test]
    fn test_label_smoothing_target_conversion() {
        let (logits, targets, _) = setup!();
        let smoothed_targets =
            CrossEntropyLoss::compute_smoothed_targets(logits.dims(), targets, 0.05);
        let targets_logits = Tensor::<2>::from_data(
            TensorData::from([
                [0.01, 0.01, 0.96, 0.01, 0.01],
                [0.96, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.96],
                [0.01, 0.96, 0.01, 0.01, 0.01],
            ]),
            &Default::default(),
        );
        smoothed_targets
            .into_data()
            .assert_approx_eq::<FT>(&targets_logits.into_data(), Tolerance::default());
    }

    #[test]
    fn test_label_smoothing() {
        let (logits, targets, _) = setup!();
        let device = Default::default();
        let loss_1 = CrossEntropyLossConfig::new()
            .with_smoothing(Some(0.05))
            .init(&device)
            .forward(logits.clone(), targets);
        let targets_logits = Tensor::<2>::from_data(
            TensorData::from([
                [0.01, 0.01, 0.96, 0.01, 0.01],
                [0.96, 0.01, 0.01, 0.01, 0.01],
                [0.01, 0.01, 0.01, 0.01, 0.96],
                [0.01, 0.96, 0.01, 0.01, 0.01],
            ]),
            &device,
        );

        let x = log_softmax(logits, 1);
        let loss_2 = (x * targets_logits).sum_dim(1).mean().neg();

        loss_1
            .into_data()
            .assert_approx_eq::<FT>(&loss_2.into_data(), Tolerance::default());
    }

    #[test]
    fn test_logits_flag_affects_output() {
        let device = Default::default();

        let probs = Tensor::<2>::from_data(
            TensorData::from([
                [0.1, 0.2, 0.7, 0.0, 0.0],
                [0.7, 0.1, 0.1, 0.1, 0.0],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.0, 0.3, 0.3, 0.2, 0.2],
            ]),
            &device,
        );

        let targets = Tensor::<1, Int>::from_data(TensorData::from([2, 0, 4, 1]), &device);

        let loss_logits = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(probs.clone(), targets.clone());

        let loss_probs = CrossEntropyLossConfig::new()
            .with_logits(false)
            .init(&device)
            .forward(probs, targets);

        // They must differ if logits flag is implemented correctly
        let loss_logits = loss_logits.into_data();
        let loss_probs = loss_probs.into_data();

        loss_logits.assert_approx_eq::<f32>(&TensorData::from([1.354197]), Tolerance::default());
        loss_probs.assert_approx_eq::<f32>(&TensorData::from([0.88169014]), Tolerance::default());

        assert_ne!(
            loss_logits.as_slice::<f32>().unwrap(),
            loss_probs.as_slice::<f32>().unwrap(),
            "logits flag should change computation (log_softmax vs log)"
        );
    }

    #[test]
    fn test_label_smoothing_with_zero_probabilities() {
        let device = Default::default();

        let probs = Tensor::<2>::from_data(
            TensorData::from([
                [0.1, 0.2, 0.7, 0.0, 0.0],
                [0.7, 0.1, 0.1, 0.1, 0.0],
                [0.2, 0.2, 0.2, 0.2, 0.2],
                [0.0, 0.3, 0.3, 0.2, 0.2],
            ]),
            &device,
        );
        let targets = Tensor::<1, Int>::from_data(TensorData::from([2, 0, 4, 1]), &device);

        let loss_default = CrossEntropyLossConfig::new()
            .with_logits(false)
            .init(&device)
            .forward(probs.clone(), targets.clone());
        let loss_no_smoothing = CrossEntropyLossConfig::new()
            .with_logits(false)
            .with_smoothing(Some(0.0))
            .init(&device)
            .forward(probs.clone(), targets.clone());
        let loss_smoothed = CrossEntropyLossConfig::new()
            .with_logits(false)
            .with_smoothing(Some(0.1))
            .init(&device)
            .forward(probs, targets);

        // Zero probabilities are clamped like in the non-smoothed path, so alpha = 0
        // matches it and alpha > 0 stays finite.
        loss_no_smoothing
            .into_data()
            .assert_approx_eq::<FT>(&loss_default.into_data(), Tolerance::default());
        loss_smoothed
            .into_data()
            .assert_approx_eq::<FT>(&TensorData::from([1.7929223]), Tolerance::default());
    }

    fn assert_padding_invariant(weights: Option<Vec<f32>>, smoothing: Option<f32>, logits: bool) {
        let device = Default::default();
        let valid_input = Tensor::<2>::from_data(
            TensorData::from([[0.7f32, 0.2, 0.1], [0.1, 0.7, 0.2]]),
            &device,
        );
        let valid_targets = Tensor::<1, Int>::from_data(TensorData::from([0i64, 1]), &device);
        let padded_input = Tensor::<2>::from_data(
            TensorData::from([
                [0.7f32, 0.2, 0.1],
                [0.1, 0.7, 0.2],
                [0.2, 0.3, 0.5],
                [0.3, 0.2, 0.5],
            ]),
            &device,
        );
        let padded_targets =
            Tensor::<1, Int>::from_data(TensorData::from([0i64, 1, 2, 2]), &device);

        let create_loss = || {
            CrossEntropyLossConfig::new()
                .with_pad_tokens(Some(vec![2]))
                .with_weights(weights.clone())
                .with_smoothing(smoothing)
                .with_logits(logits)
                .init(&device)
        };
        let expected = create_loss().forward(valid_input, valid_targets);
        let actual = create_loss().forward(padded_input, padded_targets);

        actual
            .into_data()
            .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    fn padding_does_not_change_default_loss() {
        assert_padding_invariant(None, None, true);
    }

    #[test]
    fn padding_does_not_change_weighted_loss() {
        assert_padding_invariant(Some(vec![1.0, 2.0, 4.0]), None, true);
    }

    #[test]
    fn padding_does_not_change_smoothed_loss() {
        assert_padding_invariant(None, Some(0.1), true);
    }

    #[test]
    fn padding_does_not_change_smoothed_weighted_loss() {
        assert_padding_invariant(Some(vec![1.0, 2.0, 4.0]), Some(0.1), true);
    }

    #[test]
    fn padding_does_not_change_probability_loss() {
        assert_padding_invariant(None, None, false);
    }

    #[test]
    fn multiple_pad_tokens_are_excluded_from_normalization() {
        let device = Default::default();
        let valid_input = Tensor::<2>::from_data(TensorData::from([[0.7f32, 0.2, 0.1]]), &device);
        let valid_targets = Tensor::<1, Int>::from_data(TensorData::from([0i64]), &device);
        let padded_input = Tensor::<2>::from_data(
            TensorData::from([[0.7f32, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]]),
            &device,
        );
        let padded_targets = Tensor::<1, Int>::from_data(TensorData::from([0i64, 1, 2]), &device);

        let create_loss = || {
            CrossEntropyLossConfig::new()
                .with_pad_tokens(Some(vec![1, 2]))
                .init(&device)
        };
        let expected = create_loss().forward(valid_input, valid_targets);
        let actual = create_loss().forward(padded_input, padded_targets);

        actual
            .into_data()
            .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    fn empty_pad_token_list_behaves_like_no_padding() {
        let device = Default::default();
        let input = Tensor::<2>::from_data(
            TensorData::from([[0.7f32, 0.2, 0.1], [0.1, 0.7, 0.2]]),
            &device,
        );
        let targets = Tensor::<1, Int>::from_data(TensorData::from([0i64, 1]), &device);

        let expected = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(input.clone(), targets.clone());
        let actual = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![]))
            .init(&device)
            .forward(input, targets);

        actual
            .into_data()
            .assert_approx_eq::<FT>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    fn entirely_padded_batch_returns_nan() {
        let device = Default::default();
        let input = Tensor::<2>::from_data(
            TensorData::from([[0.7f32, 0.2, 0.1], [0.1, 0.2, 0.7]]),
            &device,
        );
        let targets = Tensor::<1, Int>::from_data(TensorData::from([2i64, 2]), &device);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![2]))
            .init(&device)
            .forward(input, targets)
            .into_scalar::<FT>();

        assert!(loss.is_nan());
    }

    #[test]
    fn ignore_index_excludes_position_from_numerator_and_denominator() {
        let device = Default::default();
        let logits = Tensor::<2>::from_data(
            TensorData::from([[0.0f32, 1.0, 0.5, 0.25], [2.0, 0.0, 0.0, 0.0]]),
            &device,
        );
        let targets = Tensor::<1, Int>::from_data(TensorData::from([1i64, -100]), &device);

        let loss = CrossEntropyLossConfig::new()
            .with_ignore_index(Some(-100))
            .init(&device)
            .forward(logits, targets);

        // Loss of the first row alone.
        loss.into_data()
            .assert_approx_eq::<FT>(&TensorData::from([0.894_771_5_f32]), Tolerance::default());
    }

    #[test]
    fn ignore_index_matches_selecting_valid_rows() {
        let (logits, _, _) = setup!();
        let device = logits.device();
        let targets = Tensor::<1, Int>::from_data(TensorData::from([2i64, -100, 4, 1]), &device);
        let valid_indices = Tensor::<1, Int>::from_data(TensorData::from([0i64, 2, 3]), &device);
        let valid_targets = Tensor::<1, Int>::from_data(TensorData::from([2i64, 4, 1]), &device);
        let weights = vec![1.0, 2., 3., 4., 5.];

        for smoothing in [None, Some(0.1)] {
            for weights in [None, Some(weights.clone())] {
                let config = CrossEntropyLossConfig::new()
                    .with_smoothing(smoothing)
                    .with_weights(weights);
                let loss_1 = config
                    .clone()
                    .with_ignore_index(Some(-100))
                    .init(&device)
                    .forward(logits.clone(), targets.clone());
                let loss_2 = config.init(&device).forward(
                    logits.clone().select(0, valid_indices.clone()),
                    valid_targets.clone(),
                );

                loss_1
                    .into_data()
                    .assert_approx_eq::<FT>(&loss_2.into_data(), Tolerance::default());
            }
        }
    }

    #[test]
    fn ignore_index_and_pad_tokens_exclude_their_union() {
        let (logits, _, _) = setup!();
        let device = logits.device();
        let targets = Tensor::<1, Int>::from_data(TensorData::from([2i64, -1, 4, 1]), &device);
        let valid_indices = Tensor::<1, Int>::from_data(TensorData::from([0i64, 3]), &device);
        let valid_targets = Tensor::<1, Int>::from_data(TensorData::from([2i64, 1]), &device);

        let loss_1 = CrossEntropyLossConfig::new()
            .with_ignore_index(Some(-1))
            .with_pad_tokens(Some(vec![4]))
            .init(&device)
            .forward(logits.clone(), targets);
        let loss_2 = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits.select(0, valid_indices), valid_targets);

        loss_1
            .into_data()
            .assert_approx_eq::<FT>(&loss_2.into_data(), Tolerance::default());
    }

    #[test]
    fn entirely_ignored_batch_returns_nan() {
        let device = Default::default();
        let input = Tensor::<2>::from_data(
            TensorData::from([[0.7f32, 0.2, 0.1], [0.1, 0.2, 0.7]]),
            &device,
        );
        let targets = Tensor::<1, Int>::from_data(TensorData::from([-100i64, -100]), &device);

        let loss = CrossEntropyLossConfig::new()
            .with_ignore_index(Some(-100))
            .init(&device)
            .forward(input, targets)
            .into_scalar::<FT>();

        assert!(loss.is_nan());
    }

    #[cfg(feature = "std")]
    #[test]
    fn ignored_targets_have_zero_gradient() {
        let device = Device::default().autodiff();
        let logits = Tensor::<2>::from_data(
            TensorData::from([[2.0f32, 0.0, -1.0], [0.0, 0.0, 0.0]]),
            &device,
        )
        .require_grad();
        let targets = Tensor::<1, Int>::from_data(TensorData::from([0i64, -100]), &device);

        let loss = CrossEntropyLossConfig::new()
            .with_ignore_index(Some(-100))
            .init(&device)
            .forward(logits.clone(), targets);
        let grads = loss.backward();
        let grads_logits = logits.grad(&grads).unwrap();

        let expected = TensorData::from([[-0.1562053f32, 0.1141952, 0.04201007], [0.0, 0.0, 0.0]]);
        grads_logits
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::relative(1e-4));
    }

    #[cfg(feature = "std")]
    #[test]
    fn padded_targets_have_zero_gradient_without_scaling_valid_gradients() {
        let device = Device::default().autodiff();
        let logits = Tensor::<2>::from_data(
            TensorData::from([[2.0f32, 0.0, -1.0], [0.0, 0.0, 0.0]]),
            &device,
        )
        .require_grad();
        let targets = Tensor::<1, Int>::from_data(TensorData::from([0i64, 2]), &device);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![2]))
            .init(&device)
            .forward(logits.clone(), targets);
        let grads = loss.backward();
        let grads_logits = logits.grad(&grads).unwrap();

        let expected = TensorData::from([[-0.1562053f32, 0.1141952, 0.04201007], [0.0, 0.0, 0.0]]);
        grads_logits
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::relative(1e-4));
    }

    #[test]
    fn display() {
        let config = CrossEntropyLossConfig::new()
            .with_weights(Some(alloc::vec![3., 7., 0.9]))
            .with_smoothing(Some(0.5));
        let loss = config.init(&Default::default());

        assert_eq!(
            alloc::format!("{loss}"),
            "CrossEntropyLoss {pad_tokens: None, ignore_index: None, weights: Tensor {rank: 1, shape: [3]}, smoothing: 0.5, logits: true}"
        );
    }
}
