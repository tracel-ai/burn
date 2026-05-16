use burn_core as burn;
use burn_core::tensor::IndexingUpdateOp;

use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::tensor::activation::log_softmax;
use burn::tensor::{Bool, Device, Int, Tensor};
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
    pub pad_tokens: Option<Vec<usize>>,

    /// Create weighted cross-entropy.
    ///
    /// The loss of a specific sample will simply be given by: weight * log(p(x)) * 1,
    ///
    /// # Pre-conditions
    ///   - The order of the weight vector should correspond to the label integer assignment.
    ///   - Targets assigned negative Int's will not be allowed.
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
        match self.smoothing {
            Some(alpha) => self.forward_smoothed(logits, targets, alpha),
            _ => self.forward_default(logits, targets),
        }
    }

    fn forward_smoothed(
        &self,
        logits: Tensor<2>,
        targets: Tensor<1, Int>,
        alpha: f32,
    ) -> Tensor<1> {
        let mask = self.padding_mask(&targets);
        let tensor = if self.logits {
            log_softmax(logits, 1)
        } else {
            logits.log()
        };
        let [batch_size, nr_classes] = tensor.dims();
        let tensor = tensor
            * Self::compute_smoothed_targets([batch_size, nr_classes], targets.clone(), alpha);

        match &self.weights {
            Some(weights) => {
                let tensor = tensor
                    * weights
                        .clone()
                        .reshape([1, nr_classes])
                        .repeat_dim(0, batch_size);
                let weights = weights.clone().gather(0, targets);
                let tensor = Self::apply_mask_2d(tensor, mask);
                tensor.sum().neg() / weights.sum()
            }
            None => {
                let tensor = Self::apply_mask_2d(tensor, mask);
                tensor.sum_dim(1).mean().neg()
            }
        }
    }

    fn forward_default(&self, logits: Tensor<2>, targets: Tensor<1, Int>) -> Tensor<1> {
        let [batch_size] = targets.dims();

        let mask = self.padding_mask(&targets);
        let target_indices = targets.clone().reshape([batch_size, 1]);
        let tensor = if self.logits {
            log_softmax(logits, 1).gather(1, target_indices)
        } else {
            // TODO: finfo stable eps
            let finfo = logits.dtype().finfo().unwrap();
            let eps = finfo.min_positive.sqrt();
            logits.clamp_min(eps).gather(1, target_indices).log()
        };

        match &self.weights {
            Some(weights) => {
                let weights = weights.clone().gather(0, targets);
                let tensor = tensor.reshape([batch_size]) * weights.clone();
                let tensor = Self::apply_mask_1d(tensor, mask);
                tensor.sum().neg() / weights.sum()
            }
            None => {
                let tensor = Self::apply_mask_1d(tensor.reshape([batch_size]), mask);
                tensor.mean().neg()
            }
        }
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
        let mut mask = None;
        if let Some(pad_tokens) = &self.pad_tokens {
            let mut res = targets.clone().equal_elem(pad_tokens[0] as i64).int();
            for x in pad_tokens {
                res = res + targets.clone().equal_elem(*x as i64).int();
            }
            mask = Some(res.greater_elem(0));
        }

        mask
    }

    fn apply_mask_1d(mut tensor: Tensor<1>, mask: Option<Tensor<1, Bool>>) -> Tensor<1> {
        if let Some(mask) = mask {
            tensor = tensor.mask_fill(mask, 0);
        }

        tensor
    }

    fn apply_mask_2d(mut tensor: Tensor<2>, mask: Option<Tensor<1, Bool>>) -> Tensor<2> {
        if let Some(mask) = mask {
            let [batch_size, nr_classes] = tensor.dims();
            tensor = tensor.mask_fill(mask.reshape([batch_size, 1]).repeat_dim(1, nr_classes), 0);
        }

        tensor
    }

    fn assertions(logits: Tensor<2>, targets: Tensor<1, Int>) {
        let [logits_height, _] = logits.dims();
        let [targets_height] = targets.dims();
        assert!(
            logits_height == targets_height,
            "Shape of targets ({targets_height}) should correspond to outer shape of logits ({logits_height})."
        );
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
        let pad_index = 1;

        let loss_1 = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![pad_index, 2]))
            .init(&logits.device())
            .forward(logits.clone(), targets);
        let loss_2 = cross_entropy_with_logits(logits, targets_logits);

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
    fn display() {
        let config = CrossEntropyLossConfig::new()
            .with_weights(Some(alloc::vec![3., 7., 0.9]))
            .with_smoothing(Some(0.5));
        let loss = config.init(&Default::default());

        assert_eq!(
            alloc::format!("{loss}"),
            "CrossEntropyLoss {pad_tokens: None, weights: Tensor {rank: 1, shape: [3]}, smoothing: 0.5, logits: true}"
        );
    }

    // TODO: backward tests
}
