use crate as burn;

use crate::{config::Config, module::Module};
use alloc::vec;
use alloc::vec::Vec;
use burn_tensor::activation::log_softmax;
use burn_tensor::{backend::Backend, Bool, Int, Tensor};

/// Configuration to create a [Cross-entropy loss](CrossEntropyLoss).
#[derive(Config, Debug)]
pub struct CrossEntropyLossConfig {
    /// Create padded cross entropy.
    ///
    /// Prevents pad tokens from impacting loss calculation.
    pad_tokens: Option<Vec<usize>>,

    /// Create weighted cross-entropy.
    ///
    /// The loss of a specific sample will simply be given by: weight * log(p(x)) * 1,
    ///
    /// # Pre-conditions
    ///   - The order of the weight vector should correspond to the label integer assignment.
    ///   - Targets assigned negative Int's will not be allowed.
    weights: Option<Vec<f32>>,

    /// Create cross-entropy with label smoothing.
    ///
    /// Hard labels {0, 1} will be changed to y_smoothed = y(1 - a) + a / nr_classes.
    /// Alpha = 0 would be the same as default.
    smoothing: Option<f32>,

    /// Create cross-entropy with probabilities as input instead of logits.    
    ///
    #[config(default = true)]
    logits: bool,
}

impl CrossEntropyLossConfig {
    /// Initialize [Cross-entropy loss](CrossEntropyLoss).
    pub fn init<B: Backend>(&self) -> CrossEntropyLoss<B> {
        self.assertions();
        CrossEntropyLoss {
            pad_tokens: self.pad_tokens.clone(),
            weights: self
                .weights
                .as_ref()
                .map(|e| Tensor::<B, 1>::from_floats_devauto(e.as_slice())),
            smoothing: self.smoothing,
            logits: self.logits,
        }
    }

    fn assertions(&self) {
        if let Some(alpha) = self.smoothing {
            assert!(
                (0.0..=1.).contains(&alpha),
                "Alpha of Cross-entropy loss with smoothed labels should be in interval [0, 1]. Got {}",
                alpha
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
#[derive(Module, Debug)]
pub struct CrossEntropyLoss<B: Backend> {
    pad_tokens: Option<Vec<usize>>,
    /// Weights for cross-entropy.
    pub weights: Option<Tensor<B, 1>>,
    smoothing: Option<f32>,
    logits: bool,
}

impl<B: Backend> Default for CrossEntropyLoss<B> {
    fn default() -> Self {
        CrossEntropyLossConfig::new().init()
    }
}

impl<B: Backend> CrossEntropyLoss<B> {
    /// For backward compatibility.
    pub fn new(pad_index: Option<usize>) -> Self {
        CrossEntropyLossConfig::new()
            .with_pad_tokens(pad_index.map(|e| vec![e]))
            .init()
    }

    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// - logits: `[batch_size, num_targets]`
    /// - targets: `[batch_size]`
    pub fn forward(&self, logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Tensor<B, 1> {
        Self::assertions(logits.clone(), targets.clone());
        match self.smoothing {
            Some(alpha) => self.forward_smoothed(logits, targets, alpha),
            _ => self.forward_default(logits, targets),
        }
    }

    fn forward_smoothed(
        &self,
        logits: Tensor<B, 2>,
        targets: Tensor<B, 1, Int>,
        alpha: f32,
    ) -> Tensor<B, 1> {
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
                        .repeat(0, batch_size);
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

    fn forward_default(&self, logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Tensor<B, 1> {
        let [batch_size] = targets.dims();

        let mask = self.padding_mask(&targets);
        let tensor = log_softmax(logits, 1);
        let tensor = tensor.gather(1, targets.clone().reshape([batch_size, 1]));

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
        targets: Tensor<B, 1, Int>,
        alpha: f32,
    ) -> Tensor<B, 2> {
        let [batch_size, nr_classes] = shape;
        let device = &targets.device();
        let targets_matrix = Tensor::<B, 2>::zeros(shape, device).scatter(
            1,
            targets.reshape([batch_size, 1]),
            Tensor::ones([batch_size, 1], device),
        );
        targets_matrix * (1. - alpha) + alpha / nr_classes as f32
    }

    fn padding_mask(&self, targets: &Tensor<B, 1, Int>) -> Option<Tensor<B, 1, Bool>> {
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

    fn apply_mask_1d(mut tensor: Tensor<B, 1>, mask: Option<Tensor<B, 1, Bool>>) -> Tensor<B, 1> {
        if let Some(mask) = mask {
            tensor = tensor.mask_fill(mask, 0);
        }

        tensor
    }

    fn apply_mask_2d(mut tensor: Tensor<B, 2>, mask: Option<Tensor<B, 1, Bool>>) -> Tensor<B, 2> {
        if let Some(mask) = mask {
            let [batch_size, nr_classes] = tensor.dims();
            tensor = tensor.mask_fill(mask.reshape([batch_size, 1]).repeat(1, nr_classes), 0);
        }

        tensor
    }

    fn assertions(logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) {
        let [logits_height, _] = logits.dims();
        let [targets_height] = targets.dims();
        assert!(
            logits_height == targets_height,
            "Shape of targets ({}) should correspond to outer shape of logits ({}).",
            targets_height,
            logits_height
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::{loss::cross_entropy_with_logits, Data, Distribution};

    macro_rules! setup {
        () => {{
            let [batch_size, num_targets] = [4, 5];
            let logits = Tensor::<TestBackend, 2>::random_devauto(
                [batch_size, num_targets],
                Distribution::Normal(0., 1.0),
            );
            let targets =
                Tensor::<TestBackend, 1, Int>::from_data_devauto(Data::from([2, 0, 4, 1]));
            let targets_logits = Tensor::<TestBackend, 2>::from_data_devauto(Data::from([
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
            ]));
            (logits, targets, targets_logits)
        }};
    }

    macro_rules! setup_padded {
        () => {{
            let [batch_size, num_targets, pad_index] = [4, 5, 1];
            let logits = Tensor::<TestBackend, 2>::random_devauto(
                [batch_size, num_targets],
                Distribution::Normal(0., 1.0),
            );
            let targets = Tensor::<TestBackend, 1, Int>::from_data_devauto(
                Data::<i64, 1>::from([2, 0, 4, pad_index as i64]).convert(),
            );
            let targets_logits = Tensor::<TestBackend, 2>::from_data_devauto(Data::from([
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]));
            (logits, targets, targets_logits)
        }};
    }

    #[test]
    fn test_cross_entropy_loss_with_weights() {
        let (logits, targets, targets_logits) = setup!();
        let weights = vec![1.0, 2., 3., 4., 5.];
        let loss_1 = CrossEntropyLossConfig::new()
            .with_weights(Some(weights.clone()))
            .init()
            .forward(logits.clone(), targets);
        let tensor = log_softmax(logits, 1);
        let loss_2 = tensor
            * targets_logits
            * Tensor::<TestBackend, 1>::from_floats_devauto(weights.as_slice())
                .unsqueeze()
                .repeat(0, 4);
        let loss_2 = loss_2.sum().neg() / (1. + 2. + 3. + 5.);
        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }

    #[test]
    fn test_label_smoothing_with_weights_and_alpha_zero() {
        let (logits, targets, _) = setup!();
        let weights = vec![1.0, 2., 3., 4., 5.];
        let loss_1 = CrossEntropyLossConfig::new()
            .with_weights(Some(weights.clone()))
            .init()
            .forward(logits.clone(), targets.clone());
        let loss_2 = CrossEntropyLossConfig::new()
            .with_weights(Some(weights.clone()))
            .with_smoothing(Some(0.))
            .init()
            .forward(logits.clone(), targets);
        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }

    #[test]
    fn test_cross_entropy_loss() {
        let (logits, targets, targets_logits) = setup!();
        let loss_1 = CrossEntropyLossConfig::new()
            .init()
            .forward(logits.clone(), targets);
        let loss_2 = cross_entropy_with_logits(logits, targets_logits);

        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }

    #[test]
    fn test_label_smoothing_alpha_equal_zero() {
        let (logits, targets, _) = setup!();
        let loss_1 = CrossEntropyLossConfig::new()
            .init()
            .forward(logits.clone(), targets.clone());
        let loss_2 = CrossEntropyLossConfig::new()
            .with_smoothing(Some(0.))
            .init()
            .forward(logits, targets);

        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }

    #[test]
    fn test_cross_entropy_loss_with_pad_token() {
        let (logits, targets, targets_logits) = setup_padded!();
        let pad_index = 1;

        let loss_1 = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![pad_index, 2]))
            .init()
            .forward(logits.clone(), targets);
        let loss_2 = cross_entropy_with_logits(logits, targets_logits);

        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }

    #[test]
    fn test_label_smoothing_with_zero_alpha_and_pad_token() {
        let (logits, targets, _) = setup_padded!();
        let pad_index = 1;

        let loss_1 = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![pad_index, 2]))
            .init()
            .forward(logits.clone(), targets.clone());
        let loss_2 = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![pad_index, 2]))
            .with_smoothing(Some(0.))
            .init()
            .forward(logits.clone(), targets);

        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }

    #[test]
    fn test_label_smoothing_target_conversion() {
        let (logits, targets, _) = setup!();
        let smoothed_targets =
            CrossEntropyLoss::compute_smoothed_targets(logits.dims(), targets, 0.05);
        let targets_logits = Tensor::<TestBackend, 2>::from_data_devauto(Data::from([
            [0.01, 0.01, 0.96, 0.01, 0.01],
            [0.96, 0.01, 0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.01, 0.96],
            [0.01, 0.96, 0.01, 0.01, 0.01],
        ]));
        smoothed_targets
            .into_data()
            .assert_approx_eq(&targets_logits.into_data(), 3);
    }

    #[test]
    fn test_label_smoothing() {
        let (logits, targets, _) = setup!();
        let loss_1 = CrossEntropyLossConfig::new()
            .with_smoothing(Some(0.05))
            .init()
            .forward(logits.clone(), targets);
        let targets_logits = Tensor::<TestBackend, 2>::from_data_devauto(Data::from([
            [0.01, 0.01, 0.96, 0.01, 0.01],
            [0.96, 0.01, 0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.01, 0.96],
            [0.01, 0.96, 0.01, 0.01, 0.01],
        ]));

        let x = log_softmax(logits, 1);
        let loss_2 = (x * targets_logits).sum_dim(1).mean().neg();

        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }
}
