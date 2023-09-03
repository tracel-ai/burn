use crate as burn;

use crate::{config::Config, module::Module};
use alloc::vec::Vec;
use burn_tensor::activation::{log_softmax, sigmoid};
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
    /// The loss of a specific sample will simply be given by: weight[y] * log(p(x)) * 1,
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
                .map(|e| Tensor::<B, 1>::from_floats(e.as_slice())),
            smoothing: self.smoothing,
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
        let tensor = log_softmax(logits, 1);
        let [batch_size, nr_classes] = tensor.dims();
        let tensor = tensor
            * Self::compute_smoothed_targets([batch_size, nr_classes], targets.clone(), alpha);

        match &self.weights {
            Some(weights) => {
                assert_comprehensive_weights(weights, targets.clone());
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
                assert_comprehensive_weights(weights, targets.clone());
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
        let targets_matrix = Tensor::zeros(shape).scatter(
            1,
            targets.reshape([batch_size, 1]),
            Tensor::ones([batch_size, 1]),
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
        let [logits_height, logits_width] = logits.dims();
        let [targets_height] = targets.dims();
        assert!(
            logits_height == targets_height,
            "Shape of targets ({}) should correspond to outer shape of logits ({}).",
            targets_height,
            logits_height
        );
        let max_target = targets.clone().max();
        assert!(
            max_target
                .clone()
                .lower_elem(logits.dims()[1] as i32)
                .into_data()
                .value[0],
            "Encounter target ({}) greater than feature dimension ({}).",
            max_target,
            logits_width
        );
        let min_target = targets.clone().min();
        assert!(
            min_target.clone().equal_elem(0).into_data().value[0],
            "Lowest target ({}) is not equal, with isn't allowed.",
            min_target
        );
    }
}

/// Configuration to create a [Binary Cross-entropy loss](BinaryCrossEntropyLoss).
#[derive(Config, Debug)]
pub struct BinaryCrossEntropyLossConfig {
    /// Create weighted binary cross-entropy.
    ///
    /// The loss of a specific sample will simply be given by: weight[y] * log(p(x)) * 1,
    ///
    /// # Pre-conditions
    ///   - The order of the weight vector should correspond to the label integer assignment.
    ///   - Targets assigned negative Int's will not be allowed.
    pub weights: Option<Vec<f32>>,

    /// Create binary cross-entropy with label smoothing.
    ///
    /// Hard labels {0, 1} will be changed to y_smoothed = y(1 - a) + a / nr_classes.
    /// Alpha = 0 would be the same as default.
    smoothing: Option<f32>,
}

impl BinaryCrossEntropyLossConfig {
    /// Initialize [Binary Cross-entropy loss](BinaryCrossEntropyLoss).
    pub fn init<B: Backend>(&self) -> BinaryCrossEntropyLoss<B> {
        self.assertions();
        BinaryCrossEntropyLoss {
            weights: self
                .weights
                .as_ref()
                .map(|e| Tensor::<B, 1>::from_floats(e.as_slice())),
            smoothing: self.smoothing,
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
        if let Some(weights) = self.weights.as_ref() {
            assert!(
                weights.len() == 2,
                "Weights of binary cross-entropy should be of len 2. Got {}.",
                weights.len()
            );
        }
    }
}

/// Calculate the cross entropy loss from the input logits and the targets.
#[derive(Module, Debug)]
pub struct BinaryCrossEntropyLoss<B: Backend> {
    /// Weights for cross-entropy.
    pub weights: Option<Tensor<B, 1>>,
    smoothing: Option<f32>,
}

impl<B: Backend> Default for BinaryCrossEntropyLoss<B> {
    fn default() -> Self {
        BinaryCrossEntropyLossConfig::new().init()
    }
}

impl<B: Backend> BinaryCrossEntropyLoss<B> {
    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// - logits: `[batch_size]`
    /// - targets: `[batch_size]`
    pub fn forward(&self, logits: Tensor<B, 1>, targets: Tensor<B, 1, Int>) -> Tensor<B, 1> {
        Self::assertions(logits.clone(), targets.clone());
        let mut targets_float = targets.clone().float();
        if let Some(alpha) = self.smoothing {
            targets_float = targets_float * (1. - alpha) + alpha / 2.;
        }
        let loss = targets_float.clone() * logits.clone().log()
            + (targets_float.clone().neg() + 1.) * (logits.neg() + 1.).log();

        match &self.weights {
            Some(weights) => {
                assert_comprehensive_weights(weights, targets.clone());
                let loss = loss * weights.clone().slice([0..1]);
                let weights = weights.clone().gather(0, targets);
                loss.neg() / weights
            }
            None => loss.mean().neg(),
        }
    }

    fn assertions(logits: Tensor<B, 1>, targets: Tensor<B, 1, Int>) {
        let [logits_height] = logits.dims();
        let [targets_height] = targets.dims();
        assert!(
            logits_height == targets_height,
            "Shape of targets ({}) should correspond to outer shape of logits ({}).",
            targets_height,
            logits_height
        );
    }
}

fn assert_comprehensive_weights<B: Backend>(weights: &Tensor<B, 1>, targets: Tensor<B, 1, Int>) {
    let [weight_size] = weights.dims();
    assert!(
        targets
            .max()
            .lower_elem(weight_size as u32)
            .into_data()
            .value[0],
        "Cross entropy encountered target with no corresponding weight."
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::{activation::sigmoid, loss::cross_entropy_with_logits, Data, Distribution};

    macro_rules! setup {
        () => {{
            let [batch_size, num_targets] = [4, 5];
            let logits = Tensor::<TestBackend, 2>::random(
                [batch_size, num_targets],
                Distribution::Normal(0., 1.0),
            );
            let targets = Tensor::<TestBackend, 1, Int>::from_data(Data::from([2, 0, 4, 1]));
            let targets_logits = Tensor::<TestBackend, 2>::from_data(Data::from([
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
            let logits = Tensor::<TestBackend, 2>::random(
                [batch_size, num_targets],
                Distribution::Normal(0., 1.0),
            );
            let targets = Tensor::<TestBackend, 1, Int>::from_data(
                Data::<i64, 1>::from([2, 0, 4, pad_index as i64]).convert(),
            );
            let targets_logits = Tensor::<TestBackend, 2>::from_data(Data::from([
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
            * Tensor::<TestBackend, 1>::from_floats(weights.as_slice())
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
        let targets_logits = Tensor::<TestBackend, 2>::from_data(Data::from([
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
        let targets_logits = Tensor::<TestBackend, 2>::from_data(Data::from([
            [0.01, 0.01, 0.96, 0.01, 0.01],
            [0.96, 0.01, 0.01, 0.01, 0.01],
            [0.01, 0.01, 0.01, 0.01, 0.96],
            [0.01, 0.96, 0.01, 0.01, 0.01],
        ]));

        let x = log_softmax(logits, 1);
        let loss_2 = (x * targets_logits).sum_dim(1).mean().neg();

        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }

    #[test]
    fn test_binary_cross_entropy() {
        let [batch_size] = [4];
        let logits = Tensor::<TestBackend, 1>::random([batch_size], Distribution::Normal(0., 1.0));
        let targets = Tensor::<TestBackend, 1, Int>::from_data(Data::from([0, 1, 0, 1]));

        let loss_1 = BinaryCrossEntropyLossConfig::new()
            .init()
            .forward(logits.clone(), targets.clone());
        let logits = sigmoid(logits);
        let loss_2 = targets.clone().float() * logits.clone().log()
            + (-targets.float() + 1) * (-logits + 1).log();
        let loss_2 = loss_2.mean().neg();
        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }

    #[test]
    fn test_binary_cross_entropy_with_smoothing() {
        let [batch_size, nr_classes] = [4, 2];
        let logits = Tensor::<TestBackend, 2>::random(
            [batch_size, nr_classes],
            Distribution::Normal(10., 5.),
        );
        let targets = Tensor::<TestBackend, 1, Int>::from_data(Data::from([0, 1, 0, 1]));

        let loss_1 = CrossEntropyLossConfig::new()
            .with_smoothing(Some(0.1))
            .init()
            .forward(logits.clone(), targets.clone());
        let loss_2 = BinaryCrossEntropyLossConfig::new()
            .with_smoothing(Some(0.1))
            .init()
            .forward(
                logits
                    .clone()
                    .slice([0..batch_size, 0..1])
                    .reshape([batch_size]),
                targets.clone(),
            );
        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }
}
