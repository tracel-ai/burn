use crate as burn;

use crate::tensor::activation::log_sigmoid;
use crate::tensor::{backend::Backend, Int, Tensor};
use crate::{config::Config, module::Module};
use alloc::vec::Vec;

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
    pub fn init<B: Backend>(&self, device: &B::Device) -> BinaryCrossEntropyLoss<B> {
        self.assertions();
        BinaryCrossEntropyLoss {
            weights: self
                .weights
                .as_ref()
                .map(|e| Tensor::<B, 1>::from_floats(e.as_slice(), device)),
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

/// Calculate the binary cross entropy loss from the input logits and the targets.
///
/// Should be created using [BinaryCrossEntropyLossConfig]
#[derive(Module, Debug)]
pub struct BinaryCrossEntropyLoss<B: Backend> {
    /// Weights for cross-entropy.
    pub weights: Option<Tensor<B, 1>>,
    smoothing: Option<f32>,
    logits: bool,
}

impl<B: Backend> BinaryCrossEntropyLoss<B> {
    /// Compute the criterion on the input tensor.
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
    pub fn forward<const D: usize>(
        &self,
        logits: Tensor<B, D>,
        targets: Tensor<B, D, Int>,
    ) -> Tensor<B, 1> {
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
            (targets_float.clone() * logits.clone().log()
                + (targets_float.neg() + 1.) * (logits.neg() + 1.).log())
            .neg()
        };

        if let Some(weights) = &self.weights {
            let weights = if D > 1 {
                weights.clone().expand(shape)
            } else {
                // Flatten targets and expand resulting weights to make it compatible with
                // Tensor<B, D> for binary 1-D case
                weights
                    .clone()
                    .gather(0, targets.flatten(0, 0))
                    .expand(shape)
            };
            loss = loss * weights;
        }

        loss.mean()
    }

    fn assertions<const D: usize>(&self, logits: &Tensor<B, D>, targets: &Tensor<B, D, Int>) {
        let logits_dims = logits.dims();
        let targets_dims = targets.dims();
        assert!(
            logits_dims == targets_dims,
            "Shape of targets ({:?}) should correspond to outer shape of logits ({:?}).",
            targets_dims,
            logits_dims
        );

        if let Some(weights) = &self.weights {
            if D > 1 {
                let targets_classes = targets_dims[D - 1];
                let weights_classes = weights.dims()[0];
                assert!(
                    weights_classes == targets_classes,
                    "The number of classes ({}) does not match the weights provided ({}).",
                    weights_classes,
                    targets_classes
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{activation::sigmoid, Data};
    use crate::TestBackend;

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
        let logits =
            Tensor::<TestBackend, 1>::from_floats([0.8271, 0.9626, 0.3796, 0.2355], &device);
        let targets = Tensor::<TestBackend, 1, Int>::from_data(Data::from([0, 1, 0, 1]), &device);

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .init(&device)
            .forward(sigmoid(logits), targets)
            .into_data();

        let loss_expected = Data::from([0.7491]);
        loss_actual.assert_approx_eq(&loss_expected, 3);
    }

    #[test]
    fn test_binary_cross_entropy_with_logits() {
        let device = Default::default();
        let logits =
            Tensor::<TestBackend, 1>::from_floats([0.8271, 0.9626, 0.3796, 0.2355], &device);
        let targets = Tensor::<TestBackend, 1, Int>::from_data(Data::from([0, 1, 0, 1]), &device);

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&device)
            .forward(logits, targets)
            .into_data();

        let loss_expected = Data::from([0.7491]);
        loss_actual.assert_approx_eq(&loss_expected, 3);
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
        let logits =
            Tensor::<TestBackend, 1>::from_floats([0.8271, 0.9626, 0.3796, 0.2355], &device);
        let targets = Tensor::<TestBackend, 1, Int>::from_data(Data::from([0, 1, 0, 1]), &device);
        let weights = [3., 7.];

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .with_weights(Some(weights.to_vec()))
            .init(&device)
            .forward(sigmoid(logits), targets)
            .into_data();

        let loss_expected = Data::from([3.1531]);
        loss_actual.assert_approx_eq(&loss_expected, 3);
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
        let logits =
            Tensor::<TestBackend, 1>::from_floats([0.8271, 0.9626, 0.3796, 0.2355], &device);
        let targets = Tensor::<TestBackend, 1, Int>::from_data(Data::from([0, 1, 0, 1]), &device);

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .with_smoothing(Some(0.1))
            .init(&device)
            .forward(sigmoid(logits), targets)
            .into_data();

        let loss_expected = Data::from([0.7490]);
        loss_actual.assert_approx_eq(&loss_expected, 3);
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
        let logits = Tensor::<TestBackend, 2>::from_floats(
            [[0.5150, 0.3097, 0.7556], [0.4974, 0.9879, 0.1564]],
            &device,
        );
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(Data::from([[1, 0, 1], [1, 0, 0]]), &device);

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&device)
            .forward(logits, targets)
            .into_data();

        let loss_expected = Data::from([0.7112]);
        loss_actual.assert_approx_eq(&loss_expected, 3);
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
        let logits = Tensor::<TestBackend, 2>::from_floats(
            [[0.5150, 0.3097, 0.7556], [0.4974, 0.9879, 0.1564]],
            &device,
        );
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(Data::from([[1, 0, 1], [1, 0, 0]]), &device);
        let weights = [3., 7., 0.9];

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .with_weights(Some(weights.to_vec()))
            .init(&device)
            .forward(logits, targets)
            .into_data();

        let loss_expected = Data::from([3.1708]);
        loss_actual.assert_approx_eq(&loss_expected, 3);
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
        let logits = Tensor::<TestBackend, 2>::from_floats(
            [[0.5150, 0.3097, 0.7556], [0.4974, 0.9879, 0.1564]],
            &device,
        );
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(Data::from([[1, 0, 1], [1, 0, 0]]), &device);

        let loss_actual = BinaryCrossEntropyLossConfig::new()
            .with_smoothing(Some(0.1))
            .init(&device)
            .forward(sigmoid(logits), targets)
            .into_data();

        let loss_expected = Data::from([0.7228]);
        loss_actual.assert_approx_eq(&loss_expected, 3);
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
        let logits = Tensor::<TestBackend, 2>::from_floats(
            [[0.5150, 0.3097, 0.7556], [0.4974, 0.9879, 0.1564]],
            &device,
        );
        let targets =
            Tensor::<TestBackend, 2, Int>::from_data(Data::from([[1, 0, 1], [1, 0, 0]]), &device);
        let weights = [3., 7.];

        let _loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .with_weights(Some(weights.to_vec()))
            .init(&device)
            .forward(logits, targets);
    }
}
