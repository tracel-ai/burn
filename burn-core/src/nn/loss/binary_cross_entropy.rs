use crate as burn;

use crate::{config::Config, module::Module};
use burn_tensor::activation::sigmoid;
use burn_tensor::{backend::Backend, Int, Tensor};

/// Configuration to create a [Binary Cross-entropy loss](BinaryCrossEntropyLoss).
#[derive(Config, Debug)]
pub struct BinaryCrossEntropyLossConfig {
    /// Create weighted binary cross-entropy.
    ///
    /// The loss of a specific sample will simply be given by: weight * log(p(x)) * 1,
    ///
    /// # Pre-conditions
    ///   - The order of the weight vector should correspond to the label integer assignment.
    ///   - Targets assigned negative Int's will not be allowed.
    pub weights: Option<[f32; 2]>,

    /// Create binary cross-entropy with label smoothing.
    ///
    /// Hard labels {0, 1} will be changed to y_smoothed = y(1 - a) + a / nr_classes.
    /// Alpha = 0 would be the same as default.
    smoothing: Option<f32>,

    /// Create binary cross-entropy with probabilities as input instead of logits.    
    ///
    #[config(default = true)]
    logits: bool,
}

impl BinaryCrossEntropyLossConfig {
    /// Initialize [Binary Cross-entropy loss](BinaryCrossEntropyLoss).
    pub fn init<B: Backend>(&self) -> BinaryCrossEntropyLoss<B> {
        self.assertions();
        BinaryCrossEntropyLoss {
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
pub struct BinaryCrossEntropyLoss<B: Backend> {
    /// Weights for cross-entropy.
    pub weights: Option<Tensor<B, 1>>,
    smoothing: Option<f32>,
    logits: bool,
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
        let logits = if self.logits { sigmoid(logits) } else { logits };
        let loss = targets_float.clone() * logits.clone().log()
            + (targets_float.clone().neg() + 1.) * (logits.neg() + 1.).log();

        match &self.weights {
            Some(weights) => {
                let weights = weights.clone().gather(0, targets);
                let loss = loss * weights.clone();
                loss.neg().sum() / weights.sum()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::{activation::sigmoid, Data, Distribution};

    #[test]
    fn test_binary_cross_entropy() {
        let [batch_size] = [4];
        let logits =
            Tensor::<TestBackend, 1>::random_devauto([batch_size], Distribution::Normal(0., 1.0));
        let targets = Tensor::<TestBackend, 1, Int>::from_data_devauto(Data::from([0, 1, 0, 1]));

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
    fn test_binary_cross_entropy_with_weights() {
        let [batch_size] = [4];
        let logits =
            Tensor::<TestBackend, 1>::random_devauto([batch_size], Distribution::Normal(0., 1.0));
        let targets = Tensor::<TestBackend, 1, Int>::from_data_devauto(Data::from([0, 1, 0, 1]));
        let weights = [3., 7.];

        let loss_1 = BinaryCrossEntropyLossConfig::new()
            .with_weights(Some(weights))
            .init()
            .forward(logits.clone(), targets.clone());
        let logits = sigmoid(logits);
        let loss_2 = targets.clone().float() * logits.clone().log()
            + (-targets.float() + 1) * (-logits + 1).log();

        let loss_2 = loss_2 * Tensor::from_floats_devauto([3., 7., 3., 7.]);
        let loss_2 = loss_2.neg().sum() / (3. + 3. + 7. + 7.);
        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }

    #[test]
    fn test_binary_cross_entropy_with_smoothing() {
        let [batch_size] = [4];
        let logits =
            Tensor::<TestBackend, 1>::random_devauto([batch_size], Distribution::Normal(0., 1.0));
        let targets = Tensor::<TestBackend, 1, Int>::from_data_devauto(Data::from([0, 1, 0, 1]));

        let loss_1 = BinaryCrossEntropyLossConfig::new()
            .with_smoothing(Some(0.1))
            .init()
            .forward(logits.clone(), targets.clone());

        let logits = sigmoid(logits);
        let targets = targets.float() * (1. - 0.1) + 0.1 / 2.;
        let loss_2 = targets.clone() * logits.clone().log() + (-targets + 1) * (-logits + 1).log();
        let loss_2 = loss_2.mean().neg();

        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }
}
