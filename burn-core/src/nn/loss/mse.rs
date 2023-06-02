use core::marker::PhantomData;

use burn_tensor::{backend::Backend, Tensor};

/// Calculate the mean squared error loss from the input logits and the targets.
#[derive(Clone, Debug)]
pub struct MSELoss<B: Backend> {
    backend: PhantomData<B>,
}

impl<B: Backend> Default for MSELoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> MSELoss<B> {
    /// Create the criterion.
    pub fn new() -> Self {
        Self {
            backend: PhantomData::default(),
        }
    }

    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// - logits: [batch_size, num_targets]
    /// - targets: [batch_size, num_targets]
    pub fn forward(&self, logits: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_none(logits, targets).mean()
    }

    pub fn forward_none(&self, logits: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 2> {
        logits.sub(targets).powf(2.0)
    }

    pub fn forward_sum(&self, logits: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
        self.forward_none(logits, targets).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::{loss::mse, Data, Distribution};

    #[test]
    fn test_mse_loss() {
        let [batch_size, num_targets] = [4, 5];

        let logits = Tensor::<TestBackend, 2>::random(
            [batch_size, num_targets],
            Distribution::Normal(0., 1.0),
        );

        let targets = Tensor::<TestBackend, 2>::random(
            [batch_size, num_targets],
            Distribution::Normal(0., 1.0),
        );

        let loss_1 = MSELoss::new().forward(logits.clone(), targets.clone());
        let loss_2 = mse(logits, targets);

        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }

    #[test]
    fn test_mse_loss_sum_reduction() {
        let logits = Tensor::<TestBackend, 2>::from_data(Data::from([[1.8, 51.8], [6.66, 8.8]]));

        let targets = Tensor::<TestBackend, 2>::from_data(Data::from([[0.18, 5.18], [66.6, 0.88]]));

        let loss = MSELoss::new().forward_sum(logits, targets);

        loss.into_data()
            .assert_approx_eq(&Data::from([5831.579_f32]), 3);
    }
}
