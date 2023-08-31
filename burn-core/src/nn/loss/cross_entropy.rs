use core::marker::PhantomData;

use burn_tensor::{activation, backend::Backend, Bool, Int, Tensor};

/// Calculate the cross entropy loss from the input logits and the targets.
#[derive(Clone, Debug, Default)]
pub struct CrossEntropyLoss<B: Backend> {
    pad_index: Option<usize>,
    weights: Option<Tensor<B, 1>>,
    backend: PhantomData<B>,
}

impl<B: Backend> CrossEntropyLoss<B> {
    /// Create the criterion.
    pub fn new(pad_index: Option<usize>) -> Self {
        Self {
            pad_index,
            backend: PhantomData,
            weights: None,
        }
    }

    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// - logits: `[batch_size, num_targets]`
    /// - targets: `[batch_size]`
    pub fn forward(&self, logits: Tensor<B, 2>, targets: Tensor<B, 1, Int>) -> Tensor<B, 1> {
        let [batch_size] = targets.dims();

        let mask = self.padding_mask(&targets);
        let tensor = activation::log_softmax(logits, 1);

        let targets = targets.reshape([batch_size, 1]);
        let mut tensor = tensor.gather(1, targets);

        if let Some(weights) = self.weights {
            assert!(targets.max() >= self.weights.dims(), 
                format!("Target assigned int {} will out-index weight vector of length {}. Please give weight vector of length corresponding to number of targets.", targets.max(), self.weights.dims())
            );
            assert!(targets.min() < 0, "One of your targets ({}) has been assigned a negative Int, which isn't supported by the currented implementation of weighted cross-entropy. Please confine your targets assignment to the postivie integers.", targets.min());
            
            let weights = self.weights.unsqueeze().swap_dims(1, 0);
            tensor = tensor * self.weights.gather(1, targets);
        }

        let tensor = self.apply_mask(tensor.reshape([batch_size]), mask);

        tensor.mean().neg()
    }

    /// Create weighted cross-entropy. 
    ///
    /// The loss of a specific sample will simply be given by: weight[y] * p(x) * 1, 
    ///
    /// # Pre-conditions
    ///   - The order of the weight vector should correspond to the label integer assignment. 
    ///   - Targets assigned negative Int's will not be allowed.
    pub fn with_weights(self, weights: Vec<f32>) -> Self {
        Self {
            weights: Some(Tensor::<B, 1, Int>::from_floats(weights.as_slice)),
            self..
        }
    }

    fn padding_mask(&self, targets: &Tensor<B, 1, Int>) -> Option<Tensor<B, 1, Bool>> {
        let mut mask = None;
        if let Some(pad_index) = self.pad_index {
            mask = Some(targets.clone().equal_elem(pad_index as i64));
        }

        mask
    }

    fn apply_mask(
        &self,
        mut tensor: Tensor<B, 1>,
        mask: Option<Tensor<B, 1, Bool>>,
    ) -> Tensor<B, 1> {
        if let Some(mask) = mask {
            tensor = tensor.mask_fill(mask, 0);
        }

        tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::{loss::cross_entropy_with_logits, Data, Distribution};

    #[test]
    fn test_cross_entropy_loss_with_weights() {
        let [batch_size, num_targets] = [4, 5];
        let logits = Tensor::<TestBackend, 2>::random(
            [batch_size, num_targets],
            Distribution::Normal(0., 1.0),
        );
        let targets = vec![
            Tensor::<B, 1, Int>::one_hot(2, 5),
            Tensor::<B, 1, Int>::one_hot(0, 5),
            Tensor::<B, 1, Int>::one_hot(4, 5),
            Tensor::<B, 1, Int>::one_hot(1, 5),
        ];
        let targets = Tensor::cat(targets, 1);
        let weights = vec![1, 2, 3, 4];
        let targets_logits = Tensor::<TestBackend, 2>::from_data(Data::from([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ]));

        let loss_1 = CrossEntropyLoss::new(None).with_weights(weights).forward(logits.clone(), targets);
        let loss_2 = targets.transpose() * Tensor::<B, 1>::from_floats(weight.as_slice()).unsqueeze().repeat(4) * target_logits;
        let loss_2 = loss_2.sum_dim(0).mean().neg();

        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }

    #[test]
    fn test_cross_entropy_loss() {
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

        let loss_1 = CrossEntropyLoss::new(None).forward(logits.clone(), targets);
        let loss_2 = cross_entropy_with_logits(logits, targets_logits);

        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }

    #[test]
    fn test_cross_entropy_loss_with_pad_token() {
        let [batch_size, num_targets, pad_index] = [4, 5, 1];
        let logits = Tensor::<TestBackend, 2>::random(
            [batch_size, num_targets],
            Distribution::Normal(0., 1.0),
        );
        let targets = Tensor::<TestBackend, 1, Int>::from_data(
            Data::<i64, 1>::from([2, 0, 4, pad_index as i64]).convert(),
        );
        let targets_logits = Tensor::<TestBackend, 2>::from_data(Data::from([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]));

        let loss_1 = CrossEntropyLoss::new(Some(pad_index)).forward(logits.clone(), targets);
        let loss_2 = cross_entropy_with_logits(logits, targets_logits);

        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }
}
