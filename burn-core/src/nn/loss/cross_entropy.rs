use burn_tensor::{backend::Backend, loss::cross_entropy_with_logits, Tensor};

/// Calculate the cross entropy loss from the input logits and the targets.
pub struct CrossEntropyLoss<B: Backend> {
    num_targets: usize,
    pad_index: Option<usize>,
    _b: B,
}

impl<B: Backend> CrossEntropyLoss<B> {
    /// Create the criterion.
    ///
    /// # Notes
    ///
    /// The number of targets must be specified, this correspond to the number of classes in a
    /// classification task. A padding index can also be specified.
    pub fn new(num_targets: usize, pad_index: Option<usize>) -> Self {
        Self {
            num_targets,
            pad_index,
            _b: B::default(),
        }
    }

    /// Compute the criterion on the input tensor.
    ///
    /// # Shapes
    ///
    /// - logits: [batch_size, num_targets]
    /// - targets: [batch_size]
    pub fn forward(
        &self,
        logits: Tensor<B, 2>,
        targets: Tensor<B::IntegerBackend, 1>,
    ) -> Tensor<B, 1> {
        let device = logits.device();
        let [batch_size] = targets.dims();
        let indexes = targets.to_data();

        let mut targets_logits =
            Tensor::<B, 2>::zeros_device([batch_size, self.num_targets], &device);

        for b in 0..batch_size {
            let index = indexes.value[b] as usize;
            if let Some(pad_index) = self.pad_index {
                if index == pad_index {
                    continue;
                }
            }

            targets_logits = targets_logits.index_assign(
                [b..b + 1, index..index + 1],
                Tensor::ones_device([1, 1], &device),
            );
        }

        cross_entropy_with_logits(logits, targets_logits.detach())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn_tensor::{Data, Distribution};

    #[test]
    fn test_cross_entropy_loss() {
        let [batch_size, num_targets] = [4, 5];
        let logits = Tensor::<TestBackend, 2>::random(
            [batch_size, num_targets],
            Distribution::Normal(0., 1.0),
        );
        let targets =
            Tensor::<<TestBackend as Backend>::IntegerBackend, 1>::from_data(Data::from([
                2, 0, 4, 1_i64,
            ]));
        let targets_logits = Tensor::<TestBackend, 2>::from_data(Data::from([
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
        ]));

        let loss_1 = CrossEntropyLoss::new(5, None).forward(logits.clone(), targets);
        let loss_2 = cross_entropy_with_logits(logits, targets_logits);

        loss_1.into_data().assert_approx_eq(&loss_2.into_data(), 3);
    }
}
