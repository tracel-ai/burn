use burn_tensor::{backend::Backend, loss::cross_entropy_with_logits, Tensor};

pub struct CrossEntropyLoss<B: Backend> {
    num_targets: usize,
    pad_index: Option<usize>,
    _b: B,
}

impl<B: Backend> CrossEntropyLoss<B> {
    pub fn new(num_targets: usize, pad_index: Option<usize>) -> Self {
        Self {
            num_targets,
            pad_index,
            _b: B::default(),
        }
    }

    pub fn forward(
        &self,
        logits: &Tensor<B, 2>,
        targets: &Tensor<B::IntegerBackend, 1>,
    ) -> Tensor<B, 1> {
        let device = logits.device();
        let [batch_size] = targets.dims();
        let indexes = targets.to_data();

        let mut targets_logits =
            Tensor::<B, 2>::zeros_device([batch_size, self.num_targets], device);

        for b in 0..batch_size {
            let index = indexes.value[b] as usize;
            if let Some(pad_index) = self.pad_index {
                if index == pad_index {
                    continue;
                }
            }

            targets_logits = targets_logits.index_assign(
                [b..b + 1, index..index + 1],
                &Tensor::ones_device([1, 1], device),
            );
        }

        cross_entropy_with_logits(logits, &targets_logits.detach())
    }
}
