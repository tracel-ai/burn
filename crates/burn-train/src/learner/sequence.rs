use crate::metric::{Adaptor, CerInput, LossInput, WerInput, processor::ItemLazy};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{Int, Tensor, Transaction};
use burn_ndarray::NdArray;

/// Sequence prediction output adapted for CER, WER, and Loss metrics.
#[derive(new)]
pub struct SequenceOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,

    /// The predicted token indices. Shape: \[batch_size, seq_length\].
    pub output: Tensor<B, 2, Int>,

    /// The target token indices. Shape: \[batch_size, seq_length\].
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> ItemLazy for SequenceOutput<B> {
    type ItemSync = SequenceOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        SequenceOutput {
            output: Tensor::from_data(output, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for SequenceOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> Adaptor<CerInput<B>> for SequenceOutput<B> {
    fn adapt(&self) -> CerInput<B> {
        CerInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<WerInput<B>> for SequenceOutput<B> {
    fn adapt(&self) -> WerInput<B> {
        WerInput::new(self.output.clone(), self.targets.clone())
    }
}
