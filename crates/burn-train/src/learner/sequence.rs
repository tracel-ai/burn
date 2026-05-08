use crate::metric::{AccuracyInput, PerplexityInput, TopKAccuracyInput};
use crate::metric::{Adaptor, CerInput, LossInput, WerInput, processor::ItemLazy};
use burn_core::tensor::{Device, Int, Tensor, Transaction};
use burn_flex::FlexDevice;

/// Sequence prediction output adapted for multiple metrics.
///
/// Supported metrics:
/// - Accuracy
/// - TopKAccuracy
/// - Perplexity
/// - Loss
/// - CER
/// - WER
#[derive(new)]
pub struct SequenceOutput {
    /// The loss.
    pub loss: Tensor<1>,

    /// Raw logits. Shape: `[batch_size, seq_len, vocab_size]`
    pub logits: Tensor<3>,

    /// Optional predicted token indices. Shape: `[batch_size, seq_length]`.
    /// If not provided, predictions default to argmax of `logits` along the last dimension.
    pub predictions: Option<Tensor<2, Int>>,

    /// The target token indices. Shape: `[batch_size, seq_length]`
    pub targets: Tensor<2, Int>,
}

impl SequenceOutput {
    fn predicted_tokens(&self) -> Tensor<2, Int> {
        match &self.predictions {
            Some(preds) => preds.clone(),
            None => self.logits.clone().argmax(2).squeeze_dim::<2>(2),
        }
    }

    fn flat_logits(&self) -> Tensor<2> {
        let [batch_size, seq_len, vocab_size] = self.logits.dims();
        self.logits
            .clone()
            .reshape([batch_size * seq_len, vocab_size])
    }

    fn flat_targets(&self) -> Tensor<1, Int> {
        let [batch_size, seq_len] = self.targets.dims();
        self.targets.clone().reshape([batch_size * seq_len])
    }
}

impl ItemLazy for SequenceOutput {
    fn sync(self) -> Self {
        let device: Device = FlexDevice.into();

        match self.predictions {
            Some(preds) => {
                let [logits, loss, targets, predictions] = Transaction::default()
                    .register(self.logits)
                    .register(self.loss)
                    .register(self.targets)
                    .register(preds)
                    .execute()
                    .try_into()
                    .expect("Correct amount of tensor data");

                SequenceOutput {
                    logits: Tensor::from_data(logits, &device),
                    loss: Tensor::from_data(loss, &device),
                    targets: Tensor::from_data(targets, &device),
                    predictions: Some(Tensor::from_data(predictions, &device)),
                }
            }
            None => {
                let [logits, loss, targets] = Transaction::default()
                    .register(self.logits)
                    .register(self.loss)
                    .register(self.targets)
                    .execute()
                    .try_into()
                    .expect("Correct amount of tensor data");

                SequenceOutput {
                    logits: Tensor::from_data(logits, &device),
                    loss: Tensor::from_data(loss, &device),
                    targets: Tensor::from_data(targets, &device),
                    predictions: None,
                }
            }
        }
    }
}

impl Adaptor<LossInput> for SequenceOutput {
    fn adapt(&self) -> LossInput {
        LossInput::new(self.loss.clone())
    }
}

impl Adaptor<CerInput> for SequenceOutput {
    fn adapt(&self) -> CerInput {
        CerInput::new(self.predicted_tokens(), self.targets.clone())
    }
}

impl Adaptor<WerInput> for SequenceOutput {
    fn adapt(&self) -> WerInput {
        WerInput::new(self.predicted_tokens(), self.targets.clone())
    }
}

impl Adaptor<AccuracyInput> for SequenceOutput {
    fn adapt(&self) -> AccuracyInput {
        AccuracyInput::new(self.flat_logits(), self.flat_targets())
    }
}

impl Adaptor<TopKAccuracyInput> for SequenceOutput {
    fn adapt(&self) -> TopKAccuracyInput {
        TopKAccuracyInput::new(self.flat_logits(), self.flat_targets())
    }
}

impl Adaptor<PerplexityInput> for SequenceOutput {
    fn adapt(&self) -> PerplexityInput {
        PerplexityInput::new(self.flat_logits(), self.flat_targets())
    }
}
