use crate::metric::{AccuracyInput, PerplexityInput, TopKAccuracyInput};
use crate::metric::{Adaptor, CerInput, LossInput, WerInput, processor::ItemLazy};
use burn_core::tensor::{Int, Tensor};

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
        // No readback: the metrics compute on the device the tensors live on
        // and read back only their final scalars, which matters here because
        // the logits carry the full vocabulary dimension. Flushing dispatches
        // the producing stream's buffered work so the metric thread doesn't
        // wait on an idle queue; all tensors in a training item come off the
        // autodiff backend entirely, so the metric thread neither retains the
        // tape nor carries its dispatch context.
        self.loss.device().flush();

        SequenceOutput {
            logits: self.logits.without_autodiff(),
            loss: self.loss.without_autodiff(),
            targets: self.targets.without_autodiff(),
            predictions: self.predictions.map(|tensor| tensor.without_autodiff()),
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
