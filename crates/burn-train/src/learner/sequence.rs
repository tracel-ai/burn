use crate::metric::{AccuracyInput, PerplexityInput, TopKAccuracyInput};
use crate::metric::{Adaptor, CerInput, LossInput, WerInput, processor::ItemLazy};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{Int, Tensor, Transaction};
use burn_ndarray::NdArray;

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
pub struct SequenceOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,
    
    /// Raw logits. Shape: `[batch_size, seq_len, vocab_size]`
    pub logits: Tensor<B, 3>,

    /// Optional predicted token indices. Shape: `[batch_size, seq_length]`.
    /// If not provided, predictions default to argmax of `logits` along the last dimension.
    pub predictions: Option<Tensor<B, 2, Int>>,

    /// The target token indices. Shape: `[batch_size, seq_length]`
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> SequenceOutput<B> {
    fn predicted_tokens(&self) -> Tensor<B, 2, Int> {
        match &self.predictions {
            Some(preds) => preds.clone(),
            None => self.logits.clone().argmax(2).squeeze_dim::<2>(2),
        }
    }
    
    fn flat_logits(&self) -> Tensor<B, 2> {
        let [batch_size, seq_len, vocab_size] = self.logits.dims();
        self.logits.clone().reshape([batch_size * seq_len, vocab_size])
    }
    
    fn flat_targets(&self) -> Tensor<B, 1, Int> {
        let [batch_size, seq_len] = self.targets.dims();
        self.targets.clone().reshape([batch_size * seq_len])
    }
}

impl<B: Backend> ItemLazy for SequenceOutput<B> {
    type ItemSync = SequenceOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let device = &Default::default();
        
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
                    logits: Tensor::from_data(logits, device),
                    loss: Tensor::from_data(loss, device),
                    targets: Tensor::from_data(targets, device),
                    predictions: Some(Tensor::from_data(predictions, device))
                }
            },
            None => {
                let [logits, loss, targets] = Transaction::default()
                    .register(self.logits)
                    .register(self.loss)
                    .register(self.targets)
                    .execute()
                    .try_into()
                    .expect("Correct amount of tensor data");
                
                SequenceOutput {
                    logits: Tensor::from_data(logits, device),
                    loss: Tensor::from_data(loss, device),
                    targets: Tensor::from_data(targets, device),
                    predictions: None
                }
            },
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
        CerInput::new(self.predicted_tokens(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<WerInput<B>> for SequenceOutput<B> {
    fn adapt(&self) -> WerInput<B> {
        WerInput::new(self.predicted_tokens(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<AccuracyInput<B>> for SequenceOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.flat_logits(), self.flat_targets())
    }
}

impl<B: Backend> Adaptor<TopKAccuracyInput<B>> for SequenceOutput<B> {
    fn adapt(&self) -> TopKAccuracyInput<B> {
        TopKAccuracyInput::new(self.flat_logits(), self.flat_targets())
    }
}

impl<B: Backend> Adaptor<PerplexityInput<B>> for SequenceOutput<B> {
    fn adapt(&self) -> PerplexityInput<B> {
        PerplexityInput::new(self.flat_logits(), self.flat_targets())
    }
}

