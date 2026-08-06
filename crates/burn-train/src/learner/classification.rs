use crate::metric::{
    AccuracyInput, Adaptor, ConfusionStatsInput, HammingScoreInput, LossInput, PerplexityInput,
    TopKAccuracyInput,
    processor::ItemLazy,
};
use burn_core::tensor::{Int, Tensor};

/// Simple classification output adapted for multiple metrics.
///
/// Supported metrics:
/// - Accuracy
/// - AUROC
/// - TopKAccuracy
/// - Perplexity
/// - Precision (via ConfusionStatsInput)
/// - Recall (via ConfusionStatsInput)
/// - FBetaScore (via ConfusionStatsInput)
/// - Loss.
#[derive(new)]
pub struct ClassificationOutput {
    /// The loss.
    pub loss: Tensor<1>,

    /// The class logits or probabilities. Shape: \[batch_size, num_classes\].
    pub output: Tensor<2>,

    /// The ground truth class index for each sample. Shape: \[batch_size\].
    pub targets: Tensor<1, Int>,
}

impl ItemLazy for ClassificationOutput {
    fn sync(self) -> Self {
        // No readback: the metrics compute on the device the tensors live on
        // and read back only their final scalars. Flushing dispatches the
        // producing stream's buffered work so the metric thread doesn't wait
        // on an idle queue; a training item's float tensors come off the
        // autodiff backend entirely, so the metric thread neither retains the
        // tape nor pays its dispatch.
        self.loss.device().flush();

        ClassificationOutput {
            output: self.output.no_grad(),
            loss: self.loss.no_grad(),
            targets: self.targets,
        }
    }
}

impl Adaptor<AccuracyInput> for ClassificationOutput {
    fn adapt(&self) -> AccuracyInput {
        AccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl Adaptor<LossInput> for ClassificationOutput {
    fn adapt(&self) -> LossInput {
        LossInput::new(self.loss.clone())
    }
}

impl Adaptor<TopKAccuracyInput> for ClassificationOutput {
    fn adapt(&self) -> TopKAccuracyInput {
        TopKAccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl Adaptor<PerplexityInput> for ClassificationOutput {
    fn adapt(&self) -> PerplexityInput {
        PerplexityInput::new(self.output.clone(), self.targets.clone())
    }
}

impl Adaptor<ConfusionStatsInput> for ClassificationOutput {
    fn adapt(&self) -> ConfusionStatsInput {
        let [_, num_classes] = self.output.dims();
        if num_classes > 1 {
            ConfusionStatsInput::new(
                self.output.clone(),
                self.targets.clone().one_hot(num_classes).bool(),
            )
        } else {
            ConfusionStatsInput::new(
                self.output.clone(),
                self.targets.clone().unsqueeze_dim(1).bool(),
            )
        }
    }
}

/// Multi-label classification output adapted for multiple metrics.
///
/// Supported metrics:
/// - HammingScore
/// - Precision (via ConfusionStatsInput)
/// - Recall (via ConfusionStatsInput)
/// - FBetaScore (via ConfusionStatsInput)
/// - Loss
#[derive(new)]
pub struct MultiLabelClassificationOutput {
    /// The loss.
    pub loss: Tensor<1>,

    /// The label logits or probabilities. Shape: \[batch_size, num_classes\].
    pub output: Tensor<2>,

    /// The ground truth labels. Shape: \[batch_size, num_classes\].
    pub targets: Tensor<2, Int>,
}

impl ItemLazy for MultiLabelClassificationOutput {
    fn sync(self) -> Self {
        // Same contract as `ClassificationOutput::sync`: flush and take the
        // floats off the tape, no readback.
        self.loss.device().flush();

        MultiLabelClassificationOutput {
            output: self.output.no_grad(),
            loss: self.loss.no_grad(),
            targets: self.targets,
        }
    }
}

impl Adaptor<HammingScoreInput> for MultiLabelClassificationOutput {
    fn adapt(&self) -> HammingScoreInput {
        HammingScoreInput::new(self.output.clone(), self.targets.clone())
    }
}

impl Adaptor<LossInput> for MultiLabelClassificationOutput {
    fn adapt(&self) -> LossInput {
        LossInput::new(self.loss.clone())
    }
}

impl Adaptor<ConfusionStatsInput> for MultiLabelClassificationOutput {
    fn adapt(&self) -> ConfusionStatsInput {
        ConfusionStatsInput::new(self.output.clone(), self.targets.clone().bool())
    }
}
