use crate::metric::{
    AccuracyInput, Adaptor, AurocInput, ConfusionStatsInput, HammingScoreInput, LossInput,
    PerplexityInput, TopKAccuracyInput, processor::ItemLazy,
};
use burn_core::tensor::{Device, Int, Tensor, Transaction};
use burn_flex::FlexDevice;

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
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device: Device = FlexDevice.into();

        ClassificationOutput {
            output: Tensor::from_data(output, &device),
            loss: Tensor::from_data(loss, &device),
            targets: Tensor::from_data(targets, &device),
        }
    }
}

impl Adaptor<AccuracyInput> for ClassificationOutput {
    fn adapt(&self) -> AccuracyInput {
        AccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl Adaptor<AurocInput> for ClassificationOutput {
    fn adapt(&self) -> AurocInput {
        AurocInput::new(self.output.clone(), self.targets.clone())
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
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device: Device = FlexDevice.into();

        MultiLabelClassificationOutput {
            output: Tensor::from_data(output, &device),
            loss: Tensor::from_data(loss, &device),
            targets: Tensor::from_data(targets, &device),
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
