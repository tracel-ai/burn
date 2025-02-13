use crate::metric::TopKAccuracyInput;
use crate::metric::{
    processor::ItemLazy, AccuracyInput, Adaptor, ConfusionStatsInput, HammingScoreInput, LossInput,
};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{Int, Tensor, Transaction};
use burn_ndarray::NdArray;

/// Simple classification output adapted for multiple metrics.
#[derive(new)]
pub struct ClassificationOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,

    /// The output.
    pub output: Tensor<B, 2>,

    /// The targets.
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> ItemLazy for ClassificationOutput<B> {
    type ItemSync = ClassificationOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        ClassificationOutput {
            output: Tensor::from_data(output, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

impl<B: Backend> Adaptor<AccuracyInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> Adaptor<TopKAccuracyInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> TopKAccuracyInput<B> {
        TopKAccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<ConfusionStatsInput<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> ConfusionStatsInput<B> {
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
#[derive(new)]
pub struct MultiLabelClassificationOutput<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,

    /// The output.
    pub output: Tensor<B, 2>,

    /// The targets.
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> ItemLazy for MultiLabelClassificationOutput<B> {
    type ItemSync = MultiLabelClassificationOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        MultiLabelClassificationOutput {
            output: Tensor::from_data(output, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}

impl<B: Backend> Adaptor<HammingScoreInput<B>> for MultiLabelClassificationOutput<B> {
    fn adapt(&self) -> HammingScoreInput<B> {
        HammingScoreInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for MultiLabelClassificationOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> Adaptor<ConfusionStatsInput<B>> for MultiLabelClassificationOutput<B> {
    fn adapt(&self) -> ConfusionStatsInput<B> {
        ConfusionStatsInput::new(self.output.clone(), self.targets.clone().bool())
    }
}
