use crate::metric::{
    AccuracyInput, AccuracyMetric, ConfusionStatsInput, HammingScoreInput, LossInput, LossMetric,
    MetricAdaptor, PerplexityInput, PerplexityMetric, TopKAccuracyMetric, processor::ItemLazy,
};
use crate::metric::{HammingScore, Metric, TopKAccuracyInput};
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

impl<B: Backend> MetricAdaptor<AccuracyMetric<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> AccuracyInput<B> {
        AccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> MetricAdaptor<LossMetric<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> MetricAdaptor<TopKAccuracyMetric<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> TopKAccuracyInput<B> {
        TopKAccuracyInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> MetricAdaptor<PerplexityMetric<B>> for ClassificationOutput<B> {
    fn adapt(&self) -> PerplexityInput<B> {
        PerplexityInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend, M> MetricAdaptor<M> for ClassificationOutput<B>
where
    M: Metric<Input = ConfusionStatsInput<B>>,
{
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

impl<B: Backend> MetricAdaptor<HammingScore<B>> for MultiLabelClassificationOutput<B> {
    fn adapt(&self) -> HammingScoreInput<B> {
        HammingScoreInput::new(self.output.clone(), self.targets.clone())
    }
}

impl<B: Backend> MetricAdaptor<LossMetric<B>> for MultiLabelClassificationOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend, M> MetricAdaptor<M> for MultiLabelClassificationOutput<B>
where
    M: Metric<Input = ConfusionStatsInput<B>>,
{
    fn adapt(&self) -> ConfusionStatsInput<B> {
        ConfusionStatsInput::new(self.output.clone(), self.targets.clone().bool())
    }
}
