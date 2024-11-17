use crate::metric::processor::LazyItem;
use crate::metric::{AccuracyInput, Adaptor, HammingScoreInput, LossInput};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{Int, Tensor};
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

impl<B: Backend> LazyItem for ClassificationOutput<B> {
    type Output = ClassificationOutput<NdArray>;

    fn load(self) -> Self::Output {
        let device = self.output.device();
        let shape_output = self.output.shape();
        let shape_targets = self.targets.shape();
        let shape_loss = self.loss.shape();

        let n_items_output: usize = shape_output.dims.iter().sum();
        let n_items_targets: usize = shape_targets.dims.iter().sum();
        let n_items_loss: usize = shape_loss.dims.iter().sum();

        let index_output = n_items_output;
        let index_targets = index_output + n_items_targets;
        let index_loss = index_targets + n_items_loss;

        // To reduce to one sync, we create a single buffer encoding all data.
        //
        // TODO: We could have a way to read many tensors of different types in a single
        // transaction.
        let buffer =
            Tensor::<B, 1>::empty([n_items_output + n_items_targets + n_items_loss], &device);
        let buffer = buffer.slice_assign([0..index_output], self.output.reshape([n_items_output]));
        let buffer = buffer.slice_assign([index_output..index_targets], self.targets.float());
        let buffer = buffer.slice_assign([index_targets..index_loss], self.loss);
        let buffer = Tensor::<NdArray, 1>::from_data(buffer.into_data(), &Default::default());

        let output = buffer
            .clone()
            .slice([0..index_output])
            .reshape(shape_output);
        let targets = buffer.clone().slice([index_output..index_targets]).int();
        let loss = buffer
            .slice([index_targets..index_loss])
            .reshape(shape_loss);

        ClassificationOutput {
            output,
            loss,
            targets,
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
