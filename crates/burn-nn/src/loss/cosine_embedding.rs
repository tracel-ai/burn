use alloc::format;

use burn::tensor::linalg::cosine_similarity;

use burn_core as burn;

use crate::loss::reduction::Reduction;
use burn::config::Config;
use burn::module::{Content, DisplaySettings, ModuleDisplay};
use burn::module::{Ignored, Module};
use burn::tensor::{Int, Tensor, activation::relu, backend::Backend};

/// Configuration for CosineEmbeddingLoss.
#[derive(Config, Debug)]
pub struct CosineEmbeddingLossConfig {
    /// Margin for negative samples.
    #[config(default = 0.0)]
    pub margin: f32,

    /// Specifies the reduction to apply to the output.
    #[config(default = "Reduction::Mean")]
    pub reduction: Reduction,
}

impl CosineEmbeddingLossConfig {
    /// Initialize CosineEmbeddingLoss.
    pub fn init(&self) -> CosineEmbeddingLoss {
        CosineEmbeddingLoss {
            margin: self.margin,
            reduction: Ignored(self.reduction.clone()),
        }
    }
}

/// Cosine embedding loss between two tensors.
///
/// Measures cosine distance between tensors.
/// Used for learning embeddings or similarity.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct CosineEmbeddingLoss {
    /// Margin value. Default: 0.0
    pub margin: f32,

    /// Reduction method
    pub reduction: Ignored<Reduction>,
}

impl Default for CosineEmbeddingLoss {
    fn default() -> Self {
        CosineEmbeddingLossConfig::new().init()
    }
}

impl ModuleDisplay for CosineEmbeddingLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("margin", &self.margin)
            .add("reduction", format!("{:?}", &self.reduction.0).as_str())
            .optional()
    }
}

impl CosineEmbeddingLoss {
    /// Creates a new instance
    pub fn new() -> Self {
        CosineEmbeddingLossConfig::new().init()
    }

    /// Compute loss with reduction.
    ///
    /// # Shapes
    ///
    /// - input1: ``[batch_size, embedding_dim]``
    /// - input2: ``[batch_size, embedding_dim]``
    /// - target: ``[batch_size]`` with values 1 or -1
    ///
    /// # Returns
    ///
    /// Loss tensor of shape ``[1]``
    pub fn forward<B: Backend>(
        &self,
        input1: Tensor<B, 2>,
        input2: Tensor<B, 2>,
        target: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        let tensor = self.forward_no_reduction(input1, input2, target);
        match &self.reduction.0 {
            Reduction::Mean | Reduction::Auto => tensor.mean(),
            Reduction::Sum => tensor.sum(),
            other => panic!("{other:?} reduction is not supported"),
        }
    }

    /// Compute loss without applying reduction.
    ///
    /// # Arguments
    ///
    /// * `input1` - First input tensor of shape ``[batch_size, embedding_dim]``
    /// * `input2` - Second input tensor of shape ``[batch_size, embedding_dim]``
    /// * `target` - Target tensor of shape ``[batch_size]`` with values 1 or -1
    ///
    /// # Returns
    ///
    /// Tensor of per-element losses with shape ``[batch_size]``
    pub fn forward_no_reduction<B: Backend>(
        &self,
        input1: Tensor<B, 2>,
        input2: Tensor<B, 2>,
        target: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        self.assertions(&input1, &input2, &target);

        // cos_sim shape: [batch_size, 1]
        let cos_sim = cosine_similarity(input1, input2, 1, None);
        // cos_sim shape: [batch_size]
        let cos_sim: Tensor<B, 1> = cos_sim.squeeze_dim(1);

        let mut loss = cos_sim.zeros_like();

        // Similar pairs (target == 1) - Formula: L = 1 - cos_sim
        let similar_mask = target.clone().equal_elem(1);
        let similar_loss = cos_sim.clone().neg().add_scalar(1);
        loss = loss.mask_where(similar_mask, similar_loss);

        // Dissimilar pairs (target == -1) - Formula: L = max(0, cos_sim - margin)
        let dissimilar_mask = target.equal_elem(-1);
        let dissimilar_loss = relu(cos_sim.clone().sub_scalar(self.margin));
        loss = loss.mask_where(dissimilar_mask, dissimilar_loss);

        // return loss shape: [batch_size]
        loss
    }

    fn assertions<B: Backend>(
        &self,
        input1: &Tensor<B, 2>,
        input2: &Tensor<B, 2>,
        target: &Tensor<B, 1, Int>,
    ) {
        let [batch_size1, dim1] = input1.dims();
        let [batch_size2, dim2] = input2.dims();
        let [batch_size_target] = target.dims();

        assert_eq!(
            batch_size1, batch_size2,
            "Batch size of input1 ({batch_size1}) must match batch size of input2 ({batch_size2})"
        );

        assert_eq!(
            dim1, dim2,
            "Embedding dimension of input1 ({dim1}) must match embedding dimension of input2 ({dim2})"
        );

        assert_eq!(
            batch_size1, batch_size_target,
            "Batch size of inputs ({batch_size1}) must match batch size of target ({batch_size_target})"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use burn::tensor::TensorData;
    use burn::tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn cosine_embedding_loss_positive_target() {
        let device = Default::default();

        // Two identical vectors should have cosine similarity of 1
        let input1 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 0.0], [0.0, 1.0]]),
            &device,
        );

        let input2 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 0.0], [0.0, 1.0]]),
            &device,
        );

        // Target 1 means that inputs should be similar
        let target = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([1, 1]), &device);

        let loss = CosineEmbeddingLossConfig::new().init();
        let loss_no_reduction =
            loss.forward_no_reduction(input1.clone(), input2.clone(), target.clone());
        let loss_mean = loss.forward(input1.clone(), input2.clone(), target.clone());

        let loss_sum = loss.forward(input1, input2, target);

        // For identical vectors, 1 - cos_sim = 1 - 1 = 0
        let expected_no_reduction = TensorData::from([0.0, 0.0]);
        loss_no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected_no_reduction, Tolerance::default());

        let expected_mean = TensorData::from([0.0]);
        loss_mean
            .into_data()
            .assert_approx_eq::<FT>(&expected_mean, Tolerance::default());

        let expected_sum = TensorData::from([0.0]);
        loss_sum
            .into_data()
            .assert_approx_eq::<FT>(&expected_sum, Tolerance::default());
    }

    #[test]
    fn cosine_embedding_loss_negative_target() {
        let device = Default::default();

        // Two identical vectors should have cosine similarity of 1
        let input1 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 0.0], [0.0, 1.0]]),
            &device,
        );

        let input2 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 0.0], [0.0, 1.0]]),
            &device,
        );

        // Target -1 means that inputs should be dissimilar
        let target = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([-1, -1]), &device);

        // With margin 0.0, max(0, cos_sim - margin) = max(0, 1 - 0) = 1
        let loss = CosineEmbeddingLossConfig::new().init();
        let loss_no_reduction =
            loss.forward_no_reduction(input1.clone(), input2.clone(), target.clone());
        let loss_mean = loss.forward(input1.clone(), input2.clone(), target.clone());

        // Create a loss with Sum reduction for testing
        let loss_sum_config = CosineEmbeddingLossConfig::new().with_reduction(Reduction::Sum);
        let loss_sum =
            loss_sum_config
                .init()
                .forward(input1.clone(), input2.clone(), target.clone());

        let expected_no_reduction = TensorData::from([1.0, 1.0]);
        loss_no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected_no_reduction, Tolerance::default());

        let expected_mean = TensorData::from([1.0]);
        loss_mean
            .into_data()
            .assert_approx_eq::<FT>(&expected_mean, Tolerance::default());

        let expected_sum = TensorData::from([2.0]);
        loss_sum
            .into_data()
            .assert_approx_eq::<FT>(&expected_sum, Tolerance::default());

        // With margin 0.5, max(0, cos_sim - margin) = max(0, 1 - 0.5) = 0.5
        let loss_with_margin = CosineEmbeddingLossConfig::new().with_margin(0.5).init();
        let loss_with_margin = loss_with_margin.forward(input1, input2, target);

        let expected = TensorData::from([0.5]);
        loss_with_margin
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn cosine_embedding_loss_mixed_targets() {
        let device = Default::default();

        let input1 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 0.0], [0.0, 1.0]]),
            &device,
        );

        let input2 = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 0.0], [0.0, 1.0]]),
            &device,
        );

        // Mixed targets
        let target = Tensor::<TestBackend, 1, Int>::from_data(TensorData::from([1, -1]), &device);

        let loss = CosineEmbeddingLossConfig::new().init();
        let loss_no_reduction =
            loss.forward_no_reduction(input1.clone(), input2.clone(), target.clone());
        let loss_mean = loss.forward(input1, input2, target);

        let expected_no_reduction = TensorData::from([0.0, 1.0]);
        loss_no_reduction
            .into_data()
            .assert_approx_eq::<FT>(&expected_no_reduction, Tolerance::default());

        let expected_mean = TensorData::from([0.5]);
        loss_mean
            .into_data()
            .assert_approx_eq::<FT>(&expected_mean, Tolerance::default());
    }

    #[test]
    fn display() {
        let config = CosineEmbeddingLossConfig::new().with_margin(0.5);
        let loss = config.init();

        assert_eq!(
            alloc::format!("{loss}"),
            "CosineEmbeddingLoss {margin: 0.5, reduction: Mean}"
        );
    }
}
