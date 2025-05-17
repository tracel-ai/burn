use burn_tensor::linalg::l2_norm;

use crate as burn;

use crate::config::Config;
use crate::module::Module;
use crate::module::{Content, DisplaySettings, ModuleDisplay};
use crate::nn::loss::reduction::Reduction;
use crate::tensor::{Int, Tensor, activation::relu, backend::Backend};

/// Configuration to create a [Cosine Embedding loss](CosineEmbeddingLoss) using the [init function](CosineEmbeddingLossConfig::init).
#[derive(Config, Debug)]
pub struct CosineEmbeddingLossConfig {
    /// Margin in the loss calculation for negative samples.
    ///
    /// The cosine embedding loss is computed as:
    /// - target = 1: 1 - cos_sim
    /// - target = -1: max(0, cos_sim - margin)
    ///
    /// Default: 0.0
    #[config(default = 0.0)]
    pub margin: f32,
}

impl CosineEmbeddingLossConfig {
    /// Initialize [Cosine Embedding loss](CosineEmbeddingLoss).
    pub fn init(&self) -> CosineEmbeddingLoss {
        CosineEmbeddingLoss {
            margin: self.margin,
        }
    }
}

/// Calculate the cosine embedding loss between two tensors.
///
/// The cosine embedding loss measures the cosine distance between two tensors.
/// It can be used for tasks like learning nonlinear embeddings or learning similarity.
///
/// Should be created using [CosineEmbeddingLossConfig]
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct CosineEmbeddingLoss {
    /// Margin value in the loss calculation. Default: 0.0
    pub margin: f32,
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
        content.add("margin", &self.margin).optional()
    }
}

impl CosineEmbeddingLoss {
    /// Create the criterion with default margin of 0.0.
    ///
    /// For backward compatibility.
    pub fn new() -> Self {
        CosineEmbeddingLossConfig::new().init()
    }

    /// Create the criterion with a specified margin.
    ///
    /// For backward compatibility.
    pub fn with_margin(margin: f32) -> Self {
        CosineEmbeddingLossConfig::new().with_margin(margin).init()
    }

    /// Compute the criterion on the input tensors.
    ///
    /// # Shapes
    ///
    /// - input1: [batch_size, embedding_dim]
    /// - input2: [batch_size, embedding_dim]
    /// - target: [batch_size] with values 1 or -1
    ///
    /// # Returns
    ///
    /// Loss tensor of shape [1]
    pub fn forward<B: Backend>(
        &self,
        input1: Tensor<B, 2>,
        input2: Tensor<B, 2>,
        target: Tensor<B, 1, Int>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let tensor = self.forward_no_reduction(input1, input2, target);
        match reduction {
            Reduction::Mean | Reduction::Auto => tensor.mean(),
            Reduction::Sum => tensor.sum(),
        }
    }

    /// Compute the criterion on the input tensors without reducing.
    ///
    /// Returns a tensor with per-element losses.
    pub fn forward_no_reduction<B: Backend>(
        &self,
        input1: Tensor<B, 2>,
        input2: Tensor<B, 2>,
        target: Tensor<B, 1, Int>,
    ) -> Tensor<B, 1> {
        self.assertions(&input1, &input2, &target);

        // Compute cosine similarity
        let input1_norm = l2_norm(input1.clone(), 1);
        let input2_norm = l2_norm(input2.clone(), 1);
        let norm_product = input1_norm * input2_norm;
        let dot_product = (input1 * input2).sum_dim(1);

        let cos_sim = dot_product / norm_product.unsqueeze();

        // Target is 1: return 1 - cos_sim
        // Target is -1: return max(0, cos_sim - margin)
        let pos_part = target
            .clone()
            .equal_elem(1)
            .float()
            .mul((Tensor::ones_like(&cos_sim) - cos_sim.clone()).unsqueeze());

        let neg_part = target
            .equal_elem(-1)
            .float()
            .mul(relu(cos_sim - self.margin).unsqueeze());

        pos_part + neg_part
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
            "Batch size of input1 ({}) must match batch size of input2 ({})",
            batch_size1, batch_size2
        );

        assert_eq!(
            dim1, dim2,
            "Embedding dimension of input1 ({}) must match embedding dimension of input2 ({})",
            dim1, dim2
        );

        assert_eq!(
            batch_size1, batch_size_target,
            "Batch size of inputs ({}) must match batch size of target ({})",
            batch_size1, batch_size_target
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestBackend;
    use crate::tensor::TensorData;
    use burn_tensor::{Tolerance, ops::FloatElem};
    type FT = FloatElem<TestBackend>;

    #[test]
    fn test_cosine_embedding_loss_positive_target() {
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
        let loss_mean = loss.forward(
            input1.clone(),
            input2.clone(),
            target.clone(),
            Reduction::Mean,
        );
        let loss_sum = loss.forward(input1, input2, target, Reduction::Sum);

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
    fn test_cosine_embedding_loss_negative_target() {
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
        let loss_mean = loss.forward(
            input1.clone(),
            input2.clone(),
            target.clone(),
            Reduction::Mean,
        );
        let loss_sum = loss.forward(
            input1.clone(),
            input2.clone(),
            target.clone(),
            Reduction::Sum,
        );

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
        let loss_with_margin = loss_with_margin.forward(input1, input2, target, Reduction::Mean);

        let expected = TensorData::from([0.5]);
        loss_with_margin
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn test_cosine_embedding_loss_mixed_targets() {
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
        let loss_mean = loss.forward(input1, input2, target, Reduction::Mean);

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
            alloc::format!("{}", loss),
            "CosineEmbeddingLoss {margin: 0.5}"
        );
    }
}
