use burn_core::tensor::Tensor;

/// Shared input for [`MseMetric`](super::MseMetric), [`MaeMetric`](super::MaeMetric),
/// and [`R2ScoreMetric`](super::R2ScoreMetric).
#[derive(new)]
pub struct RegressionInput {
    /// The model outputs, shaped `[samples, targets]`.
    pub outputs: Tensor<2>,
    /// The targets, with the same shape as the outputs.
    pub targets: Tensor<2>,
}
