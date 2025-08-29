use crate as burn;
use crate::nn::{
    BatchNorm, BatchNormConfig, GroupNorm, GroupNormConfig, InstanceNorm, InstanceNormConfig,
    LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig,
};
use burn_derive::{Config, Module};
use burn_tensor::Tensor;
use burn_tensor::backend::Backend;

/// ['Normalization'] Configuration.
///
/// The enum is non-exhaustive to prepare for future additions.
#[derive(Config, Debug)]
#[non_exhaustive]
pub enum NormalizationConfig {
    /// ['BatchNorm'] Configuration.
    Batch(BatchNormConfig),

    /// ['GroupNorm'] Configuration.
    Group(GroupNormConfig),

    /// ['InstanceNorm'] Configuration.
    Instance(InstanceNormConfig),

    /// ['LayerNorm'] Configuration.
    Layer(LayerNormConfig),

    /// ['RmsNorm'] Configuration.
    Rms(RmsNormConfig),
}

impl From<BatchNormConfig> for NormalizationConfig {
    fn from(config: BatchNormConfig) -> Self {
        Self::Batch(config)
    }
}

impl From<GroupNormConfig> for NormalizationConfig {
    fn from(config: GroupNormConfig) -> Self {
        Self::Group(config)
    }
}

impl From<InstanceNormConfig> for NormalizationConfig {
    fn from(config: InstanceNormConfig) -> Self {
        Self::Instance(config)
    }
}

impl From<LayerNormConfig> for NormalizationConfig {
    fn from(config: LayerNormConfig) -> Self {
        Self::Layer(config)
    }
}

impl From<RmsNormConfig> for NormalizationConfig {
    fn from(config: RmsNormConfig) -> Self {
        Self::Rms(config)
    }
}

impl NormalizationConfig {
    /// Initialize a ['Norm'] layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Normalization<B> {
        match self {
            NormalizationConfig::Batch(config) => Normalization::Batch(config.init(device)),
            NormalizationConfig::Group(config) => Normalization::Group(config.init(device)),
            NormalizationConfig::Instance(config) => Normalization::Instance(config.init(device)),
            NormalizationConfig::Layer(config) => Normalization::Layer(config.init(device)),
            NormalizationConfig::Rms(config) => Normalization::Rms(config.init(device)),
        }
    }
}

/// Normalization Layer Wrapper
///
/// Provides support for built-in ``burn::nn::norm`` norm layers:
/// * [`Batch`] - [`BatchNorm`]
/// * [`Group`] - [`GroupNorm`]
/// * [`Instance`] - [`InstanceNorm`]
/// * [`Layer`] - [`LayerNorm`]
/// * [`Rms`] - [`RmsNorm`]
///
/// The enum is non-exhaustive, to prepare for future additions.
#[derive(Module, Debug)]
#[non_exhaustive]
pub enum Normalization<B: Backend> {
    /// [`BatchNorm`] layer.
    Batch(BatchNorm<B>),

    /// [`GroupNorm`] layer.
    Group(GroupNorm<B>),

    /// ['InstanceNorm'] layer.
    Instance(InstanceNorm<B>),

    /// [`LayerNorm`] layer.
    Layer(LayerNorm<B>),

    /// ['RmsNorm'] layer.
    Rms(RmsNorm<B>),
}

impl<B: Backend> Normalization<B> {
    /// Applies normalization to a tensor.
    ///
    /// The normalization contract depends upon the wrapped norm layer;
    /// but all norm layers assume an input of at least rank 2;
    /// and produce an output of the same rank and shape.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Normalization::Batch(batch_norm) => batch_norm.forward(input),
            Normalization::Group(group_norm) => group_norm.forward(input),
            Normalization::Instance(instance_norm) => instance_norm.forward(input),
            Normalization::Layer(layer_norm) => layer_norm.forward(input),
            Normalization::Rms(rms_norm) => rms_norm.forward(input),
        }
    }
}

#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::TestAutodiffBackend;

    #[test]
    fn test_batch_norm() {
        type B = TestAutodiffBackend;
        let device = Default::default();

        let num_features = 12;
        let input: Tensor<B, 4> = Tensor::ones([2, num_features, 3, 4], &device);

        let config: NormalizationConfig = BatchNormConfig::new(12).into();

        let layer: Normalization<B> = config.init(&device);

        let expected = match &layer {
            Normalization::Batch(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn test_group_norm() {
        type B = TestAutodiffBackend;
        let device = Default::default();

        let num_features = 12;
        let input: Tensor<B, 4> = Tensor::ones([2, num_features, 3, 4], &device);

        let config: NormalizationConfig = GroupNormConfig::new(3, num_features).into();

        let layer: Normalization<B> = config.init(&device);

        let expected = match &layer {
            Normalization::Group(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn test_instance_norm() {
        type B = TestAutodiffBackend;
        let device = Default::default();

        let num_features = 12;
        let input: Tensor<B, 4> = Tensor::ones([2, num_features, 3, 4], &device);

        let config: NormalizationConfig = InstanceNormConfig::new(num_features).into();

        let layer: Normalization<B> = config.init(&device);

        let expected = match &layer {
            Normalization::Instance(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn test_layer_norm() {
        type B = TestAutodiffBackend;
        let device = Default::default();

        let num_features = 12;
        let input: Tensor<B, 4> = Tensor::ones([2, 3, 4, num_features], &device);

        let config: NormalizationConfig = LayerNormConfig::new(num_features).into();

        let layer: Normalization<B> = config.init(&device);

        let expected = match &layer {
            Normalization::Layer(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn test_rms_norm() {
        type B = TestAutodiffBackend;
        let device = Default::default();

        let num_features = 12;
        let input: Tensor<B, 4> = Tensor::ones([2, 3, 4, num_features], &device);

        let config: NormalizationConfig = RmsNormConfig::new(num_features).into();

        let layer: Normalization<B> = config.init(&device);

        let expected = match &layer {
            Normalization::Rms(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output.to_data().assert_eq(&expected.to_data(), true);
    }
}
