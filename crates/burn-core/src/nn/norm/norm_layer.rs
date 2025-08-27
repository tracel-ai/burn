use crate as burn;
use crate::nn::{
    BatchNorm, BatchNormConfig, GroupNorm, GroupNormConfig, InstanceNorm, InstanceNormConfig,
    LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig,
};
use burn_derive::{Config, Module};
use burn_tensor::Tensor;
use burn_tensor::backend::Backend;

/// ['Norm'] Configuration.
#[derive(Config, Debug)]
#[non_exhaustive]
pub enum NormLayerConfig {
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

impl From<BatchNormConfig> for NormLayerConfig {
    fn from(config: BatchNormConfig) -> Self {
        Self::Batch(config)
    }
}

impl From<GroupNormConfig> for NormLayerConfig {
    fn from(config: GroupNormConfig) -> Self {
        Self::Group(config)
    }
}

impl From<InstanceNormConfig> for NormLayerConfig {
    fn from(config: InstanceNormConfig) -> Self {
        Self::Instance(config)
    }
}

impl From<LayerNormConfig> for NormLayerConfig {
    fn from(config: LayerNormConfig) -> Self {
        Self::Layer(config)
    }
}

impl From<RmsNormConfig> for NormLayerConfig {
    fn from(config: RmsNormConfig) -> Self {
        Self::Rms(config)
    }
}

impl NormLayerConfig {
    /// Initialize a ['Norm'] layer.
    pub fn init<B: Backend>(&self, device: &B::Device) -> NormLayer<B> {
        match self {
            NormLayerConfig::Batch(config) => NormLayer::Batch(config.init(device)),
            NormLayerConfig::Group(config) => NormLayer::Group(config.init(device)),
            NormLayerConfig::Instance(config) => NormLayer::Instance(config.init(device)),
            NormLayerConfig::Layer(config) => NormLayer::Layer(config.init(device)),
            NormLayerConfig::Rms(config) => NormLayer::Rms(config.init(device)),
        }
    }
}

/// Norm Layer Wrapper
///
/// Provides support for many built-in ``burn::nn::norm`` norm layers.
#[derive(Module, Debug)]
#[non_exhaustive]
pub enum NormLayer<B: Backend> {
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

impl<B: Backend> NormLayer<B> {
    /// Applies normalization to a tensor.
    ///
    /// The normalization contract depends upon the wrapped norm layer;
    /// but all norm layers assume an input of at least rank 2;
    /// and produce an output of the same rank and shape.
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            NormLayer::Batch(batch_norm) => batch_norm.forward(input),
            NormLayer::Group(group_norm) => group_norm.forward(input),
            NormLayer::Instance(instance_norm) => instance_norm.forward(input),
            NormLayer::Layer(layer_norm) => layer_norm.forward(input),
            NormLayer::Rms(rms_norm) => rms_norm.forward(input),
        }
    }
}

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

        let config: NormLayerConfig = BatchNormConfig::new(12).into();

        let layer: NormLayer<B> = config.init(&device);

        let expected = match &layer {
            NormLayer::Batch(inner) => inner.forward(input.clone()),
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

        let config: NormLayerConfig = GroupNormConfig::new(3, num_features).into();

        let layer: NormLayer<B> = config.init(&device);

        let expected = match &layer {
            NormLayer::Group(inner) => inner.forward(input.clone()),
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

        let config: NormLayerConfig = InstanceNormConfig::new(num_features).into();

        let layer: NormLayer<B> = config.init(&device);

        let expected = match &layer {
            NormLayer::Instance(inner) => inner.forward(input.clone()),
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

        let config: NormLayerConfig = LayerNormConfig::new(num_features).into();

        let layer: NormLayer<B> = config.init(&device);

        let expected = match &layer {
            NormLayer::Layer(inner) => inner.forward(input.clone()),
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

        let config: NormLayerConfig = RmsNormConfig::new(num_features).into();

        let layer: NormLayer<B> = config.init(&device);

        let expected = match &layer {
            NormLayer::Rms(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output.to_data().assert_eq(&expected.to_data(), true);
    }
}
