use burn_core as burn;

use crate::{
    BatchNorm, BatchNormConfig, GroupNorm, GroupNormConfig, InstanceNorm, InstanceNormConfig,
    LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig,
};
use burn::prelude::{Config, Module};
use burn::tensor::Device;
use burn::tensor::Tensor;

/// ['Normalization'] Configuration.
///
/// The enum is non-exhaustive to prepare for future additions.
///
/// Can be used as a generic configuration for normalization layers:
/// * Construct a config with arbitrary input features (we suggest `0`).
/// * Clone and match that config to the target input layer,
///   using the [`NormalizationConfig::with_num_features()`] method.
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
    pub fn init(&self, device: &Device) -> Normalization {
        match self {
            NormalizationConfig::Batch(config) => config.init(device).into(),
            NormalizationConfig::Group(config) => config.init(device).into(),
            NormalizationConfig::Instance(config) => config.init(device).into(),
            NormalizationConfig::Layer(config) => config.init(device).into(),
            NormalizationConfig::Rms(config) => config.init(device).into(),
        }
    }

    /// Set the number of features.
    pub fn with_num_features(self, num_features: usize) -> Self {
        match self {
            NormalizationConfig::Batch(config) => BatchNormConfig {
                num_features,
                ..config
            }
            .into(),
            NormalizationConfig::Group(config) => GroupNormConfig {
                num_channels: num_features,
                ..config
            }
            .into(),
            NormalizationConfig::Instance(config) => InstanceNormConfig {
                num_channels: num_features,
                ..config
            }
            .into(),
            NormalizationConfig::Layer(config) => LayerNormConfig {
                d_model: num_features,
                ..config
            }
            .into(),
            NormalizationConfig::Rms(config) => RmsNormConfig {
                d_model: num_features,
                ..config
            }
            .into(),
        }
    }

    /// Get the number of features.
    pub fn num_features(&self) -> usize {
        match self {
            NormalizationConfig::Batch(config) => config.num_features,
            NormalizationConfig::Group(config) => config.num_channels,
            NormalizationConfig::Instance(config) => config.num_channels,
            NormalizationConfig::Layer(config) => config.d_model,
            NormalizationConfig::Rms(config) => config.d_model,
        }
    }
}

/// Normalization Layer Wrapper
///
/// Provides support for built-in ``burn::nn::norm`` norm layers:
/// * [`Normalization::Batch`] - [`BatchNorm`]
/// * [`Normalization::Group`] - [`GroupNorm`]
/// * [`Normalization::Instance`] - [`InstanceNorm`]
/// * [`Normalization::Layer`] - [`LayerNorm`]
/// * [`Normalization::Rms`] - [`RmsNorm`]
///
/// The enum is non-exhaustive, to prepare for future additions.
#[derive(Module, Debug)]
#[non_exhaustive]
pub enum Normalization {
    /// [`BatchNorm`] layer.
    Batch(BatchNorm),

    /// [`GroupNorm`] layer.
    Group(GroupNorm),

    /// ['InstanceNorm'] layer.
    Instance(InstanceNorm),

    /// [`LayerNorm`] layer.
    Layer(LayerNorm),

    /// ['RmsNorm'] layer.
    Rms(RmsNorm),
}

impl From<BatchNorm> for Normalization {
    fn from(layer: BatchNorm) -> Self {
        Self::Batch(layer)
    }
}

impl From<GroupNorm> for Normalization {
    fn from(layer: GroupNorm) -> Self {
        Self::Group(layer)
    }
}

impl From<InstanceNorm> for Normalization {
    fn from(layer: InstanceNorm) -> Self {
        Self::Instance(layer)
    }
}

impl From<LayerNorm> for Normalization {
    fn from(layer: LayerNorm) -> Self {
        Self::Layer(layer)
    }
}

impl From<RmsNorm> for Normalization {
    fn from(layer: RmsNorm) -> Self {
        Self::Rms(layer)
    }
}

impl Normalization {
    /// Applies normalization to a tensor.
    ///
    /// The normalization contract depends upon the wrapped norm layer;
    /// but all norm layers assume an input of at least rank 2;
    /// and produce an output of the same rank and shape.
    pub fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        match self {
            Normalization::Batch(norm) => norm.forward(input),
            Normalization::Group(norm) => norm.forward(input),
            Normalization::Instance(norm) => norm.forward(input),
            Normalization::Layer(norm) => norm.forward(input),
            Normalization::Rms(norm) => norm.forward(input),
        }
    }

    /// Get the number of features.
    pub fn num_features(&self) -> usize {
        match self {
            Normalization::Batch(norm) => norm.gamma.shape()[0],
            Normalization::Group(norm) => norm.num_channels,
            Normalization::Instance(norm) => norm.num_channels,
            Normalization::Layer(norm) => norm.gamma.shape()[0],
            Normalization::Rms(norm) => norm.gamma.shape()[0],
        }
    }
}

#[cfg(feature = "std")]
#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tolerance;
    type FT = f32;

    #[test]
    fn test_match_feature_size() {
        let config: NormalizationConfig = BatchNormConfig::new(0).into();
        assert_eq!(config.num_features(), 0);
        let config = config.with_num_features(12);
        assert_eq!(config.num_features(), 12);

        let config: NormalizationConfig = GroupNormConfig::new(4, 0).into();
        assert_eq!(config.num_features(), 0);
        let config = config.with_num_features(12);
        assert_eq!(config.num_features(), 12);

        let config: NormalizationConfig = InstanceNormConfig::new(0).into();
        assert_eq!(config.num_features(), 0);
        let config = config.with_num_features(12);
        assert_eq!(config.num_features(), 12);

        let config: NormalizationConfig = LayerNormConfig::new(0).into();
        assert_eq!(config.num_features(), 0);
        let config = config.with_num_features(12);
        assert_eq!(config.num_features(), 12);

        let config: NormalizationConfig = RmsNormConfig::new(0).into();
        assert_eq!(config.num_features(), 0);
        let config = config.with_num_features(12);
        assert_eq!(config.num_features(), 12);
    }

    #[test]
    fn test_batch_norm() {
        let device = Device::default().autodiff();

        let num_features = 12;
        let input: Tensor<4> = Tensor::ones([2, num_features, 3, 4], &device);

        let config: NormalizationConfig = BatchNormConfig::new(12).into();

        let layer = config.init(&device);
        assert_eq!(layer.num_features(), 12);

        let expected = match &layer {
            Normalization::Batch(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn test_group_norm() {
        let device = Device::default().autodiff();

        let num_features = 12;
        let input: Tensor<4> = Tensor::ones([2, num_features, 3, 4], &device);

        let config: NormalizationConfig = GroupNormConfig::new(3, num_features).into();

        let layer = config.init(&device);
        assert_eq!(layer.num_features(), 12);

        let expected = match &layer {
            Normalization::Group(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }

    #[test]
    fn test_instance_norm() {
        let device = Device::default().autodiff();

        let num_features = 12;
        let input: Tensor<4> = Tensor::ones([2, num_features, 3, 4], &device);

        let config: NormalizationConfig = InstanceNormConfig::new(num_features).into();

        let layer = config.init(&device);
        assert_eq!(layer.num_features(), 12);

        let expected = match &layer {
            Normalization::Instance(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }

    #[test]
    fn test_layer_norm() {
        let device = Device::default().autodiff();

        let num_features = 12;
        let input: Tensor<4> = Tensor::ones([2, 3, 4, num_features], &device);

        let config: NormalizationConfig = LayerNormConfig::new(num_features).into();

        let layer = config.init(&device);
        assert_eq!(layer.num_features(), 12);

        let expected = match &layer {
            Normalization::Layer(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }

    #[test]
    fn test_rms_norm() {
        let device = Device::default().autodiff();

        let num_features = 12;
        let input: Tensor<4> = Tensor::ones([2, 3, 4, num_features], &device);

        let config: NormalizationConfig = RmsNormConfig::new(num_features).into();

        let layer = config.init(&device);
        assert_eq!(layer.num_features(), 12);

        let expected = match &layer {
            Normalization::Rms(inner) => inner.forward(input.clone()),
            _ => panic!("Unexpected layer type"),
        };

        let output = layer.forward(input);

        output
            .to_data()
            .assert_approx_eq::<FT>(&expected.to_data(), Tolerance::default());
    }
}
