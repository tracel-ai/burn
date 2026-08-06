use burn_tensor::{Shape, Tensor};
use std::vec::Vec;

use burn_core as burn;
use burn_core::module::{Module, Param, ParamGroup, Reparameterization, Reparameterizer};
use burn_tensor::Device;

/// Trainable state for weight normalization.
///
/// The direction `v` is stored as the base value of the containing [`Param`].
#[derive(Debug, Module)]
struct WeightNorm {
    /// Trainable magnitude, with shape `[weight.shape[dim]]`.
    g: Param<Tensor<1>>,
    /// Dimension indexing the independently normalized weight vectors.
    dim: usize,
}

impl Reparameterization for WeightNorm {
    const NAME: &'static str = "weight_norm";

    fn materialize<const D: usize>(&self, base: Tensor<D>) -> Tensor<D> {
        assert!(self.dim < D, "Weight normalization dimension is invalid");
        let reduce_dims: Vec<_> = (0..D).filter(|dim| *dim != self.dim).collect();
        let norm = base.clone().powf_scalar(2.0).sum_dims(&reduce_dims).sqrt();
        let mut magnitude_shape = vec![1; D];
        magnitude_shape[self.dim] = base.dims()[self.dim];
        let magnitude = self.g.val().reshape(Shape::from(magnitude_shape));
        base.mul(magnitude.div(norm))
    }
}

/// Configuration for weight normalization.
#[derive(Debug, Clone)]
struct WeightNormConfig {
    /// Dimension indexing the independently normalized weight vectors. Defaults to `1` for
    /// `[d_input, d_output]` linear weights.
    dim: usize,
    /// Parameter group on which to apply weight normalization.
    param_group: ParamGroup,
}

impl WeightNormConfig {
    /// Create a weight normalization configuration using dimension `1`.
    fn new() -> Self {
        Self {
            dim: 1,
            param_group: ParamGroup::all(),
        }
    }
}

impl Default for WeightNormConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Reparameterizer that replaces matching rank-2 or higher floating-point weights by
/// weight-normalized ones. Rank-1 parameters such as biases are left unchanged.
#[derive(Debug, Clone)]
struct WeightNormMapper {
    config: WeightNormConfig,
}

impl WeightNormMapper {
    /// Create a mapper from the given configuration.
    fn new(config: WeightNormConfig) -> Self {
        Self { config }
    }
}

impl Reparameterizer for WeightNormMapper {
    type Reparam = WeightNorm;

    fn reparameterize<const D: usize>(
        &mut self,
        path: &str,
        param: Param<Tensor<D>>,
    ) -> (Param<Tensor<D>>, Option<Self::Reparam>) {
        if D < 2 {
            return (param, None);
        }
        if !self.config.param_group.matches(&param.id, Some(path)) {
            return (param, None);
        }

        let dim = self.config.dim;
        assert!(
            dim < D,
            "Weight normalization dimension {dim} is invalid for a rank-{D} parameter"
        );

        let direction = param.base();
        let reduce_dims: Vec<_> = (0..D).filter(|axis| *axis != dim).collect();
        let magnitude = direction
            .clone()
            .powf_scalar(2.0)
            .sum_dims(&reduce_dims)
            .sqrt()
            .reshape([direction.dims()[dim]])
            .detach();
        let weight_norm = WeightNorm {
            g: Param::from_tensor(magnitude),
            dim,
        };

        (param, Some(weight_norm))
    }
}

mod tests {
    use super::*;
    #[cfg(feature = "autodiff")]
    use burn_core::module::AutodiffModule;
    use burn_tensor::Tolerance;

    #[derive(Debug, Module)]
    struct SimpleLinear {
        weight: Param<Tensor<2>>,
        bias: Param<Tensor<1>>,
    }

    impl SimpleLinear {
        fn new(in_features: usize, out_features: usize, device: &Device) -> Self {
            Self {
                weight: Param::from_tensor(Tensor::random(
                    [in_features, out_features],
                    burn_tensor::Distribution::Default,
                    device,
                )),
                bias: Param::from_tensor(Tensor::zeros([out_features], device)),
            }
        }
    }

    fn test_device() -> Device {
        Device::flex()
    }

    fn simple_model() -> SimpleLinear {
        let device = test_device();
        SimpleLinear::new(4, 6, &device)
            .apply_reparameterization(WeightNormMapper::new(WeightNormConfig::new()))
    }

    #[test]
    fn preserves_effective_weight_when_attached() {
        let device = test_device();
        let original = SimpleLinear::new(4, 6, &device);
        let expected = original.weight.val();
        let model =
            original.apply_reparameterization(WeightNormMapper::new(WeightNormConfig::new()));

        model
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    fn exposes_magnitude_as_parameter() {
        let model = simple_model();
        let weight_norm = model.weight.reparameterization::<WeightNorm>().unwrap();

        assert_eq!(weight_norm.g.val().dims(), [model.weight.base().dims()[1]]);
        assert_ne!(weight_norm.g.id, model.weight.id);
        assert_eq!(
            model.num_params(),
            model.weight.base().shape().num_elements()
                + model.bias.base().shape().num_elements()
                + weight_norm.g.val().shape().num_elements()
        );
    }

    #[test]
    fn record_roundtrip_preserves_direction_and_magnitude() {
        let source = simple_model();
        let target = simple_model();
        let loaded = target.load_record(source.clone().into_record());

        loaded
            .weight
            .base()
            .into_data()
            .assert_eq(&source.weight.base().into_data(), true);
        loaded
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(&source.weight.val().into_data(), Tolerance::default());
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn gradients_flow_to_direction_and_magnitude() {
        let device = test_device().autodiff();
        let model = SimpleLinear::new(4, 6, &device)
            .apply_reparameterization(WeightNormMapper::new(WeightNormConfig::new()));
        let grads = model.weight.val().sum().backward();
        let weight_norm = model.weight.reparameterization::<WeightNorm>().unwrap();

        assert!(model.weight.base().grad(&grads).is_some());
        assert!(weight_norm.g.val().grad(&grads).is_some());
    }

    #[cfg(feature = "autodiff")]
    #[test]
    fn valid_folds_reparameterization() {
        let device = test_device().autodiff();
        let model = SimpleLinear::new(4, 6, &device)
            .apply_reparameterization(WeightNormMapper::new(WeightNormConfig::new()));
        let inference = model.valid();

        assert!(
            inference
                .weight
                .reparameterization::<WeightNorm>()
                .is_none()
        );
        inference.weight.val().into_data().assert_approx_eq::<f32>(
            &model.weight.val().inner().into_data(),
            Tolerance::default(),
        );
    }
}
