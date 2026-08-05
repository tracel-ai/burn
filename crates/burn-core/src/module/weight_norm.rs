use alloc::vec::Vec;
use burn_tensor::{Shape, Tensor};

use crate as burn;
use crate::module::{Module, Param, ParamGroup, Reparameterization, Reparameterizer};

/// Trainable state for weight normalization.
///
/// The direction `v` is stored as the base value of the containing [`Param`].
#[derive(Debug, Module)]
pub struct WeightNorm {
    /// Trainable magnitude, with shape `[weight.shape[dim]]`.
    pub g: Param<Tensor<1>>,
    /// Dimension indexing the independently normalized weight vectors.
    pub dim: usize,
}

impl Reparameterization for WeightNorm {
    const NAME: &'static str = "weight_norm";

    fn materialize<const D: usize>(&self, base: Tensor<D>) -> Tensor<D> {
        assert!(self.dim < D, "Weight normalization dimension is invalid");
        let reduce_dims: Vec<_> = (0..D).filter(|dim| *dim != self.dim).collect();
        let norm = base.clone().powf_scalar(2.0).sum_dims(&reduce_dims).sqrt();
        let mut magnitude_shape = alloc::vec![1; D];
        magnitude_shape[self.dim] = base.dims()[self.dim];
        let magnitude = self.g.val().reshape(Shape::from(magnitude_shape));
        base.div(norm).mul(magnitude)
    }
}

/// Configuration for weight normalization.
#[derive(Debug, Clone)]
pub struct WeightNormConfig {
    /// Dimension indexing the independently normalized weight vectors. Defaults to `0`.
    pub dim: usize,
    /// Parameter group on which to apply weight normalization.
    pub param_group: ParamGroup,
}

impl WeightNormConfig {
    /// Create a weight normalization configuration using dimension `0`.
    pub fn new() -> Self {
        Self {
            dim: 0,
            param_group: ParamGroup::all(),
        }
    }

    /// Set the dimension indexing the independently normalized weight vectors.
    pub fn with_dim(mut self, dim: usize) -> Self {
        self.dim = dim;
        self
    }

    /// Restrict weight normalization to a parameter group.
    pub fn set_param_group(mut self, group: ParamGroup) -> Self {
        self.param_group = group;
        self
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
pub struct WeightNormMapper {
    config: WeightNormConfig,
}

impl WeightNormMapper {
    /// Create a mapper from the given configuration.
    pub fn new(config: WeightNormConfig) -> Self {
        Self { config }
    }

    /// Restrict weight normalization to a parameter group.
    pub fn for_group(mut self, group: ParamGroup) -> Self {
        self.config.param_group = group;
        self
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

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "autodiff")]
    use crate::module::AutodiffModule;
    use crate::{module::Module, test_device, test_utils::SimpleLinear};
    use burn_tensor::Tolerance;

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
        let weight_norm = model.weight.reparameterization_as::<WeightNorm>().unwrap();

        assert_eq!(weight_norm.g.val().dims(), [model.weight.base().dims()[0]]);
        assert_ne!(weight_norm.g.id, model.weight.id);
        assert_eq!(model.num_params(), 24 + 6 + 6);
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
        let weight_norm = model.weight.reparameterization_as::<WeightNorm>().unwrap();

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

        assert!(inference.weight.reparameterization().is_none());
        inference.weight.val().into_data().assert_approx_eq::<f32>(
            &model.weight.val().inner().into_data(),
            Tolerance::default(),
        );
    }
}
