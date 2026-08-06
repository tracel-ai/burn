use alloc::{string::String, string::ToString, vec::Vec};
use burn_tensor::Tensor;

use crate::module::{AutodiffModule, ModuleMapper};

use super::Param;

/// A rank-specific parameter reparameterization.
///
/// Implementations are regular [`Module`](crate::module::Module)s, so their parameters automatically participate in
/// optimizer, record, device and autodiff traversal. The implementation only needs to describe
/// how its state materializes an effective value from the stored base parameter.
pub trait Reparameterization: AutodiffModule + Sync + 'static {
    /// Stable path component used for the reparameterization's nested parameters.
    const NAME: &'static str;
    /// Materialize the effective parameter value from its stored base.
    fn materialize<const D: usize>(&self, base: Tensor<D>) -> Tensor<D>;
}

/// Defines how floating-point parameters are prepared for reparameterization.
///
/// [`Module::apply_reparameterization`](crate::module::Module::apply_reparameterization) passes
/// every floating-point parameter encountered during module traversal to [`reparameterize`](Self::reparameterize).
/// Implementations may use the parameter path to decide whether to attach a
/// [`Reparameterization`] and may transform the parameter into the structural base that should be
/// stored.
pub trait Reparameterizer {
    /// Reparameterization produced for a parameter.
    type Reparam: Reparameterization;

    /// Prepare a parameter and optionally create a reparameterization for it.
    ///
    /// The returned parameter is always used as the structural base. Returning `None` leaves that
    /// base without a reparameterization.
    fn reparameterize<const D: usize>(
        &mut self,
        path: &str,
        param: Param<Tensor<D>>,
    ) -> (Param<Tensor<D>>, Option<Self::Reparam>);
}

pub(crate) struct ApplyReparameterization<R> {
    reparameterizer: R,
    path: Vec<String>,
}

impl<R> ApplyReparameterization<R> {
    pub(crate) fn new(reparameterizer: R) -> Self {
        Self {
            reparameterizer,
            path: Vec::new(),
        }
    }
}

impl<R: Reparameterizer> ModuleMapper for ApplyReparameterization<R> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let path = self.path.join(".");
        let (base, reparameterization) = self.reparameterizer.reparameterize(&path, param);
        match reparameterization {
            Some(reparameterization) => base.with_reparameterization(reparameterization),
            None => base,
        }
    }
}

#[cfg(all(test, feature = "autodiff"))]
mod tests {
    use super::*;
    use crate as burn;
    use crate::module::Reparameterizer;
    use crate::{module::Module, test_device, test_utils::SimpleLinear};
    use burn_tensor::{Shape, Tolerance};

    #[derive(Debug, Module)]
    struct CustomScale {
        scale: Param<Tensor<1>>,
    }

    impl Reparameterization for CustomScale {
        const NAME: &'static str = "custom_scale";

        fn materialize<const D: usize>(&self, base: Tensor<D>) -> Tensor<D> {
            base * self.scale.val().reshape(Shape::from(alloc::vec![1; D]))
        }
    }

    struct CustomScaleMapper;

    impl Reparameterizer for CustomScaleMapper {
        type Reparam = CustomScale;

        fn reparameterize<const D: usize>(
            &mut self,
            _path: &str,
            param: Param<Tensor<D>>,
        ) -> (Param<Tensor<D>>, Option<Self::Reparam>) {
            if D != 2 {
                return (param, None);
            }
            let scale = Tensor::<1>::ones([1], &param.lazy_device());
            (
                param,
                Some(CustomScale {
                    scale: Param::from_tensor(scale),
                }),
            )
        }
    }

    #[test]
    fn custom_reparameterization_supports_full_module_lifecycle() {
        let device = test_device().autodiff();
        let model = SimpleLinear::new(4, 6, &device).apply_reparameterization(CustomScaleMapper);
        let custom = model
            .weight
            .reparameterization::<CustomScale>()
            .expect("custom reparameterization should be attached");

        model
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(&model.weight.base().into_data(), Tolerance::default());
        assert_eq!(model.num_params(), 24 + 6 + 1);

        let grads = model.weight.val().sum().backward();
        assert!(model.weight.base().grad(&grads).is_some());
        assert!(custom.scale.val().grad(&grads).is_some());

        let target = SimpleLinear::new(4, 6, &device).apply_reparameterization(CustomScaleMapper);
        let loaded = target.load_record(model.clone().into_record());
        loaded
            .weight
            .val()
            .into_data()
            .assert_approx_eq::<f32>(&model.weight.val().into_data(), Tolerance::default());

        let inference = model.valid();
        assert!(inference.weight.reparameterization_dyn().is_none());
    }
}
