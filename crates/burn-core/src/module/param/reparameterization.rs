use alloc::{
    boxed::Box,
    string::{String, ToString},
    vec::Vec,
};
use burn_tensor::{Bool, Device, Int, Tensor};
use core::{any::Any, fmt::Debug};

use crate::module::{AutodiffModule, ModuleMapper, ModuleVisitor};

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

/// Object-safe internal representation of a rank-specific [`Reparameterization`].
#[doc(hidden)]
pub trait DynReparameterization: Debug + Send + Sync {
    /// Stable path component used for nested parameters.
    fn name(&self) -> &'static str;
    /// Materialize a type-erased tensor.
    #[doc(hidden)]
    fn materialize_dyn(&self, base: Box<dyn Any + Send>) -> Box<dyn Any + Send>;
    /// Visit nested module state.
    #[doc(hidden)]
    fn visit_dyn(&self, visitor: &mut dyn DynModuleVisitor);
    /// Map nested module state.
    #[doc(hidden)]
    fn map_dyn(self: Box<Self>, mapper: &mut dyn DynModuleMapper)
    -> Box<dyn DynReparameterization>;
    /// Move nested module state to a device.
    #[doc(hidden)]
    fn to_device_dyn(self: Box<Self>, device: &Device) -> Box<dyn DynReparameterization>;
    /// Fork nested module state to a device.
    #[doc(hidden)]
    fn fork_dyn(self: Box<Self>, device: &Device) -> Box<dyn DynReparameterization>;
    /// Convert nested module state from its inner backend.
    #[doc(hidden)]
    fn from_inner_dyn(self: Box<Self>) -> Box<dyn DynReparameterization>;
    /// Collect devices from nested module state.
    #[doc(hidden)]
    fn collect_devices_dyn(&self, devices: Vec<Device>) -> Vec<Device>;
    /// Access the concrete reparameterization for internal downcasting.
    #[doc(hidden)]
    fn as_any(&self) -> &dyn Any;
    /// Clone the dynamic value.
    #[doc(hidden)]
    fn clone_dyn(&self) -> Box<dyn DynReparameterization>;
}

impl Clone for Box<dyn DynReparameterization> {
    fn clone(&self) -> Self {
        self.clone_dyn()
    }
}

#[derive(Clone, Debug)]
struct ReparameterizationAdapter<R, const D: usize> {
    inner: R,
}

impl<R, const D: usize> ReparameterizationAdapter<R, D>
where
    R: Reparameterization,
{
    fn new(inner: R) -> Self {
        Self { inner }
    }
}

impl<R, const D: usize> DynReparameterization for ReparameterizationAdapter<R, D>
where
    R: Reparameterization,
{
    fn name(&self) -> &'static str {
        R::NAME
    }

    fn materialize_dyn(&self, base: Box<dyn Any + Send>) -> Box<dyn Any + Send> {
        let base = *base
            .downcast::<Tensor<D>>()
            .expect("Reparameterization tensor should match its attached rank");
        Box::new(self.inner.materialize(base))
    }

    fn visit_dyn(&self, visitor: &mut dyn DynModuleVisitor) {
        self.inner.visit(&mut ModuleVisitorToDyn { inner: visitor });
    }

    fn map_dyn(
        self: Box<Self>,
        mapper: &mut dyn DynModuleMapper,
    ) -> Box<dyn DynReparameterization> {
        Box::new(Self::new(
            self.inner.map(&mut ModuleMapperToDyn { inner: mapper }),
        ))
    }

    fn to_device_dyn(self: Box<Self>, device: &Device) -> Box<dyn DynReparameterization> {
        Box::new(Self::new(self.inner.to_device(device)))
    }

    fn fork_dyn(self: Box<Self>, device: &Device) -> Box<dyn DynReparameterization> {
        Box::new(Self::new(self.inner.fork(device)))
    }

    fn from_inner_dyn(self: Box<Self>) -> Box<dyn DynReparameterization> {
        Box::new(Self::new(AutodiffModule::from_inner(self.inner)))
    }

    fn collect_devices_dyn(&self, devices: Vec<Device>) -> Vec<Device> {
        self.inner.collect_devices(devices)
    }

    fn as_any(&self) -> &dyn Any {
        &self.inner
    }

    fn clone_dyn(&self) -> Box<dyn DynReparameterization> {
        Box::new(self.clone())
    }
}

pub(crate) fn boxed<R, const D: usize>(value: R) -> Box<dyn DynReparameterization>
where
    R: Reparameterization,
{
    Box::new(ReparameterizationAdapter::<R, D>::new(value))
}

macro_rules! dispatch_rank {
    ($rank:expr, $d:ident => $body:block) => {
        match $rank {
            0 => {
                const $d: usize = 0;
                $body
            }
            1 => {
                const $d: usize = 1;
                $body
            }
            2 => {
                const $d: usize = 2;
                $body
            }
            3 => {
                const $d: usize = 3;
                $body
            }
            4 => {
                const $d: usize = 4;
                $body
            }
            5 => {
                const $d: usize = 5;
                $body
            }
            6 => {
                const $d: usize = 6;
                $body
            }
            7 => {
                const $d: usize = 7;
                $body
            }
            8 => {
                const $d: usize = 8;
                $body
            }
            other => panic!("Unsupported reparameterization tensor rank: {other}"),
        }
    };
}

#[doc(hidden)]
pub struct DynParamRef<'a> {
    rank: usize,
    value: &'a dyn Any,
}

impl<'a> DynParamRef<'a> {
    fn new<T: Any>(rank: usize, value: &'a T) -> Self {
        Self { rank, value }
    }
}

#[doc(hidden)]
pub struct DynParam {
    rank: usize,
    value: Box<dyn Any + Send>,
}

impl DynParam {
    fn new<T: Any + Send>(rank: usize, value: T) -> Self {
        Self {
            rank,
            value: Box::new(value),
        }
    }

    fn downcast<T: Any + Send>(self) -> T {
        *self
            .value
            .downcast()
            .expect("Dynamic parameter type should match its rank and kind")
    }
}

#[doc(hidden)]
pub trait DynModuleVisitor {
    fn visit_float(&mut self, param: DynParamRef<'_>);
    fn visit_int(&mut self, param: DynParamRef<'_>);
    fn visit_bool(&mut self, param: DynParamRef<'_>);
    fn enter_module(&mut self, name: &str, container_type: &str);
    fn exit_module(&mut self, name: &str, container_type: &str);
}

#[doc(hidden)]
pub trait DynModuleMapper {
    fn map_float(&mut self, param: DynParam) -> DynParam;
    fn map_int(&mut self, param: DynParam) -> DynParam;
    fn map_bool(&mut self, param: DynParam) -> DynParam;
    fn enter_module(&mut self, name: &str, container_type: &str);
    fn exit_module(&mut self, name: &str, container_type: &str);
}

struct ModuleVisitorToDyn<'a> {
    inner: &'a mut dyn DynModuleVisitor,
}

impl ModuleVisitor for ModuleVisitorToDyn<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.inner.visit_float(DynParamRef::new(D, param));
    }

    fn visit_int<const D: usize>(&mut self, param: &Param<Tensor<D, Int>>) {
        self.inner.visit_int(DynParamRef::new(D, param));
    }

    fn visit_bool<const D: usize>(&mut self, param: &Param<Tensor<D, Bool>>) {
        self.inner.visit_bool(DynParamRef::new(D, param));
    }

    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.inner.enter_module(name, container_type);
    }

    fn exit_module(&mut self, name: &str, container_type: &str) {
        self.inner.exit_module(name, container_type);
    }
}

struct ModuleMapperToDyn<'a> {
    inner: &'a mut dyn DynModuleMapper,
}

impl ModuleMapper for ModuleMapperToDyn<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        self.inner.map_float(DynParam::new(D, param)).downcast()
    }

    fn map_int<const D: usize>(&mut self, param: Param<Tensor<D, Int>>) -> Param<Tensor<D, Int>> {
        self.inner.map_int(DynParam::new(D, param)).downcast()
    }

    fn map_bool<const D: usize>(
        &mut self,
        param: Param<Tensor<D, Bool>>,
    ) -> Param<Tensor<D, Bool>> {
        self.inner.map_bool(DynParam::new(D, param)).downcast()
    }

    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.inner.enter_module(name, container_type);
    }

    fn exit_module(&mut self, name: &str, container_type: &str) {
        self.inner.exit_module(name, container_type);
    }
}

pub(crate) struct DynVisitor<'a, V> {
    pub visitor: &'a mut V,
}

impl<V: ModuleVisitor> DynModuleVisitor for DynVisitor<'_, V> {
    fn visit_float(&mut self, param: DynParamRef<'_>) {
        dispatch_rank!(param.rank, D => {
            self.visitor.visit_float(param.value.downcast_ref::<Param<Tensor<D>>>().unwrap());
        })
    }

    fn visit_int(&mut self, param: DynParamRef<'_>) {
        dispatch_rank!(param.rank, D => {
            self.visitor.visit_int(param.value.downcast_ref::<Param<Tensor<D, Int>>>().unwrap());
        })
    }

    fn visit_bool(&mut self, param: DynParamRef<'_>) {
        dispatch_rank!(param.rank, D => {
            self.visitor.visit_bool(param.value.downcast_ref::<Param<Tensor<D, Bool>>>().unwrap());
        })
    }

    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.visitor.enter_module(name, container_type);
    }

    fn exit_module(&mut self, name: &str, container_type: &str) {
        self.visitor.exit_module(name, container_type);
    }
}

pub(crate) struct DynMapper<'a, M> {
    pub mapper: &'a mut M,
}

impl<M: ModuleMapper> DynModuleMapper for DynMapper<'_, M> {
    fn map_float(&mut self, param: DynParam) -> DynParam {
        let rank = param.rank;
        dispatch_rank!(rank, D => {
            DynParam::new(D, self.mapper.map_float(param.downcast::<Param<Tensor<D>>>() ))
        })
    }

    fn map_int(&mut self, param: DynParam) -> DynParam {
        let rank = param.rank;
        dispatch_rank!(rank, D => {
            DynParam::new(D, self.mapper.map_int(param.downcast::<Param<Tensor<D, Int>>>() ))
        })
    }

    fn map_bool(&mut self, param: DynParam) -> DynParam {
        let rank = param.rank;
        dispatch_rank!(rank, D => {
            DynParam::new(D, self.mapper.map_bool(param.downcast::<Param<Tensor<D, Bool>>>() ))
        })
    }

    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.mapper.enter_module(name, container_type);
    }

    fn exit_module(&mut self, name: &str, container_type: &str) {
        self.mapper.exit_module(name, container_type);
    }
}

#[cfg(all(test, feature = "autodiff"))]
mod tests {
    use super::*;
    use crate as burn;
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
