//! Type-erased storage and module traversal for parameter reparameterizations.
//!
//! The public [`Reparameterization`] and [`Reparameterizer`](crate::module::Reparameterizer)
//! traits are statically typed. However,
//! [`Module::apply_reparameterization`](crate::module::Module::apply_reparameterization) must
//! attach an arbitrary user-defined reparameterization to a [`Param`] without changing the
//! parameter or containing module type. [`DynReparameterization`] is the private object-safe
//! representation stored by `Param` to make that possible.
//!
//! Reparameterizations are regular [`Module`](crate::module::Module)s, whose visitors and mappers
//! have const-generic methods and therefore aren't object-safe. Traversing their state crosses the
//! erased boundary in both directions:
//!
//! - [`ModuleVisitorToDyn`] and [`ModuleMapperToDyn`] erase the rank-specific parameters emitted
//!   by the user-defined reparameterization module.
//! - [`DynModuleVisitor`] and [`DynModuleMapper`] carry those parameters across the object-safe
//!   boundary.
//! - [`DynVisitor`] and [`DynMapper`] restore static rank dispatch and forward the parameters to
//!   the original [`ModuleVisitor`] or [`ModuleMapper`].
//!
//! This keeps all type erasure internal; users only implement the public reparameterization
//! traits.

use alloc::{boxed::Box, vec::Vec};
use burn_tensor::{Bool, Device, Int, Tensor};
use core::{any::Any, fmt::Debug};

use crate::module::{AutodiffModule, ModuleMapper, ModuleVisitor};

use super::{Param, Reparameterization};

/// Object-safe internal representation of a rank-specific [`Reparameterization`].
pub trait DynReparameterization: Debug + Send + Sync {
    /// Stable path component used for nested parameters.
    fn name(&self) -> &'static str;
    /// Materialize a type-erased tensor.
    fn materialize_dyn(&self, base: Box<dyn Any + Send>) -> Box<dyn Any + Send>;
    /// Visit nested module state.
    fn visit_dyn(&self, visitor: &mut dyn DynModuleVisitor);
    /// Map nested module state.
    fn map_dyn(self: Box<Self>, mapper: &mut dyn DynModuleMapper)
    -> Box<dyn DynReparameterization>;
    /// Move nested module state to a device.
    fn to_device_dyn(self: Box<Self>, device: &Device) -> Box<dyn DynReparameterization>;
    /// Fork nested module state to a device.
    fn fork_dyn(self: Box<Self>, device: &Device) -> Box<dyn DynReparameterization>;
    /// Convert nested module state from its inner backend.
    fn from_inner_dyn(self: Box<Self>) -> Box<dyn DynReparameterization>;
    /// Collect devices from nested module state.
    fn collect_devices_dyn(&self, devices: Vec<Device>) -> Vec<Device>;
    /// Access the concrete reparameterization for internal downcasting.
    fn as_any(&self) -> &dyn Any;
    /// Clone the dynamic value.
    fn clone_dyn(&self) -> Box<dyn DynReparameterization>;
}

impl Clone for Box<dyn DynReparameterization> {
    fn clone(&self) -> Self {
        self.clone_dyn()
    }
}

#[derive(Clone, Debug)]
/// Associates a concrete reparameterization with the rank of the parameter it is attached to.
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

/// Dispatch a runtime tensor rank back to a const-generic module visitor or mapper method.
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

/// Borrowed type-erased parameter passed through [`DynModuleVisitor`].
///
/// The runtime rank identifies the concrete `Param<Tensor<D, K>>` stored in `value`.
pub struct DynParamRef<'a> {
    rank: usize,
    value: &'a dyn Any,
}

impl<'a> DynParamRef<'a> {
    fn new<T: Any>(rank: usize, value: &'a T) -> Self {
        Self { rank, value }
    }
}

/// Owned type-erased parameter passed through [`DynModuleMapper`].
///
/// Mapping consumes and returns parameters, so this is the owned counterpart to [`DynParamRef`].
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

/// Object-safe visitor used while crossing the erased reparameterization boundary.
pub trait DynModuleVisitor {
    fn visit_float(&mut self, param: DynParamRef<'_>);
    fn visit_int(&mut self, param: DynParamRef<'_>);
    fn visit_bool(&mut self, param: DynParamRef<'_>);
    fn enter_module(&mut self, name: &str, container_type: &str);
    fn exit_module(&mut self, name: &str, container_type: &str);
}

/// Object-safe mapper used while crossing the erased reparameterization boundary.
pub trait DynModuleMapper {
    fn map_float(&mut self, param: DynParam) -> DynParam;
    fn map_int(&mut self, param: DynParam) -> DynParam;
    fn map_bool(&mut self, param: DynParam) -> DynParam;
    fn enter_module(&mut self, name: &str, container_type: &str);
    fn exit_module(&mut self, name: &str, container_type: &str);
}

/// Adapts the generic [`ModuleVisitor`] calls made by a concrete reparameterization to a dynamic
/// visitor.
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

/// Adapts the generic [`ModuleMapper`] calls made by a concrete reparameterization to a dynamic
/// mapper.
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

/// Restores erased parameter ranks and forwards them to a concrete [`ModuleVisitor`].
struct DynVisitor<'a, V> {
    visitor: &'a mut V,
}

impl<V: ModuleVisitor> DynModuleVisitor for DynVisitor<'_, V> {
    fn visit_float(&mut self, param: DynParamRef<'_>) {
        dispatch_rank!(param.rank, D => {
            self.visitor.visit_float(param.value.downcast_ref::<Param<Tensor<D>>>().unwrap());
        });
    }

    fn visit_int(&mut self, param: DynParamRef<'_>) {
        dispatch_rank!(param.rank, D => {
            self.visitor.visit_int(param.value.downcast_ref::<Param<Tensor<D, Int>>>().unwrap());
        });
    }

    fn visit_bool(&mut self, param: DynParamRef<'_>) {
        dispatch_rank!(param.rank, D => {
            self.visitor.visit_bool(param.value.downcast_ref::<Param<Tensor<D, Bool>>>().unwrap());
        });
    }

    fn enter_module(&mut self, name: &str, container_type: &str) {
        self.visitor.enter_module(name, container_type);
    }

    fn exit_module(&mut self, name: &str, container_type: &str) {
        self.visitor.exit_module(name, container_type);
    }
}

/// Restores erased parameter ranks and forwards them to a concrete [`ModuleMapper`].
struct DynMapper<'a, M> {
    mapper: &'a mut M,
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

pub(crate) fn visit<V: ModuleVisitor>(
    reparameterization: &dyn DynReparameterization,
    visitor: &mut V,
) {
    reparameterization.visit_dyn(&mut DynVisitor { visitor });
}

pub(crate) fn map<M: ModuleMapper>(
    reparameterization: Box<dyn DynReparameterization>,
    mapper: &mut M,
) -> Box<dyn DynReparameterization> {
    reparameterization.map_dyn(&mut DynMapper { mapper })
}
