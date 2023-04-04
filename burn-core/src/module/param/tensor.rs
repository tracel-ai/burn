use alloc::{vec, vec::Vec};

use super::{Param, ParamId};
use crate::module::{ADModule, Module, ModuleMapper, ModuleVisitor};
use crate::tensor::{
    backend::{ADBackend, Backend},
    Tensor,
};

impl<B: Backend, const D: usize> From<Tensor<B, D>> for Param<Tensor<B, D>> {
    fn from(value: Tensor<B, D>) -> Self {
        Param {
            id: ParamId::new(),
            value: value.require_grad(),
        }
    }
}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D>> {
    type Record = Param<Tensor<B, D>>;

    fn num_params(&self) -> usize {
        self.value.shape().num_elements()
    }

    fn devices(&self) -> Vec<B::Device> {
        vec![self.value.device()]
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            id: self.id,
            value: self.value.to_device(device).require_grad(),
        }
    }

    fn detach(self) -> Self {
        Self {
            id: self.id,
            value: self.value.detach().require_grad(),
        }
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit(&self.id, &self.value)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let value = mapper.map(&self.id, self.value).require_grad();
        Self { id: self.id, value }
    }

    fn into_record(self) -> Self::Record {
        self
    }

    fn load_record(self, record: Self::Record) -> Self {
        record.to_device(&self.device())
    }
}

impl<const D: usize, B: ADBackend> ADModule<B> for Param<Tensor<B, D>> {
    type InnerModule = Param<Tensor<B::InnerBackend, D>>;

    fn inner(self) -> Self::InnerModule {
        Param {
            id: self.id,
            value: self.value.inner(),
        }
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        Param {
            id: module.id,
            value: Tensor::from_inner(module.value).require_grad(),
        }
    }
}
