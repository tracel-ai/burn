use super::{Param, ParamId};
use crate::module::{ADModule, Module, ModuleMapper, ModuleVisitor};
use crate::tensor::{
    backend::{ADBackend, Backend},
    Tensor,
};

impl<B: Backend, const D: usize> From<Tensor<B, D>> for Param<Tensor<B, D>> {
    fn from(value: Tensor<B, D>) -> Self {
        Param::new(ParamId::new(), value.require_grad(), true)
    }
}

impl<B: Backend, const D: usize> Param<Tensor<B, D>> {
    /// The tensor won't have gradient when the backward pass is calculated.
    pub fn no_grad(mut self) -> Self {
        self.require_grad = false;
        self.value = self.value.set_require_grad(false);
        self
    }
}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D>> {
    type Record = Param<Tensor<B, D>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit(&self.id, &self.value)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let value = mapper.map(&self.id, self.value);
        Self::new(self.id, value, self.require_grad)
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
        Param::new(self.id, self.value.inner(), self.require_grad)
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        let mut value = module.value;
        if module.require_grad {
            value = value.require_grad();
        }
        Param::new(module.id, Tensor::from_inner(value), module.require_grad)
    }
}
