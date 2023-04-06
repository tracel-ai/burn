use super::{Param, ParamId, Untracked};
use crate::module::{ADModule, Module, ModuleMapper, ModuleVisitor};
use crate::tensor::{
    backend::{ADBackend, Backend},
    Tensor,
};

// -- Tracked tensors -- //

impl<B: Backend, const D: usize> From<Tensor<B, D>> for Param<Tensor<B, D>> {
    fn from(value: Tensor<B, D>) -> Self {
        Param::new(ParamId::new(), value.require_grad())
    }
}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D>> {
    type Record = Param<Tensor<B, D>>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit(&self.id, &self.value)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let value = mapper.map(&self.id, self.value);
        Self::new(self.id, value)
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
        Param::new(self.id, self.value.inner())
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        Param::new(module.id, Tensor::from_inner(module.value).require_grad())
    }
}

// -- Untracked tensors -- //

impl<B: Backend, const D: usize> From<Tensor<B, D>> for Param<Tensor<B, D>, Untracked> {
    fn from(value: Tensor<B, D>) -> Self {
        Param::new(ParamId::new(), value)
    }
}

impl<const D: usize, B: Backend> Module<B> for Param<Tensor<B, D>, Untracked> {
    type Record = Param<Tensor<B, D>, Untracked>;

    fn require_grad(self) -> Self {
        // Nop
        self
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        visitor.visit(&self.id, &self.value)
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let value = mapper.map(&self.id, self.value);
        Self::new(self.id, value)
    }

    fn into_record(self) -> Self::Record {
        self
    }

    fn load_record(self, record: Self::Record) -> Self {
        record.to_device(&self.device())
    }
}

impl<const D: usize, B: ADBackend> ADModule<B> for Param<Tensor<B, D>, Untracked> {
    type InnerModule = Param<Tensor<B::InnerBackend, D>>;

    fn inner(self) -> Self::InnerModule {
        Param::new(self.id, self.value.inner())
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        Param::new(module.id, Tensor::from_inner(module.value))
    }
}
