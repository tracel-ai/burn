use alloc::{string::ToString, vec, vec::Vec};

use super::{load_with_id, state_with_id, Param, ParamId};
use crate::module::{
    ADModule, LoadingError, Module, ModuleMapper, ModuleVisitor, State, StateNamed,
};
use crate::tensor::{
    backend::{ADBackend, Backend},
    Data, Tensor,
};

impl<B: Backend, const D: usize> From<Tensor<B, D>> for Param<Tensor<B, D>> {
    fn from(value: Tensor<B, D>) -> Self {
        Param {
            id: ParamId::new(),
            value: value.require_grad(),
        }
    }
}

impl<B: Backend, const D: usize> From<Option<Tensor<B, D>>> for Param<Option<Tensor<B, D>>> {
    fn from(value: Option<Tensor<B, D>>) -> Self {
        Param {
            id: ParamId::new(),
            value: value.map(|tensor| tensor.require_grad()),
        }
    }
}

impl<const D: usize, B: Backend> Module for Param<Tensor<B, D>> {
    type Backend = B;

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

    fn state(&self) -> State<B::FloatElem> {
        let state = State::Data(self.value.to_data().serialize());

        state_with_id(self.id.clone(), state)
    }

    fn load(self, state: &State<B::FloatElem>) -> Result<Self, LoadingError> {
        let (id, state) = load_with_id(state)?;
        let id = id.clone();

        let tensor = match state {
            State::Data(data) => {
                Tensor::from_data_device(Data::from(data), &self.value.device()).require_grad()
            }
            _ => return Err(LoadingError::new("Can't load tensor".to_string())),
        };

        Ok(Self { id, value: tensor })
    }

    fn detach(self) -> Self {
        Self {
            id: self.id,
            value: self.value.detach().require_grad(),
        }
    }

    fn visit<V: ModuleVisitor<Self::Backend>>(&self, visitor: &mut V) {
        visitor.visit(&self.id, &self.value)
    }

    fn map<M: ModuleMapper<Self::Backend>>(self, mapper: &mut M) -> Self {
        let value = mapper.map(&self.id, self.value).require_grad();
        Self { id: self.id, value }
    }
}

impl<const D: usize, B: Backend> Module for Param<Option<Tensor<B, D>>> {
    type Backend = B;

    fn num_params(&self) -> usize {
        if let Some(value) = &self.value {
            return value.shape().num_elements();
        }

        0
    }

    fn devices(&self) -> Vec<B::Device> {
        if let Some(value) = &self.value {
            return vec![value.device()];
        }

        vec![]
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            id: self.id,
            value: self
                .value
                .map(|value| value.to_device(device).require_grad()),
        }
    }

    fn state(&self) -> State<B::FloatElem> {
        let state = match &self.value {
            Some(value) => State::Data(value.to_data().serialize()),
            None => State::StateNamed(StateNamed::new()),
        };

        state_with_id(self.id.clone(), state)
    }

    fn load(self, state: &State<B::FloatElem>) -> Result<Self, LoadingError> {
        let (id, state) = load_with_id(state)?;
        let id = id.clone();

        let data = match state {
            State::Data(data) => data,
            _ => {
                return Err(LoadingError::new(
                    "Can't load Option<Tensor> from NamedState".to_string(),
                ))
            }
        };

        let tensor = self.value.map(|tensor| {
            Tensor::from_data_device(Data::from(data), &tensor.device()).require_grad()
        });

        Ok(Self { id, value: tensor })
    }

    fn detach(self) -> Self {
        Self {
            id: self.id,
            value: self.value.map(|value| value.detach().require_grad()),
        }
    }

    fn visit<V: ModuleVisitor<Self::Backend>>(&self, visitor: &mut V) {
        if let Some(value) = &self.value {
            visitor.visit(&self.id, value)
        }
    }

    fn map<M: ModuleMapper<Self::Backend>>(self, mapper: &mut M) -> Self {
        let value = self
            .value
            .map(|value| mapper.map(&self.id, value).require_grad());
        Self { id: self.id, value }
    }
}

impl<const D: usize, B: ADBackend> ADModule for Param<Tensor<B, D>> {
    type ADBackend = B;

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
            value: Tensor::from_inner(module.value),
        }
    }
}

impl<const D: usize, B: ADBackend> ADModule for Param<Option<Tensor<B, D>>> {
    type ADBackend = B;

    type InnerModule = Param<Option<Tensor<B::InnerBackend, D>>>;

    fn inner(self) -> Self::InnerModule {
        Param {
            id: self.id,
            value: self.value.map(|val| val.inner()),
        }
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        Param {
            id: module.id,
            value: module.value.map(Tensor::from_inner),
        }
    }
}
