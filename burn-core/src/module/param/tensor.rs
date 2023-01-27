use super::{load_with_id, state_with_id, Param};
use crate::module::{LoadingError, Module, ModuleVisitor, ModuleVisitorMut, State, StateNamed};
use crate::tensor::{
    backend::{ADBackend, Backend},
    Data, Tensor,
};

impl<const D: usize, B: Backend> Module for Param<Tensor<B, D>> {
    type Backend = B;

    fn num_params(&self) -> usize {
        self.value.shape().num_elements()
    }

    fn devices(&self) -> Vec<B::Device> {
        vec![self.value.device()]
    }

    fn to_device(&mut self, device: &B::Device) {
        self.value = self.value.to_device(device);
    }

    fn state(&self) -> State<B::Elem> {
        let state = State::Data(self.value.to_data().serialize());

        state_with_id(self.id.clone(), state)
    }

    fn load(&mut self, state: &State<B::Elem>) -> Result<(), LoadingError> {
        let (id, state) = load_with_id(state)?;
        self.id = id.clone();

        match state {
            State::Data(data) => {
                self.value = Tensor::from_data_device(Data::from(data), &self.value.device());
            }
            _ => return Err(LoadingError::new("Can't load tensor".to_string())),
        };

        Ok(())
    }

    fn detach(&mut self) {
        self.value = self.value.clone().detach()
    }

    fn visit<V: ModuleVisitor<Self::Backend>>(&self, visitor: &mut V) {
        visitor.visit(&self.id, &self.value)
    }

    fn visit_mut<V: ModuleVisitorMut<Self::Backend>>(&mut self, visitor: &mut V) {
        visitor.visit_mut(&self.id, &mut self.value)
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

    fn to_device(&mut self, device: &B::Device) {
        if let Some(value) = &self.value {
            self.value = Some(value.to_device(device));
        }
    }

    fn state(&self) -> State<B::Elem> {
        let state = match &self.value {
            Some(value) => State::Data(value.to_data().serialize()),
            None => State::StateNamed(StateNamed::new()),
        };

        state_with_id(self.id.clone(), state)
    }

    fn load(&mut self, state: &State<B::Elem>) -> Result<(), LoadingError> {
        let (id, state) = load_with_id(state)?;
        self.id = id.clone();

        let data = match state {
            State::Data(data) => data,
            _ => {
                return Err(LoadingError::new(
                    "Can't load Option<Tensor> from NamedState".to_string(),
                ))
            }
        };

        if let Some(value) = &self.value {
            self.value = Some(Tensor::from_data_device(Data::from(data), &value.device()));
        }

        Ok(())
    }

    fn detach(&mut self) {
        self.value = self.value.clone().map(|tensor| tensor.detach());
    }

    fn visit<V: ModuleVisitor<Self::Backend>>(&self, visitor: &mut V) {
        if let Some(value) = &self.value {
            visitor.visit(&self.id, value)
        }
    }

    fn visit_mut<V: ModuleVisitorMut<Self::Backend>>(&mut self, visitor: &mut V) {
        if let Some(value) = &mut self.value {
            visitor.visit_mut(&self.id, value)
        }
    }
}

impl<const D: usize, B: Backend> Param<Tensor<B, D>> {
    pub fn inner(&self) -> Param<Tensor<B::InnerBackend, D>>
    where
        B: ADBackend,
    {
        Param::new(self.value.inner())
    }
}

impl<const D: usize, B: Backend> Param<Option<Tensor<B, D>>> {
    pub fn inner(&self) -> Param<Option<Tensor<B::InnerBackend, D>>>
    where
        B: ADBackend,
    {
        match &self.value {
            Some(tensor) => Param::new(Some(tensor.inner())),
            None => Param::new(None),
        }
    }
}
