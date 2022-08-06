use crate::module::{Module, State};
use crate::optim::Optimizer;
use crate::tensor::{back, Gradients, Tensor};
use serde::de::DeserializeOwned;
use serde::Serialize;

#[derive(Debug)]
pub struct Param<T> {
    value: T,
}

impl<T> std::ops::Deref for Param<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> Param<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<const D: usize, B: back::Backend> Param<Tensor<D, B>> {
    pub fn num_params(&self) -> usize {
        self.value.shape().num_elements()
    }

    pub fn update_params<O: Optimizer<B>>(&mut self, grads: &Gradients, optim: &mut O)
    where
        B: back::ad::Backend,
    {
        optim.update(&mut self.value, grads);
    }

    pub fn devices(&self) -> Vec<B::Device> {
        vec![self.value.device()]
    }

    pub fn to_device(&mut self, device: B::Device) {
        self.value = self.value.to_device(device);
    }

    pub fn state(&self, name: &str) -> State<B>
    where
        B::Elem: Serialize,
        B::Elem: DeserializeOwned,
    {
        let mut state = State::new(name);
        state.register(self.value.to_data().serialize());
        state
    }
    pub fn load_from_parent(&mut self, name: &str, state: &State<B>)
    where
        B::Elem: Serialize,
        B::Elem: DeserializeOwned,
    {
        let data = state.get(name);
        self.value = Tensor::from_data_device(data, self.value.device());
    }
}

impl<const D: usize, B: back::Backend> Param<Option<Tensor<D, B>>> {
    pub fn num_params(&self) -> usize {
        if let Some(value) = &self.value {
            return value.shape().num_elements();
        }

        0
    }

    pub fn update_params<O: Optimizer<B>>(&mut self, grads: &Gradients, optim: &mut O)
    where
        B: back::ad::Backend,
    {
        if let Some(value) = &mut self.value {
            optim.update(value, grads);
        }
    }

    pub fn devices(&self) -> Vec<B::Device> {
        if let Some(value) = &self.value {
            return vec![value.device()];
        }

        vec![]
    }

    pub fn to_device(&mut self, device: B::Device) {
        if let Some(value) = &self.value {
            self.value = Some(value.to_device(device));
        }
    }

    pub fn state(&self, name: &str) -> State<B>
    where
        B::Elem: Serialize,
        B::Elem: DeserializeOwned,
    {
        let mut state = State::new(name);
        if let Some(value) = &self.value {
            state.register(value.to_data().serialize());
        }
        state
    }

    pub fn load_from_parent(&mut self, name: &str, state: &State<B>)
    where
        B::Elem: Serialize,
        B::Elem: DeserializeOwned,
    {
        let value = match &self.value {
            Some(value) => Some(Tensor::from_data_device(state.get(name), value.device())),
            None => None,
        };

        self.value = value;
    }
}

impl<M: Module> Param<M> {
    pub fn num_params(&self) -> usize {
        self.value.num_params()
    }

    pub fn update_params<O: Optimizer<M::Backend>>(&mut self, grads: &Gradients, optim: &mut O)
    where
        M::Backend: back::ad::Backend,
    {
        self.value.update_params(grads, optim);
    }

    pub fn devices(&self) -> Vec<<M::Backend as back::Backend>::Device> {
        self.value.devices()
    }

    pub fn to_device(&mut self, device: <M::Backend as back::Backend>::Device) {
        self.value.to_device(device)
    }

    pub fn state(&self, name: &str) -> State<M::Backend>
    where
        <M::Backend as back::Backend>::Elem: Serialize,
        <M::Backend as back::Backend>::Elem: DeserializeOwned,
    {
        let mut state = State::new(name);
        state.register_child(self.value.state());
        state
    }

    pub fn load_from_parent(&mut self, name: &str, state: &State<M::Backend>)
    where
        <M::Backend as back::Backend>::Elem: Serialize,
        <M::Backend as back::Backend>::Elem: DeserializeOwned,
    {
        self.value.load_from_parent(name, state);
    }
}
