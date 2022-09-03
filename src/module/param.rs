use crate::module::{ADModule, LoadingError, Module, State, StateNamed};
use crate::optim::Optimizer;
use crate::tensor::{
    backend::{ADBackend, Backend},
    Data, Gradients, Tensor,
};

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

impl<const D: usize, B: Backend> Param<Tensor<B, D>> {
    pub fn num_params(&self) -> usize {
        self.value.shape().num_elements()
    }

    pub fn update_params<O: Optimizer<Backend = B>>(&mut self, grads: &Gradients, optim: &mut O)
    where
        B: ADBackend,
    {
        optim.update(&mut self.value, grads);
    }

    pub fn devices(&self) -> Vec<B::Device> {
        vec![self.value.device()]
    }

    pub fn to_device(&mut self, device: B::Device) {
        self.value = self.value.to_device(device);
    }

    pub fn state(&self) -> State<B::Elem> {
        State::Data(self.value.to_data().serialize())
    }

    pub fn load(&mut self, state: &State<B::Elem>) -> Result<(), LoadingError> {
        match state {
            State::Data(data) => {
                self.value = Tensor::from_data_device(Data::from(data), self.value.device());
            }
            _ => return Err(LoadingError::new("Can't load tensor".to_string())),
        };

        Ok(())
    }

    pub fn inner(&self) -> Param<Tensor<B::InnerBackend, D>>
    where
        B: ADBackend,
    {
        Param::new(self.value.inner())
    }
}

impl<const D: usize, B: Backend> Param<Option<Tensor<B, D>>> {
    pub fn num_params(&self) -> usize {
        if let Some(value) = &self.value {
            return value.shape().num_elements();
        }

        0
    }

    pub fn update_params<O: Optimizer<Backend = B>>(&mut self, grads: &Gradients, optim: &mut O)
    where
        B: ADBackend,
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

    pub fn state(&self) -> State<B::Elem> {
        if let Some(value) = &self.value {
            return State::Data(value.to_data().serialize());
        }

        State::StateNamed(StateNamed::new())
    }

    pub fn load(&mut self, state: &State<B::Elem>) -> Result<(), LoadingError> {
        let data = match state {
            State::Data(data) => data,
            _ => {
                return Err(LoadingError::new(
                    "Can't load Option<Tensor> from NamedState".to_string(),
                ))
            }
        };

        if let Some(value) = &self.value {
            self.value = Some(Tensor::from_data_device(Data::from(data), value.device()));
        }

        Ok(())
    }

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

impl<M: Module> Param<M> {
    pub fn num_params(&self) -> usize {
        self.value.num_params()
    }

    pub fn update_params<O: Optimizer<Backend = M::Backend>>(
        &mut self,
        grads: &Gradients,
        optim: &mut O,
    ) where
        M::Backend: ADBackend,
    {
        self.value.update_params(grads, optim);
    }

    pub fn devices(&self) -> Vec<<M::Backend as Backend>::Device> {
        self.value.devices()
    }

    pub fn to_device(&mut self, device: <M::Backend as Backend>::Device) {
        self.value.to_device(device)
    }

    pub fn state(&self) -> State<<M::Backend as Backend>::Elem> {
        self.value.state()
    }

    pub fn load(
        &mut self,
        state: &State<<M::Backend as Backend>::Elem>,
    ) -> Result<(), LoadingError> {
        self.value.load(state)
    }

    pub fn inner(&self) -> Param<M::InnerModule>
    where
        M: ADModule,
        M::Backend: ADBackend,
    {
        Param::new(self.value.inner())
    }
}

impl<M: Module> Param<Vec<M>> {
    pub fn num_params(&self) -> usize {
        let mut num_params = 0;
        for module in self.value.iter() {
            num_params += module.num_params();
        }

        num_params
    }

    pub fn update_params<O: Optimizer<Backend = M::Backend>>(
        &mut self,
        grads: &Gradients,
        optim: &mut O,
    ) where
        M::Backend: ADBackend,
    {
        for module in self.value.iter_mut() {
            module.update_params(grads, optim);
        }
    }

    pub fn devices(&self) -> Vec<<M::Backend as Backend>::Device> {
        let mut devices = Vec::new();
        for module in self.value.iter() {
            devices.append(&mut module.devices());
        }
        devices
    }

    pub fn to_device(&mut self, device: <M::Backend as Backend>::Device) {
        for module in self.value.iter_mut() {
            module.to_device(device);
        }
    }

    pub fn state(&self) -> State<<M::Backend as Backend>::Elem> {
        let mut state = StateNamed::new();

        for (i, module) in self.value.iter().enumerate() {
            state.register_state(format!("mod-{}", i).as_str(), module.state());
        }

        State::StateNamed(state)
    }

    pub fn load(
        &mut self,
        state: &State<<M::Backend as Backend>::Elem>,
    ) -> Result<(), LoadingError> {
        let num = self.value.len();
        for (i, module) in self.value.iter_mut().enumerate() {
            module
                .load(
                    state
                        .get(format!("mod-{}", i).as_str())
                        .ok_or(LoadingError::new(format!(
                            "Invalid number of modules, expected {} modules missing #{}",
                            num, i
                        )))?,
                )
                .map_err(|err| {
                    LoadingError::new(format!("Can't load modules mod-{}: {}", i, err))
                })?;
        }

        Ok(())
    }

    pub fn inner(&self) -> Param<Vec<M::InnerModule>>
    where
        M: ADModule,
        M::Backend: ADBackend,
    {
        Param::new(self.value.iter().map(|v| v.inner()).collect())
    }
}
