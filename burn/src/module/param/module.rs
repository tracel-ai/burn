use super::{load_with_id, state_with_id, Param};
use crate::module::{ADModule, LoadingError, Module, State, StateNamed};
use crate::optim::Optimizer;
use crate::tensor::{
    backend::{ADBackend, Backend},
    Gradients,
};

impl<M: Module> Module for Param<M> {
    type Backend = M::Backend;

    fn num_params(&self) -> usize {
        self.value.num_params()
    }

    fn update_params<O: Optimizer<Backend = M::Backend>>(
        &mut self,
        grads: &Gradients,
        optim: &mut O,
    ) where
        M::Backend: ADBackend,
    {
        self.value.update_params(grads, optim);
    }

    fn load_optim_state<O: Optimizer<Backend = M::Backend>>(
        &self,
        optim: &mut O,
        state_optim: &StateNamed<<M::Backend as Backend>::Elem>,
    ) where
        M::Backend: ADBackend,
    {
        self.value.load_optim_state(optim, state_optim);
    }

    fn register_optim_state<O: Optimizer<Backend = M::Backend>>(
        &self,
        optim: &O,
        state_optim: &mut StateNamed<<M::Backend as Backend>::Elem>,
    ) where
        M::Backend: ADBackend,
    {
        self.value.register_optim_state(optim, state_optim);
    }

    fn devices(&self) -> Vec<<M::Backend as Backend>::Device> {
        self.value.devices()
    }

    fn to_device(&mut self, device: <Self::Backend as Backend>::Device) {
        self.value.to_device(device)
    }

    fn state(&self) -> State<<M::Backend as Backend>::Elem> {
        let state = self.value.state();

        state_with_id(self.id.clone(), state)
    }

    fn load(&mut self, state: &State<<M::Backend as Backend>::Elem>) -> Result<(), LoadingError> {
        let (id, state) = load_with_id(state)?;
        self.id = id.clone();

        self.value.load(state)
    }
}

impl<M: Module> Module for Param<Vec<M>> {
    type Backend = M::Backend;

    fn num_params(&self) -> usize {
        let mut num_params = 0;
        for module in self.value.iter() {
            num_params += module.num_params();
        }

        num_params
    }

    fn update_params<O: Optimizer<Backend = M::Backend>>(
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

    fn load_optim_state<O: Optimizer<Backend = M::Backend>>(
        &self,
        optim: &mut O,
        state_optim: &StateNamed<<M::Backend as Backend>::Elem>,
    ) where
        M::Backend: ADBackend,
    {
        for module in self.value.iter() {
            module.load_optim_state(optim, state_optim);
        }
    }
    fn register_optim_state<O: Optimizer<Backend = M::Backend>>(
        &self,
        optim: &O,
        state_optim: &mut StateNamed<<M::Backend as Backend>::Elem>,
    ) where
        M::Backend: ADBackend,
    {
        for module in self.value.iter() {
            module.register_optim_state(optim, state_optim);
        }
    }

    fn devices(&self) -> Vec<<M::Backend as Backend>::Device> {
        let mut devices = Vec::new();
        for module in self.value.iter() {
            devices.append(&mut module.devices());
        }
        devices
    }

    fn to_device(&mut self, device: <M::Backend as Backend>::Device) {
        for module in self.value.iter_mut() {
            module.to_device(device);
        }
    }

    fn state(&self) -> State<<M::Backend as Backend>::Elem> {
        let mut state = StateNamed::new();

        for (i, module) in self.value.iter().enumerate() {
            state.register_state(format!("mod-{}", i).as_str(), module.state());
        }

        let state = State::StateNamed(state);

        state_with_id(self.id.clone(), state)
    }

    fn load(&mut self, state: &State<<M::Backend as Backend>::Elem>) -> Result<(), LoadingError> {
        let (id, state) = load_with_id(state)?;
        self.id = id.clone();

        let num = self.value.len();
        for (i, module) in self.value.iter_mut().enumerate() {
            module
                .load(state.get(format!("mod-{}", i).as_str()).ok_or_else(|| {
                    LoadingError::new(format!(
                        "Invalid number of modules, expected {} modules missing #{}",
                        num, i
                    ))
                })?)
                .map_err(|err| {
                    LoadingError::new(format!("Can't load modules mod-{}: {}", i, err))
                })?;
        }

        Ok(())
    }
}

impl<M: Module> Param<Vec<M>> {
    pub fn inner(&self) -> Param<Vec<M::InnerModule>>
    where
        M: ADModule,
        M::Backend: ADBackend,
    {
        Param::new(self.value.iter().map(|v| v.inner()).collect())
    }
}

impl<M: Module> Param<M> {
    pub fn inner(&self) -> Param<M::InnerModule>
    where
        M: ADModule,
        M::Backend: ADBackend,
    {
        Param::new(self.value.inner())
    }
}
