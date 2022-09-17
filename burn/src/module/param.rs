use crate::module::{ADModule, LoadingError, Module, State, StateNamed};
use crate::optim::Optimizer;
use crate::tensor::{
    backend::{ADBackend, Backend},
    Data, Element, Gradients, Tensor,
};

#[derive(Debug)]
pub struct Param<T> {
    pub id: ParamId,
    value: T,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct ParamId {
    pub(crate) value: String,
}

impl Default for ParamId {
    fn default() -> Self {
        Self::new()
    }
}

impl ParamId {
    pub fn new() -> Self {
        Self {
            value: nanoid::nanoid!(),
        }
    }
}

impl std::fmt::Display for ParamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.value.as_str())
    }
}

impl<T> std::ops::Deref for Param<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> Param<T> {
    pub fn new(value: T) -> Self {
        Self {
            id: ParamId::new(),
            value,
        }
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
        optim.update(&self.id, &mut self.value, grads);
    }

    pub fn load_optim_state<O: Optimizer<Backend = B>>(
        &self,
        optim: &mut O,
        state_optim: &StateNamed<B::Elem>,
    ) where
        B: ADBackend,
    {
        optim.load_state::<D>(&self.id, state_optim, &self.value.device());
    }

    pub fn devices(&self) -> Vec<B::Device> {
        vec![self.value.device()]
    }

    pub fn to_device(&mut self, device: B::Device) {
        self.value = self.value.to_device(device);
    }

    pub fn state(&self) -> State<B::Elem> {
        let state = State::Data(self.value.to_data().serialize());

        state_with_id(self.id.clone(), state)
    }

    pub fn load(&mut self, state: &State<B::Elem>) -> Result<(), LoadingError> {
        let (id, state) = load_with_id(state)?;
        self.id = id.clone();

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
            optim.update(&self.id, value, grads);
        }
    }

    pub fn load_optim_state<O: Optimizer<Backend = B>>(
        &self,
        optim: &mut O,
        state_optim: &StateNamed<B::Elem>,
    ) where
        B: ADBackend,
    {
        if let Some(value) = &self.value {
            optim.load_state::<D>(&self.id, state_optim, &value.device());
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
        let state = match &self.value {
            Some(value) => State::Data(value.to_data().serialize()),
            None => State::StateNamed(StateNamed::new()),
        };

        state_with_id(self.id.clone(), state)
    }

    pub fn load(&mut self, state: &State<B::Elem>) -> Result<(), LoadingError> {
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

    pub fn load_optim_state<O: Optimizer<Backend = M::Backend>>(
        &self,
        optim: &mut O,
        state_optim: &StateNamed<<M::Backend as Backend>::Elem>,
    ) where
        M::Backend: ADBackend,
    {
        self.value.load_optim_state(optim, state_optim);
    }

    pub fn devices(&self) -> Vec<<M::Backend as Backend>::Device> {
        self.value.devices()
    }

    pub fn to_device(&mut self, device: <M::Backend as Backend>::Device) {
        self.value.to_device(device)
    }

    pub fn state(&self) -> State<<M::Backend as Backend>::Elem> {
        let state = self.value.state();

        state_with_id(self.id.clone(), state)
    }

    pub fn load(
        &mut self,
        state: &State<<M::Backend as Backend>::Elem>,
    ) -> Result<(), LoadingError> {
        let (id, state) = load_with_id(state)?;
        self.id = id.clone();

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

    pub fn load_optim_state<O: Optimizer<Backend = M::Backend>>(
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

        let state = State::StateNamed(state);

        state_with_id(self.id.clone(), state)
    }

    pub fn load(
        &mut self,
        state: &State<<M::Backend as Backend>::Elem>,
    ) -> Result<(), LoadingError> {
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

    pub fn inner(&self) -> Param<Vec<M::InnerModule>>
    where
        M: ADModule,
        M::Backend: ADBackend,
    {
        Param::new(self.value.iter().map(|v| v.inner()).collect())
    }
}

fn state_with_id<E: Element>(id: ParamId, state: State<E>) -> State<E> {
    let mut state_wrapper = StateNamed::new();

    state_wrapper.register_state("data", state);
    state_wrapper.register_state("id", State::ParamId(id));

    State::StateNamed(state_wrapper)
}

fn load_with_id<E: Element>(state: &State<E>) -> Result<(&ParamId, &State<E>), LoadingError> {
    let state_wrapper = match state {
        State::StateNamed(state) => state,
        _ => {
            return Err(LoadingError::new(
                "Can't load state wrapper to fetch id and data".to_string(),
            ))
        }
    };

    let state = match state_wrapper.get("data") {
        Some(state) => state.clone(),
        None => {
            return Err(LoadingError::new(
                "Can't load state data from state wrapper".to_string(),
            ))
        }
    };

    let id = match state_wrapper.get("id") {
        Some(State::ParamId(id)) => id,
        _ => {
            return Err(LoadingError::new(
                "Can't load state id from state wrapper".to_string(),
            ))
        }
    };

    Ok((id, state))
}
