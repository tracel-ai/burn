use crate::module::{
    ADModule, LoadingError, Module, ModuleMapper, ModuleVisitor, State, StateNamed,
};
use alloc::format;
use alloc::vec::Vec;
use burn_tensor::backend::{ADBackend, Backend};
use core::fmt::Debug;

impl<T, B> Module<B> for Option<T>
where
    T: Module<B> + Debug + Send + Sync + Clone,
    B: Backend,
{
    fn devices(&self) -> Vec<<B as burn_tensor::backend::Backend>::Device> {
        if let Some(module) = self {
            return Module::<B>::devices(module);
        }

        Vec::new()
    }

    fn to_device(self, device: &<B as burn_tensor::backend::Backend>::Device) -> Self {
        self.map(|module| module.to_device(device))
    }

    fn load(self, state: &State<B::FloatElem>) -> Result<Self, LoadingError> {
        self.map(|module| module.load(state).map(|val| Some(val)))
            .unwrap_or(Ok(None))
    }

    fn state(&self) -> State<B::FloatElem> {
        if let Some(module) = self {
            return module.state();
        }

        State::StateNamed(StateNamed::new())
    }

    fn detach(self) -> Self {
        self.map(|module| module.detach())
    }

    fn num_params(&self) -> usize {
        match &self {
            Some(module) => module.num_params(),
            None => 0,
        }
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        if let Some(module) = self {
            module.visit(visitor)
        }
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        self.map(|module| module.map(mapper))
    }
}

impl<T, B> ADModule<B> for Option<T>
where
    T: ADModule<B> + Debug + Send + Sync + Clone,
    B: ADBackend,
{
    type InnerModule = Option<T::InnerModule>;

    fn inner(self) -> Self::InnerModule {
        self.map(|module| module.inner())
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module.map(|module| T::from_inner(module))
    }
}

impl<T, B> Module<B> for Vec<T>
where
    T: Module<B> + Debug + Send + Sync + Clone,
    B: Backend,
{
    fn devices(&self) -> Vec<<B as burn_tensor::backend::Backend>::Device> {
        let mut devices = Vec::new();
        for module in self.iter() {
            devices.append(&mut module.devices());
        }
        devices
    }

    fn to_device(self, device: &<B as burn_tensor::backend::Backend>::Device) -> Self {
        self.into_iter().map(|val| val.to_device(device)).collect()
    }

    fn load(self, state: &State<B::FloatElem>) -> Result<Self, LoadingError> {
        let num = self.len();
        let mut modules = Vec::with_capacity(num);

        for (i, module) in self.into_iter().enumerate() {
            let module = module
                .load(state.get(format!("mod-{i}").as_str()).ok_or_else(|| {
                    LoadingError::new(format!(
                        "Invalid number of modules, expected {num} modules missing #{i}"
                    ))
                })?)
                .map_err(|err| LoadingError::new(format!("Can't load modules mod-{i}: {err}")))?;

            modules.push(module);
        }

        Ok(modules)
    }

    fn state(&self) -> State<B::FloatElem> {
        let mut state = StateNamed::new();

        for (i, module) in self.iter().enumerate() {
            state.register_state(format!("mod-{i}").as_str(), module.state());
        }

        State::StateNamed(state)
    }

    fn detach(self) -> Self {
        self.into_iter().map(|module| module.detach()).collect()
    }

    fn num_params(&self) -> usize {
        let mut num_params = 0;
        for module in self.iter() {
            num_params += module.num_params();
        }

        num_params
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.iter().for_each(|module| {
            module.visit(visitor);
        });
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        self.into_iter().map(|module| module.map(mapper)).collect()
    }
}

impl<T, B> ADModule<B> for Vec<T>
where
    T: ADModule<B> + Debug + Send + Sync + Clone,
    B: ADBackend,
{
    type InnerModule = Vec<T::InnerModule>;

    fn inner(self) -> Self::InnerModule {
        self.into_iter().map(|module| module.inner()).collect()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
            .into_iter()
            .map(|module| T::from_inner(module))
            .collect()
    }
}

impl<const N: usize, T, B> Module<B> for [T; N]
where
    T: Module<B> + Debug + Send + Sync + Clone + Copy,
    B: Backend,
{
    fn devices(&self) -> Vec<<B as burn_tensor::backend::Backend>::Device> {
        let mut devices = Vec::new();
        for module in self.iter() {
            devices.append(&mut module.devices());
        }
        devices
    }

    fn to_device(self, device: &<B as burn_tensor::backend::Backend>::Device) -> Self {
        self.map(|val| val.to_device(device))
    }

    fn load(mut self, state: &State<B::FloatElem>) -> Result<Self, LoadingError> {
        let num = self.len();

        for (i, module) in self.into_iter().enumerate().take(N) {
            self[i] = module
                .load(state.get(format!("mod-{i}").as_str()).ok_or_else(|| {
                    LoadingError::new(format!(
                        "Invalid number of modules, expected {num} modules missing #{i}"
                    ))
                })?)
                .map_err(|err| LoadingError::new(format!("Can't load modules mod-{i}: {err}")))?;
        }

        Ok(self)
    }

    fn state(&self) -> State<B::FloatElem> {
        let mut state = StateNamed::new();

        for (i, module) in self.iter().enumerate() {
            state.register_state(format!("mod-{i}").as_str(), module.state());
        }

        State::StateNamed(state)
    }

    fn detach(self) -> Self {
        self.map(|module| module.detach())
    }

    fn num_params(&self) -> usize {
        let mut num_params = 0;
        for module in self.iter() {
            num_params += module.num_params();
        }

        num_params
    }

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.iter().for_each(|module| {
            module.visit(visitor);
        });
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        self.map(|module| module.map(mapper))
    }
}

impl<const N: usize, T, B> ADModule<B> for [T; N]
where
    T: ADModule<B> + Debug + Send + Sync + Clone + Copy,
    T::InnerModule: Copy,
    B: ADBackend,
{
    type InnerModule = [T::InnerModule; N];

    fn inner(self) -> Self::InnerModule {
        self.map(|module| module.inner())
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module.map(|module| T::from_inner(module))
    }
}
