use crate::module::{ADModule, Module, ModuleMapper, ModuleVisitor};
use alloc::vec::Vec;
use burn_tensor::backend::{ADBackend, Backend};
use core::fmt::Debug;

impl<T, B> Module<B> for Option<T>
where
    T: Module<B> + Debug + Send + Sync + Clone,
    B: Backend,
{
    type Record = Option<T::Record>;

    fn devices(&self) -> Vec<<B as burn_tensor::backend::Backend>::Device> {
        if let Some(module) = self {
            return Module::<B>::devices(module);
        }

        Vec::new()
    }

    fn to_device(self, device: &<B as burn_tensor::backend::Backend>::Device) -> Self {
        self.map(|module| module.to_device(device))
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

    fn load_record(self, record: Self::Record) -> Self {
        self.zip(record)
            .map(|(module, record)| module.load_record(record))
    }

    fn into_record(self) -> Self::Record {
        self.map(Module::into_record)
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
    type Record = Vec<T::Record>;

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

    fn into_record(self) -> Self::Record {
        self.into_iter().map(Module::into_record).collect()
    }

    fn load_record(self, record: Self::Record) -> Self {
        self.into_iter()
            .zip(record.into_iter())
            .map(|(module, record)| module.load_record(record))
            .collect()
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
    T::Record: Debug,
    B: Backend,
{
    type Record = [T::Record; N];

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

    fn load_record(self, record: Self::Record) -> Self {
        self.into_iter()
            .zip(record)
            .map(|(module, record)| module.load_record(record))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    fn into_record(self) -> Self::Record {
        self.map(Module::into_record)
    }
}

impl<const N: usize, T, B> ADModule<B> for [T; N]
where
    T: ADModule<B> + Debug + Send + Sync + Clone + Copy,
    T::InnerModule: Copy + Debug,
    <T::InnerModule as Module<B::InnerBackend>>::Record: Debug,
    <T as Module<B>>::Record: Debug,
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
