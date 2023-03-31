// use alloc::{format, vec::Vec};
// use burn_tensor::backend::ADBackend;
//
// use super::{load_with_id, state_with_id, Param, ParamId};
// use crate::module::{
//     ADModule, LoadingError, Module, ModuleMapper, ModuleVisitor, State, StateNamed,
// };
// use crate::tensor::backend::Backend;

// impl<M> From<M> for Param<M> {
//     fn from(value: M) -> Self {
//         Param {
//             id: ParamId::new(),
//             value,
//         }
//     }
// }
//
// impl<M> From<Vec<M>> for Param<Vec<M>> {
//     fn from(value: Vec<M>) -> Self {
//         Param {
//             id: ParamId::new(),
//             value,
//         }
//     }
// }

// impl<M, B> Module<B> for Param<M>
// where
//     M: Module<B>,
//     B: Backend,
// {
//     fn num_params(&self) -> usize {
//         self.value.num_params()
//     }
//
//     fn devices(&self) -> Vec<B::Device> {
//         self.value.devices()
//     }
//
//     fn to_device(self, device: &B::Device) -> Self {
//         Param {
//             id: self.id,
//             value: self.value.to_device(device),
//         }
//     }
//
//     fn state(&self) -> State<B::FloatElem> {
//         let state = self.value.state();
//
//         state_with_id(self.id.clone(), state)
//     }
//
//     fn load(self, state: &State<B::FloatElem>) -> Result<Self, LoadingError> {
//         let (id, state) = load_with_id(state)?;
//
//         Ok(Self {
//             id: id.clone(),
//             value: self.value.load(state)?,
//         })
//     }
//
//     fn detach(self) -> Self {
//         Param {
//             id: self.id,
//             value: self.value.detach(),
//         }
//     }
//
//     fn visit<V: ModuleVisitor<Self, B>>(&self, visitor: &mut V) {
//         self.value.visit(visitor);
//     }
//
//     fn map<V: ModuleMapper<Self, B>>(self, mapper: &mut V) -> Self {
//         Self {
//             id: self.id,
//             value: self.value.map(mapper),
//         }
//     }
// }
//
// impl<M, B> Module<B> for Param<Vec<M>>
// where
//     B: Backend,
//     M: Module<B>,
// {
//     fn num_params(&self) -> usize {
//         let mut num_params = 0;
//         for module in self.value.iter() {
//             num_params += module.num_params();
//         }
//
//         num_params
//     }
//
//     fn devices(&self) -> Vec<B::Device> {
//         let mut devices = Vec::new();
//         for module in self.value.iter() {
//             devices.append(&mut module.devices());
//         }
//         devices
//     }
//
//     fn to_device(self, device: &B::Device) -> Self {
//         Param {
//             id: self.id,
//             value: self
//                 .value
//                 .into_iter()
//                 .map(|val| val.to_device(device))
//                 .collect(),
//         }
//     }
//
//     fn state(&self) -> State<B::FloatElem> {
//         let mut state = StateNamed::new();
//
//         for (i, module) in self.value.iter().enumerate() {
//             state.register_state(format!("mod-{i}").as_str(), module.state());
//         }
//
//         let state = State::StateNamed(state);
//
//         state_with_id(self.id.clone(), state)
//     }
//
//     fn load(self, state: &State<B::FloatElem>) -> Result<Self, LoadingError> {
//         let (id, state) = load_with_id(state)?;
//         let id = id.clone();
//
//         let num = self.value.len();
//         let mut modules = Vec::with_capacity(num);
//
//         for (i, module) in self.value.into_iter().enumerate() {
//             let module = module
//                 .load(state.get(format!("mod-{i}").as_str()).ok_or_else(|| {
//                     LoadingError::new(format!(
//                         "Invalid number of modules, expected {num} modules missing #{i}"
//                     ))
//                 })?)
//                 .map_err(|err| LoadingError::new(format!("Can't load modules mod-{i}: {err}")))?;
//
//             modules.push(module);
//         }
//
//         Ok(Self { id, value: modules })
//     }
//
//     fn detach(self) -> Self {
//         Param {
//             id: self.id,
//             value: self.value.into_iter().map(|val| val.detach()).collect(),
//         }
//     }
//
//     fn visit<V: ModuleVisitor<Self, B>>(&self, visitor: &mut V) {
//         for module in self.value.iter() {
//             module.visit(visitor);
//         }
//     }
//
//     fn map<V: ModuleMapper<Self, B>>(self, mapper: &mut V) -> Self {
//         Self {
//             id: self.id,
//             value: self.value.into_iter().map(|val| val.map(mapper)).collect(),
//         }
//     }
// }
//
// impl<M, B> ADModule<B> for Param<Vec<M>>
// where
//     B: ADBackend,
//     M: ADModule<B>,
// {
//     type InnerModule = Param<Vec<M::InnerModule>>;
//
//     fn inner(self) -> Self::InnerModule {
//         Param::from(
//             self.value
//                 .into_iter()
//                 .map(|v| v.inner())
//                 .collect::<Vec<_>>(),
//         )
//     }
//
//     fn from_inner(module: Self::InnerModule) -> Self {
//         Param {
//             id: module.id,
//             value: module.value.into_iter().map(ADModule::from_inner).collect(),
//         }
//     }
// }
//
// impl<M, B> ADModule<B> for Param<M>
// where
//     B: ADBackend,
//     M: ADModule<B>,
// {
//     type InnerModule = Param<M::InnerModule>;
//
//     fn inner(self) -> Self::InnerModule {
//         Param::from(self.value.inner())
//     }
//
//     fn from_inner(module: Self::InnerModule) -> Self {
//         Param {
//             id: module.id,
//             value: ADModule::from_inner(module.value),
//         }
//     }
// }
