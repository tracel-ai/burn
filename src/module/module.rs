use crate::optim::Optimizer;
use crate::tensor::{back, DataSerialize, Gradients};
pub use burn_derive::Module;
use std::collections::HashMap;

#[derive(Debug)]
pub struct StateNamed<B: back::Backend> {
    pub values: HashMap<String, State<B>>,
}

#[derive(Debug)]
pub enum State<B: back::Backend> {
    StateNamed(StateNamed<B>),
    Data(DataSerialize<B::Elem>),
}

impl<B: back::Backend> StateNamed<B> {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn register_state(&mut self, name: &str, state: State<B>) {
        self.values.insert(name.to_string(), state);
    }
}

impl<B: back::Backend> StateNamed<B> {
    pub fn get(&self, name: &str) -> &State<B> {
        self.values.get(name).unwrap()
    }
}

impl<B: back::Backend> State<B> {
    pub fn get(&self, name: &str) -> &Self {
        match self {
            State::StateNamed(named) => named.get(name),
            _ => panic!("Can't"),
        }
    }
}

impl<B: back::Backend> State<B> {
    pub fn save(&self, _file: &str) {
        // let values = serde_json::to_string(&self).unwrap();
        // std::fs::write(file, values).unwrap();
        todo!()
    }

    pub fn load(_file: &str) -> Self {
        // let values = std::fs::read_to_string(file).unwrap();
        // serde_json::from_str(values.as_str()).unwrap()
        todo!()
    }
}

pub trait Module: Send + Sync + std::fmt::Debug + std::fmt::Display {
    type Backend: back::Backend;

    fn update_params<O: Optimizer<Backend = Self::Backend>>(
        &mut self,
        grads: &Gradients,
        optim: &mut O,
    ) where
        Self::Backend: back::ad::Backend;
    fn devices(&self) -> Vec<<Self::Backend as back::Backend>::Device>;
    fn to_device(&mut self, device: <Self::Backend as back::Backend>::Device);
    fn name(&self) -> &str;
    fn load(&mut self, state: &State<Self::Backend>);
    fn state(&self) -> State<Self::Backend>;
    fn num_params(&self) -> usize;
}

pub trait ADModule: Module + Send + Sync + std::fmt::Debug + std::fmt::Display {
    type ADBackend: back::ad::Backend;
    type InnerModule: Module<Backend = <Self::ADBackend as back::ad::Backend>::InnerBackend>;

    fn inner(&self) -> Self::InnerModule;
}

pub trait Forward<In, Out> {
    fn forward(&self, input: In) -> Out;
}
