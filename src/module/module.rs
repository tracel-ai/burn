use crate::optim::Optimizer;
use crate::tensor::{back, DataSerialize, Gradients};
pub use burn_derive::Module;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize)]
pub struct State<B: back::Backend>
where
    B::Elem: Serialize,
    B::Elem: DeserializeOwned,
{
    root: String,
    pub values: HashMap<String, DataSerialize<B::Elem>>,
}

impl<B: back::Backend> State<B>
where
    B::Elem: Serialize,
    B::Elem: DeserializeOwned,
{
    pub fn new(name: &str) -> State<B> {
        Self {
            root: name.to_string(),
            values: HashMap::new(),
        }
    }

    pub fn get(&self, name: &str) -> &DataSerialize<B::Elem> {
        let key = format!("{}.{}", self.root, name);
        self.values.get(&key).expect("param with the name")
    }

    pub fn register_child(&mut self, child: Self) {
        for (key, value) in child.values.into_iter() {
            let key = format!("{}.{}", self.root, key);
            self.values.insert(key, value);
        }
    }

    pub fn register(&mut self, data: DataSerialize<B::Elem>) {
        self.values.insert(self.root.to_string(), data);
    }
}

impl<B: back::Backend> State<B>
where
    B::Elem: Serialize,
    B::Elem: DeserializeOwned,
{
    pub fn save(&self, file: &str) {
        let values = serde_json::to_string(&self).unwrap();
        std::fs::write(file, values).unwrap();
    }

    pub fn load(file: &str) -> Self {
        let values = std::fs::read_to_string(file).unwrap();
        serde_json::from_str(values.as_str()).unwrap()
    }
}

pub trait Module: Send + Sync + std::fmt::Debug + std::fmt::Display {
    type Backend: back::Backend;

    fn update_params<O: Optimizer<Self::Backend>>(&mut self, grads: &Gradients, optim: &mut O)
    where
        Self::Backend: back::ad::Backend;
    fn devices(&self) -> Vec<<Self::Backend as back::Backend>::Device>;
    fn to_device(&mut self, device: <Self::Backend as back::Backend>::Device);
    fn name(&self) -> &str;
    // fn load(self, state: State<B>) -> Self;
    fn state(&self) -> State<Self::Backend>
    where
        <Self::Backend as back::Backend>::Elem: Serialize,
        <Self::Backend as back::Backend>::Elem: DeserializeOwned;
    fn num_params(&self) -> usize;
}

pub trait Forward<In, Out> {
    fn forward(&self, input: In) -> Out;
}
