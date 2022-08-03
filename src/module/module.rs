pub use burn_derive::Module;

use crate::optim::Optimizer;
use crate::tensor::back;
use crate::tensor::Gradients;
use std::collections::HashMap;
use std::rc::Rc;

pub struct State<B: back::Backend> {
    root: String,
    values: Rc<HashMap<String, Vec<B::Elem>>>,
}

impl<B: back::Backend> State<B> {
    pub fn new(name: &str) -> State<B> {
        Self {
            root: name.to_string(),
            values: Rc::new(HashMap::new()),
        }
    }

    pub fn get(&self, name: &str) -> &Vec<B::Elem> {
        let key = format!("{}.{}", self.root, name);
        self.values.get(&key).expect("param with the name")
    }

    pub fn with_name(&self, name: &str) -> State<B> {
        Self {
            root: format!("{}.{}", self.root, name),
            values: self.values.clone(),
        }
    }
}

pub trait Module<B: back::Backend>: Send + Sync + std::fmt::Debug + std::fmt::Display {
    // fn update_params<O: Optimizer<B>>(&mut self, grads: &Gradients, optim: &mut O)
    // where
    //     B: back::ad::Backend;
    // fn get_devices(&self) -> Vec<B::Device>;
    // fn to_device(self, device: B::Device) -> Self;
    // fn state(&self) -> State<B>;
    // fn load(self, state: State<B>) -> Self;
    fn num_params(&self) -> usize;
    fn save(&self);
}

pub trait Forward<In, Out> {
    fn forward(&self, input: In) -> Out;
}
