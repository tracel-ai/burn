use crate::tensor::back::Backend;
use crate::tensor::Gradients;
use std::collections::HashMap;
use std::rc::Rc;

pub struct State<B: Backend> {
    root: String,
    values: Rc<HashMap<String, Vec<B::Elem>>>,
}

impl<B: Backend> State<B> {
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

pub trait Module<B: Backend>: Send + Sync + std::fmt::Debug + std::fmt::Display {
    fn update(&mut self, grads: &Gradients);
    fn get_devices(&self) -> Vec<B::Device>;
    fn to_device(self, device: B::Device) -> Self;
    fn state(&self) -> State<B>;
    fn load(self, state: State<B>) -> Self;
}

pub trait Forward<In, Out> {
    fn forward(&self, input: In) -> Out;
}
