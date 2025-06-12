use burn_common::stub::Mutex;
use burn_tensor::{Tensor, backend::Backend};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

use crate::aggregator::{Aggregator, AggregatorClient};

static STATE: Mutex<Option<HashMap<TypeId, Box<dyn Any + Send + Sync>>>> = Mutex::new(None);

pub fn aggregator<B: Backend>() -> AggregatorClient<B> {
    let mut state = STATE.lock().unwrap();

    if state.is_none() {
        *state = Some(HashMap::new());
    }
    let hashmap = state.as_mut().unwrap();

    let typeid = core::any::TypeId::of::<B>();

    let val = match hashmap.get(&typeid) {
        Some(val) => val,
        None => {
            let client = Aggregator::start();
            hashmap.insert(typeid, Box::new(client.clone()));
            return client;
        }
    };

    val.downcast_ref().cloned().unwrap()
}

pub fn register<B: Backend>(num_nodes: u32) {
    let client = aggregator::<B>();
    client.register(num_nodes);
}

pub fn collective_sum<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let client: AggregatorClient<B> = aggregator();
    let device = tensor.device();
    if let burn_tensor::TensorPrimitive::Float(tensor) = tensor.into_primitive() {
        let primitive = client.aggregate(tensor);
        Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(primitive)).to_device(&device)
    } else {
        unimplemented!("qfloat unsupported");
    }
}
