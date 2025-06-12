use burn_common::stub::Mutex;
use burn_ndarray::NdArrayTensor;
use burn_tensor::{Tensor, backend::Backend};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
    sync::mpsc::{self, Receiver, Sender},
};

use crate::cluster::{Aggregator, AggregatorClient, ClusterMetadata, ClusterOps};

static STATE: Mutex<Option<HashMap<TypeId, Box<dyn Any + Send + Sync>>>> = Mutex::new(None);

pub fn aggregator<B: Backend>() -> AggregatorClient<B> {
    let mut state = STATE.lock().unwrap();

    let hashmap = match state {
        Some(val) => val,
        None => {
            *state = Some(HashMap::new());
            core::
        }
    };

    let typeid = core::any::TypeId::of::<B>();

    let val = match state.get(&typeid) {
        Some(val) => val,
        None => {
            let client = Aggregator::start();
            state.insert(typeid, Box::new(client.clone()));
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
    let client = aggregator();
    let device = tensor.device();
    let primitive = client.register(tensor.into_primitive());

    Tensor::from_primitive(primitive).to_device(device)
}
