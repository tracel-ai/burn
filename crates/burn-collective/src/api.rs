use burn_tensor::{Tensor, backend::Backend};

use crate::aggregator::{AggregatorClient, aggregator};

#[derive(Debug, PartialEq, Clone)]
pub enum AggregateStrategy {
    Centralized,
    Tree(u32),
    Ring,
}

#[derive(Debug, PartialEq, Clone)]
pub enum AggregateKind {
    Sum,
    Mean,
}

#[derive(Debug, PartialEq, Clone)]
pub struct AggregateParams {
    pub kind: AggregateKind,
    pub strategy: AggregateStrategy,
}

/// Resets the aggregator. All registered callers and ongoing operations are forgotten
pub fn reset_collective<B: Backend>() {
    let client = aggregator::<B>();
    client.reset();
}

/// Registers a "node". `num_nodes` must be the same as the other calls to register,
/// and `id` must be unique.
pub fn register<B: Backend>(id: u32, num_nodes: u32) {
    let client = aggregator::<B>();
    client.register(id, num_nodes);
}

/// Calls for an all-reduce operation with the given parameters, and returns the result.
/// The `params` must be the same as the parameters passed by the other nodes.
pub fn all_reduce<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    params: AggregateParams,
) -> Tensor<B, D> {
    let client: AggregatorClient<B> = aggregator();
    let device = tensor.device();
    let tensor = tensor.into_primitive().tensor();
    let primitive = client.aggregate(tensor, params);
    Tensor::from_primitive(burn_tensor::TensorPrimitive::Float(primitive)).to_device(&device)
}
