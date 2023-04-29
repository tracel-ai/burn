use std::{marker::PhantomData, path::Path};

use crate::onnx::{ir::TensorData, op_configuration::linear_config};

use super::{
    from_onnx::parse_onnx,
    ir::{self, ArgType, Graph, Node, NodeType},
    op_configuration::{batch_norm_config, conv2d_config},
    shape_inference::first_input_dim,
};

use burn::{
    module::{Module, Param},
    nn::{
        conv::{Conv2dRecord, Conv2dRecordItem},
        BatchNormRecordItem, LinearRecord, LinearRecordItem,
    },
    record::{Record, RecordSettings},
    tensor::Tensor,
};

use burn_ndarray::NdArrayBackend;

use serde::{
    de::Deserializer,
    ser::{SerializeMap, Serializer},
    Deserialize, Serialize,
};

type B = NdArrayBackend<f32>;

pub struct ModelState<RS: RecordSettings> {
    pub graph: Graph,
    _record_settings: PhantomData<RS>,
}

impl<RS: RecordSettings> ModelState<RS> {
    /// Create a new model state from the onnx file
    pub fn new<P: AsRef<Path>>(onnx_path: P) -> Self {
        let graph = parse_onnx(onnx_path.as_ref());
        Self {
            graph,
            _record_settings: PhantomData::default(),
        }
    }

    pub fn new_from_graph(graph: Graph) -> Self {
        Self {
            graph,
            _record_settings: PhantomData::default(),
        }
    }
}

impl<RS: RecordSettings> Record for ModelState<RS> {
    type Item<S: RecordSettings> = ModelState<RS>;

    fn into_item<S: RecordSettings>(self) -> Self::Item<S> {
        self
    }

    fn from_item<S: RecordSettings>(_item: Self::Item<S>) -> Self {
        unimplemented!()
    }
}

impl<RS: RecordSettings> Serialize for ModelState<RS> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Collect all the stateful nodes
        let stateful_nodes: Vec<_> = self
            .graph
            .nodes
            .iter()
            .filter(|node| node.is_stateful)
            .collect();

        // Create a map to serialize the stateful nodes
        let mut map = serializer.serialize_map(Some(stateful_nodes.len()))?;

        // Serialize each stateful node into a map entry
        for node in stateful_nodes.into_iter() {
            match node.node_type {
                NodeType::Conv2d => {
                    let (name, record_item) = conv2d_state::<RS>(node);
                    map.serialize_entry(&name, &record_item)?;
                }
                NodeType::Linear => {
                    let (name, record_item) = linear_state::<RS>(node);
                    map.serialize_entry(&name, &record_item)?;
                }
                NodeType::BatchNormalization => match first_input_dim(node).unwrap() {
                    2 => {
                        let (name, record_item) = batch_norm_state::<0, RS>(node);
                        map.serialize_entry(&name, &record_item)?;
                    }
                    3 => {
                        let (name, record_item) = batch_norm_state::<1, RS>(node);
                        map.serialize_entry(&name, &record_item)?;
                    }
                    4 => {
                        let (name, record_item) = batch_norm_state::<2, RS>(node);
                        map.serialize_entry(&name, &record_item)?;
                    }
                    5 => {
                        let (name, record_item) = batch_norm_state::<3, RS>(node);
                        map.serialize_entry(&name, &record_item)?;
                    }
                    dim => todo!("BatchNorm for dim = {dim} is not implemented yet"),
                },

                _ => todo!(
                    "Serialize state for node type {:?} is not implemented yet",
                    node.node_type
                ),
            }
        }
        map.end()
    }
}

/// Dummy implementation of Deserialize for ModelState
impl<'de, RS: RecordSettings> Deserialize<'de> for ModelState<RS> {
    fn deserialize<D>(_deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        unimplemented!() // No intention to deserialize a ModelState
    }
}

/// Convert itermediate representation of tensor into a burn tensor
///
/// TODO: implemenate for all tensor element types
impl<const D: usize> TryFrom<&ir::Tensor> for Tensor<B, D> {
    type Error = ();

    fn try_from(value: &ir::Tensor) -> Result<Self, Self::Error> {
        let shape: [usize; D] = value.shape.clone().try_into().unwrap();
        let TensorData::Float32s(floats) = value.data.clone().unwrap() else {
            todo!("Tensor data must be float32s");
        };
        let tensor: Tensor<B, D> = Tensor::from_data(floats.as_slice()).reshape(shape);
        Ok(tensor)
    }
}

/// Convert a Conv2d node into a Name, Conv2dRecordItem pair
pub fn conv2d_state<RS: RecordSettings>(node: &Node) -> (String, Conv2dRecordItem<B, RS>) {
    if node.initializers.is_empty() {
        panic!("Conv2d node must have at least 1 initializer");
    }

    let config = conv2d_config(node);
    let ArgType::Tensor(node_weight) = node.initializers[0].arg_type.as_ref().unwrap();

    let weight: Tensor<B, 4> = node_weight.try_into().unwrap();

    let bias = if node.initializers.len() == 2 {
        let ArgType::Tensor(node_bias) = node.initializers[1].arg_type.as_ref().unwrap();
        let bias: Tensor<B, 1> = node_bias.try_into().unwrap();
        Some(Param::from(bias))
    } else {
        None
    };

    // Create a Conv2dRecord from the config
    let conv2d = config.init();
    let mut record: Conv2dRecord<B> = conv2d.into_record();
    record.weight = Param::from(weight);
    record.bias = bias;

    let item = Record::into_item::<RS>(record);

    (node.name.clone(), item)
}

/// Convert a Linear node into a Name, LinearRecordItem pair
///
/// TODO: implement for all tensor element types
fn linear_state<RS: RecordSettings>(node: &Node) -> (String, LinearRecordItem<B, RS>) {
    if node.initializers.is_empty() {
        panic!("Linear node must have at least 1 initializer");
    }
    let ArgType::Tensor(node_weight) = node.initializers[0].arg_type.as_ref().unwrap();

    let weight: Tensor<B, 2> = node_weight.try_into().unwrap();

    let bias = if node.initializers.len() == 2 {
        let ArgType::Tensor(node_bias) = node.initializers[1].arg_type.as_ref().unwrap();
        let bias: Tensor<B, 1> = node_bias.try_into().unwrap();
        Some(Param::from(bias))
    } else {
        None
    };

    // Create a LinearRecord from the config
    let config = linear_config(node);
    let linear = config.init();
    let mut record: LinearRecord<B> = linear.into_record();
    record.weight = Param::from(weight);
    record.bias = bias;

    let item = Record::into_item::<RS>(record);
    (node.name.clone(), item)
}

/// Convert a BatchNorm node into a Name, BatchNormRecordItem pair
fn batch_norm_state<const D: usize, RS: RecordSettings>(
    node: &Node,
) -> (String, BatchNormRecordItem<B, D, RS>) {
    let config = batch_norm_config(node);
    let norm: burn::nn::BatchNorm<B, D> = config.init();
    let mut record = norm.into_record();

    // weight
    record.gamma = {
        let ArgType::Tensor(node_gamma) = node.initializers[0].arg_type.as_ref().unwrap();
        let gamma: Tensor<B, 1> = node_gamma.try_into().unwrap();
        Param::from(gamma)
    };

    // bias
    record.beta = {
        let ArgType::Tensor(node_beta) = node.initializers[1].arg_type.as_ref().unwrap();
        let beta: Tensor<B, 1> = node_beta.try_into().unwrap();
        Param::from(beta)
    };

    record.running_mean = {
        let ArgType::Tensor(node_mean) = node.initializers[2].arg_type.as_ref().unwrap();
        let mean: Tensor<B, 1> = node_mean.try_into().unwrap();
        Param::from(mean)
    };

    record.running_var = {
        let ArgType::Tensor(node_var) = node.initializers[3].arg_type.as_ref().unwrap();
        let var: Tensor<B, 1> = node_var.try_into().unwrap();
        Param::from(var)
    };

    let item = Record::into_item::<RS>(record);
    (node.name.clone(), item)
}
